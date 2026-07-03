#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeCAD 插件：Mesh 孔洞圆弧检测与重建

功能：
1. 检测 Mesh 中的孔洞边界
2. 识别圆弧形边界（使用 AnalysisSitus 的圆弧拟合算法）
3. 重建圆弧形孔洞

依赖：
- FreeCAD 0.20+
- AnalysisSitus（编译后的库文件）
- numpy
"""

import os
import sys
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# FreeCAD imports
import FreeCAD
import FreeCADGui
import Mesh
import MeshPart
import Part
from FreeCAD import Base

# PySide imports for GUI
from PySide2 import QtWidgets, QtCore, QtGui


# ============================================================================
# 数据结构
# ============================================================================

class HoleType(Enum):
    """孔洞类型"""
    CIRCLE = "circle"           # 圆形孔洞
    ARC = "arc"                 # 圆弧形孔洞
    ELLIPSE = "ellipse"         # 椭圆形孔洞
    POLYGON = "polygon"         # 多边形孔洞
    UNKNOWN = "unknown"         # 未知类型


@dataclass
class BoundaryLoop:
    """边界环数据结构"""
    vertex_indices: List[int]       # 顶点索引列表
    edge_indices: List[Tuple[int, int]]  # 边列表（顶点对）
    perimeter: float                # 周长
    area: float                     # 面积（如果闭合）
    centroid: Base.Vector           # 质心
    hole_type: HoleType             # 孔洞类型
    fit_circle: Optional[Any]       # 拟合的圆（如果有）
    fit_error: float                # 拟合误差


@dataclass
class ArcDetectionResult:
    """圆弧检测结果"""
    loop: BoundaryLoop              # 边界环
    circle_center: Base.Vector      # 圆心
    circle_radius: float            # 半径
    circle_normal: Base.Vector      # 法向量
    arc_start: Base.Vector          # 圆弧起点
    arc_end: Base.Vector            # 圆弧终点
    arc_angle: float                # 圆弧角度（弧度）
    deviation: float                # 偏差
    is_full_circle: bool            # 是否完整圆


# ============================================================================
# 核心算法：边界检测
# ============================================================================

class BoundaryDetector:
    """边界检测器"""
    
    def __init__(self, mesh: Mesh.Mesh):
        self.mesh = mesh
        self.vertices = mesh.Topology[0]
        self.faces = mesh.Topology[1]
        self._edge_face_map = None
        self._boundary_edges = None
    
    def _build_edge_face_map(self) -> Dict[Tuple[int, int], List[int]]:
        """构建边到面的映射"""
        if self._edge_face_map is not None:
            return self._edge_face_map
        
        edge_face_map = {}
        for face_idx, face in enumerate(self.faces):
            for i in range(3):
                v1 = face[i]
                v2 = face[(i + 1) % 3]
                edge = (min(v1, v2), max(v1, v2))
                if edge not in edge_face_map:
                    edge_face_map[edge] = []
                edge_face_map[edge].append(face_idx)
        
        self._edge_face_map = edge_face_map
        return edge_face_map
    
    def find_boundary_edges(self) -> List[Tuple[int, int]]:
        """找到所有边界边（只属于一个面的边）"""
        if self._boundary_edges is not None:
            return self._boundary_edges
        
        edge_face_map = self._build_edge_face_map()
        boundary_edges = []
        
        for edge, faces in edge_face_map.items():
            if len(faces) == 1:  # 边界边只属于一个面
                boundary_edges.append(edge)
        
        self._boundary_edges = boundary_edges
        return boundary_edges
    
    def extract_boundary_loops(self) -> List[BoundaryLoop]:
        """提取所有边界环"""
        boundary_edges = self.find_boundary_edges()
        
        if not boundary_edges:
            return []
        
        # 构建邻接表
        adjacency = {}
        for v1, v2 in boundary_edges:
            if v1 not in adjacency:
                adjacency[v1] = []
            if v2 not in adjacency:
                adjacency[v2] = []
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
        
        # 提取环
        loops = []
        visited_edges = set()
        
        for start_edge in boundary_edges:
            if start_edge in visited_edges:
                continue
            
            # 开始一个新的环
            loop_vertices = []
            loop_edges = []
            current = start_edge[0]
            target = start_edge[1]
            
            while True:
                loop_vertices.append(current)
                loop_edges.append((current, target))
                visited_edges.add((min(current, target), max(current, target)))
                
                current = target
                # 找到下一个未访问的邻接顶点
                next_vertex = None
                for neighbor in adjacency.get(current, []):
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge not in visited_edges:
                        next_vertex = neighbor
                        break
                
                if next_vertex is None:
                    break
                
                target = next_vertex
                if current == start_edge[0] and target == start_edge[1]:
                    break  # 回到起点
            
            if len(loop_vertices) >= 3:
                # 计算环的属性
                loop = self._analyze_loop(loop_vertices, loop_edges)
                loops.append(loop)
        
        return loops
    
    def _analyze_loop(self, vertices: List[int], edges: List[Tuple[int, int]]) -> BoundaryLoop:
        """分析边界环的属性"""
        # 获取顶点坐标
        points = [Base.Vector(*self.vertices[v]) for v in vertices]
        
        # 计算周长
        perimeter = 0.0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            perimeter += p1.distanceToPoint(p2)
        
        # 计算质心
        centroid = Base.Vector(0, 0, 0)
        for p in points:
            centroid += p
        centroid /= len(points)
        
        # 计算面积（使用 Shoelace 公式，假设在 XY 平面投影）
        area = self._compute_loop_area(points)
        
        return BoundaryLoop(
            vertex_indices=vertices,
            edge_indices=edges,
            perimeter=perimeter,
            area=area,
            centroid=centroid,
            hole_type=HoleType.UNKNOWN,
            fit_circle=None,
            fit_error=float('inf')
        )
    
    def _compute_loop_area(self, points: List[Base.Vector]) -> float:
        """计算环的面积（投影到最佳拟合平面）"""
        if len(points) < 3:
            return 0.0
        
        # 计算法向量
        normal = self._compute_normal(points)
        
        # 投影到 2D
        projected = self._project_to_2d(points, normal)
        
        # Shoelace 公式
        n = len(projected)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += projected[i][0] * projected[j][1]
            area -= projected[j][0] * projected[i][1]
        return abs(area) / 2.0
    
    def _compute_normal(self, points: List[Base.Vector]) -> Base.Vector:
        """计算点集的法向量"""
        if len(points) < 3:
            return Base.Vector(0, 0, 1)
        
        # 使用 Newell 方法计算法向量
        normal = Base.Vector(0, 0, 0)
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            normal.x += (p1.y - p2.y) * (p1.z + p2.z)
            normal.y += (p1.z - p2.z) * (p1.x + p2.x)
            normal.z += (p1.x - p2.x) * (p1.y + p2.y)
        
        length = normal.Length
        if length > 1e-10:
            normal /= length
        else:
            normal = Base.Vector(0, 0, 1)
        
        return normal
    
    def _project_to_2d(self, points: List[Base.Vector], normal: Base.Vector) -> List[Tuple[float, float]]:
        """将 3D 点投影到 2D 平面"""
        # 选择投影平面
        if abs(normal.z) > abs(normal.x) and abs(normal.z) > abs(normal.y):
            # 投影到 XY 平面
            return [(p.x, p.y) for p in points]
        elif abs(normal.x) > abs(normal.y):
            # 投影到 YZ 平面
            return [(p.y, p.z) for p in points]
        else:
            # 投影到 XZ 平面
            return [(p.x, p.z) for p in points]


# ============================================================================
# 核心算法：圆弧检测
# ============================================================================

class ArcDetector:
    """圆弧检测器（使用 AnalysisSitus 的算法思想）"""
    
    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance
    
    def detect_circle(self, loop: BoundaryLoop, vertices: List[Tuple[float, float, float]]) -> Optional[ArcDetectionResult]:
        """检测边界环是否为圆弧"""
        if len(loop.vertex_indices) < 3:
            return None
        
        # 获取点坐标
        points = [Base.Vector(*vertices[v]) for v in loop.vertex_indices]
        
        # 计算最佳拟合圆
        circle = self._fit_circle_3d(points)
        if circle is None:
            return None
        
        center, radius, normal = circle
        
        # 计算偏差
        max_deviation = 0.0
        for p in points:
            dist = p.distanceToPoint(center)
            deviation = abs(dist - radius)
            max_deviation = max(max_deviation, deviation)
        
        # 检查是否在容差范围内
        if max_deviation > self.tolerance:
            return None
        
        # 计算圆弧角度
        arc_angle = self._compute_arc_angle(points, center, normal)
        
        # 判断是否为完整圆
        is_full_circle = abs(arc_angle - 2 * math.pi) < 0.1
        
        return ArcDetectionResult(
            loop=loop,
            circle_center=center,
            circle_radius=radius,
            circle_normal=normal,
            arc_start=points[0],
            arc_end=points[-1],
            arc_angle=arc_angle,
            deviation=max_deviation,
            is_full_circle=is_full_circle
        )
    
    def _fit_circle_3d(self, points: List[Base.Vector]) -> Optional[Tuple[Base.Vector, float, Base.Vector]]:
        """3D 圆拟合（最小二乘法）"""
        if len(points) < 3:
            return None
        
        # 计算质心
        centroid = Base.Vector(0, 0, 0)
        for p in points:
            centroid += p
        centroid /= len(points)
        
        # 计算法向量（使用 SVD）
        matrix = []
        for p in points:
            diff = p - centroid
            matrix.append([diff.x, diff.y, diff.z])
        
        matrix = np.array(matrix)
        _, _, vh = np.linalg.svd(matrix)
        normal = Base.Vector(*vh[2])
        
        # 投影到 2D 平面
        projected = []
        for p in points:
            diff = p - centroid
            # 使用法向量构建局部坐标系
            u = self._orthogonal_vector(normal)
            v = normal.cross(u)
            u.normalize()
            v.normalize()
            
            x = diff.dot(u)
            y = diff.dot(v)
            projected.append((x, y))
        
        # 2D 圆拟合
        circle_2d = self._fit_circle_2d(projected)
        if circle_2d is None:
            return None
        
        cx, cy, radius = circle_2d
        
        # 转换回 3D
        u = self._orthogonal_vector(normal)
        v = normal.cross(u)
        u.normalize()
        v.normalize()
        
        center_3d = centroid + u * cx + v * cy
        
        return center_3d, radius, normal
    
    def _fit_circle_2d(self, points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
        """2D 圆拟合（最小二乘法）"""
        if len(points) < 3:
            return None
        
        # 构建线性方程组
        # (x - cx)^2 + (y - cy)^2 = r^2
        # x^2 - 2*cx*x + cx^2 + y^2 - 2*cy*y + cy^2 = r^2
        # -2*cx*x - 2*cy*y + (cx^2 + cy^2 - r^2) = -(x^2 + y^2)
        
        A = []
        b = []
        for x, y in points:
            A.append([-2*x, -2*y, 1])
            b.append(-(x*x + y*y))
        
        A = np.array(A)
        b = np.array(b)
        
        # 最小二乘求解
        try:
            result = np.linalg.lstsq(A, b, rcond=None)
            params = result[0]
            cx = params[0]
            cy = params[1]
            radius = math.sqrt(cx*cx + cy*cy - params[2])
            return cx, cy, radius
        except:
            return None
    
    def _orthogonal_vector(self, v: Base.Vector) -> Base.Vector:
        """计算与给定向量正交的向量"""
        if abs(v.x) < 0.9:
            return Base.Vector(1, 0, 0).cross(v).normalize()
        else:
            return Base.Vector(0, 1, 0).cross(v).normalize()
    
    def _compute_arc_angle(self, points: List[Base.Vector], center: Base.Vector, normal: Base.Vector) -> float:
        """计算圆弧角度"""
        if len(points) < 2:
            return 0.0
        
        # 计算到圆心的向量
        vectors = []
        for p in points:
            v = p - center
            v.normalize()
            vectors.append(v)
        
        # 计算角度和
        total_angle = 0.0
        for i in range(len(vectors)):
            v1 = vectors[i]
            v2 = vectors[(i + 1) % len(vectors)]
            dot = v1.dot(v2)
            dot = max(-1.0, min(1.0, dot))
            angle = math.acos(dot)
            
            # 检查方向
            cross = v1.cross(v2)
            if cross.dot(normal) < 0:
                angle = -angle
            
            total_angle += angle
        
        return abs(total_angle)


# ============================================================================
# 核心算法：孔洞重建
# ============================================================================

class HoleRebuilder:
    """孔洞重建器"""
    
    def __init__(self, mesh: Mesh.Mesh):
        self.mesh = mesh
        self.vertices = list(mesh.Topology[0])
        self.faces = list(mesh.Topology[1])
    
    def rebuild_circle_hole(self, result: ArcDetectionResult, 
                           num_segments: int = 32) -> Mesh.Mesh:
        """重建圆形孔洞"""
        center = result.circle_center
        radius = result.circle_radius
        normal = result.circle_normal
        
        # 创建局部坐标系
        u = self._orthogonal_vector(normal)
        v = normal.cross(u)
        u.normalize()
        v.normalize()
        
        # 生成圆上的点
        circle_points = []
        for i in range(num_segments):
            angle = 2 * math.pi * i / num_segments
            point = center + u * (radius * math.cos(angle)) + v * (radius * math.sin(angle))
            circle_points.append(point)
        
        # 获取边界点
        boundary_points = [Base.Vector(*self.vertices[v]) for v in result.loop.vertex_indices]
        
        # 创建网格
        new_mesh = Mesh.Mesh()
        
        # 添加中心点
        center_idx = len(self.vertices)
        self.vertices.append((center.x, center.y, center.z))
        
        # 添加边界点
        boundary_start_idx = len(self.vertices)
        for p in boundary_points:
            self.vertices.append((p.x, p.y, p.z))
        
        # 创建扇形三角形
        for i in range(len(boundary_points)):
            j = (i + 1) % len(boundary_points)
            v1 = boundary_start_idx + i
            v2 = boundary_start_idx + j
            v3 = center_idx
            self.faces.append((v1, v2, v3))
        
        # 创建新网格
        new_mesh.addMesh(self.vertices, self.faces)
        return new_mesh
    
    def rebuild_arc_hole(self, result: ArcDetectionResult,
                        num_segments: int = 32) -> Mesh.Mesh:
        """重建圆弧形孔洞"""
        # 对于圆弧形孔洞，使用扇形填充
        return self._fan_triangulation(result.loop)
    
    def _fan_triangulation(self, loop: BoundaryLoop) -> Mesh.Mesh:
        """扇形三角化"""
        boundary_points = [Base.Vector(*self.vertices[v]) for v in loop.vertex_indices]
        
        if len(boundary_points) < 3:
            return self.mesh
        
        # 计算质心作为填充点
        centroid = Base.Vector(0, 0, 0)
        for p in boundary_points:
            centroid += p
        centroid /= len(boundary_points)
        
        # 添加质心点
        center_idx = len(self.vertices)
        self.vertices.append((centroid.x, centroid.y, centroid.z))
        
        # 添加边界点
        boundary_start_idx = len(self.vertices)
        for p in boundary_points:
            self.vertices.append((p.x, p.y, p.z))
        
        # 创建扇形三角形
        for i in range(len(boundary_points)):
            j = (i + 1) % len(boundary_points)
            v1 = boundary_start_idx + i
            v2 = boundary_start_idx + j
            v3 = center_idx
            self.faces.append((v1, v2, v3))
        
        # 创建新网格
        new_mesh = Mesh.Mesh()
        new_mesh.addMesh(self.vertices, self.faces)
        return new_mesh
    
    def _orthogonal_vector(self, v: Base.Vector) -> Base.Vector:
        """计算与给定向量正交的向量"""
        if abs(v.x) < 0.9:
            return Base.Vector(1, 0, 0).cross(v).normalize()
        else:
            return Base.Vector(0, 1, 0).cross(v).normalize()


# ============================================================================
# GUI 对话框
# ============================================================================

class HoleRepairDialog(QtWidgets.QDialog):
    """孔洞修复对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mesh 孔洞圆弧检测与重建")
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        
        # 数据
        self.mesh_obj = None
        self.boundary_loops = []
        self.arc_results = []
        
        # 创建 UI
        self._create_ui()
        
        # 连接信号
        self._connect_signals()
    
    def _create_ui(self):
        """创建用户界面"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # 网格选择组
        mesh_group = QtWidgets.QGroupBox("网格选择")
        mesh_layout = QtWidgets.QHBoxLayout()
        
        self.mesh_combo = QtWidgets.QComboBox()
        self.mesh_combo.setMinimumWidth(300)
        mesh_layout.addWidget(QtWidgets.QLabel("选择网格对象:"))
        mesh_layout.addWidget(self.mesh_combo)
        
        self.refresh_btn = QtWidgets.QPushButton("刷新")
        mesh_layout.addWidget(self.refresh_btn)
        
        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)
        
        # 参数设置组
        param_group = QtWidgets.QGroupBox("检测参数")
        param_layout = QtWidgets.QFormLayout()
        
        self.tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.tolerance_spin.setRange(0.001, 10.0)
        self.tolerance_spin.setValue(0.1)
        self.tolerance_spin.setSingleStep(0.01)
        param_layout.addRow("圆弧检测容差:", self.tolerance_spin)
        
        self.min_vertices_spin = QtWidgets.QSpinBox()
        self.min_vertices_spin.setRange(3, 100)
        self.min_vertices_spin.setValue(6)
        param_layout.addRow("最小顶点数:", self.min_vertices_spin)
        
        self.segments_spin = QtWidgets.QSpinBox()
        self.segments_spin.setRange(8, 128)
        self.segments_spin.setValue(32)
        param_layout.addRow("重建细分段数:", self.segments_spin)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 操作按钮
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.detect_btn = QtWidgets.QPushButton("检测孔洞")
        self.detect_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        btn_layout.addWidget(self.detect_btn)
        
        self.rebuild_btn = QtWidgets.QPushButton("重建选中孔洞")
        self.rebuild_btn.setEnabled(False)
        self.rebuild_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        btn_layout.addWidget(self.rebuild_btn)
        
        self.rebuild_all_btn = QtWidgets.QPushButton("重建所有圆弧孔洞")
        self.rebuild_all_btn.setEnabled(False)
        self.rebuild_all_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        btn_layout.addWidget(self.rebuild_all_btn)
        
        layout.addLayout(btn_layout)
        
        # 检测结果列表
        result_group = QtWidgets.QGroupBox("检测结果")
        result_layout = QtWidgets.QVBoxLayout()
        
        self.result_list = QtWidgets.QListWidget()
        self.result_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        result_layout.addWidget(self.result_list)
        
        # 统计信息
        self.stats_label = QtWidgets.QLabel("未检测")
        result_layout.addWidget(self.stats_label)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # 日志区域
        log_group = QtWidgets.QGroupBox("日志")
        log_layout = QtWidgets.QVBoxLayout()
        
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # 状态栏
        self.status_bar = QtWidgets.QStatusBar()
        layout.addWidget(self.status_bar)
    
    def _connect_signals(self):
        """连接信号和槽"""
        self.refresh_btn.clicked.connect(self._refresh_mesh_list)
        self.detect_btn.clicked.connect(self._detect_holes)
        self.rebuild_btn.clicked.connect(self._rebuild_selected)
        self.rebuild_all_btn.clicked.connect(self._rebuild_all)
        self.result_list.itemSelectionChanged.connect(self._on_selection_changed)
    
    def _refresh_mesh_list(self):
        """刷新网格列表"""
        self.mesh_combo.clear()
        for obj in FreeCAD.ActiveDocument.Objects:
            if hasattr(obj, 'Mesh'):
                self.mesh_combo.addItem(obj.Name)
        
        self._log("已刷新网格列表")
    
    def _detect_holes(self):
        """检测孔洞"""
        if self.mesh_combo.count() == 0:
            self._log("错误: 没有可用的网格对象")
            return
        
        mesh_name = self.mesh_combo.currentText()
        obj = FreeCAD.ActiveDocument.getObject(mesh_name)
        
        if obj is None or not hasattr(obj, 'Mesh'):
            self._log("错误: 无效的网格对象")
            return
        
        self.mesh_obj = obj
        mesh = obj.Mesh
        
        self._log(f"开始检测网格 '{mesh_name}' 的孔洞...")
        self._log(f"  顶点数: {len(mesh.Topology[0])}")
        self._log(f"  面数: {len(mesh.Topology[1])}")
        
        # 检测边界
        detector = BoundaryDetector(mesh)
        self.boundary_loops = detector.extract_boundary_loops()
        
        self._log(f"  找到 {len(self.boundary_loops)} 个边界环")
        
        # 检测圆弧
        tolerance = self.tolerance_spin.value()
        min_vertices = self.min_vertices_spin.value()
        
        arc_detector = ArcDetector(tolerance=tolerance)
        self.arc_results = []
        
        for i, loop in enumerate(self.boundary_loops):
            if len(loop.vertex_indices) < min_vertices:
                continue
            
            result = arc_detector.detect_circle(loop, mesh.Topology[0])
            if result is not None:
                self.arc_results.append(result)
        
        self._log(f"  检测到 {len(self.arc_results)} 个圆弧形孔洞")
        
        # 更新结果列表
        self._update_result_list()
        
        # 更新按钮状态
        self.rebuild_all_btn.setEnabled(len(self.arc_results) > 0)
        
        self._log("检测完成")
        self.status_bar.showMessage(f"检测到 {len(self.arc_results)} 个圆弧形孔洞")
    
    def _update_result_list(self):
        """更新结果列表"""
        self.result_list.clear()
        
        for i, result in enumerate(self.arc_results):
            item_text = f"孔洞 {i+1}: "
            item_text += f"中心=({result.circle_center.x:.2f}, {result.circle_center.y:.2f}, {result.circle_center.z:.2f}), "
            item_text += f"半径={result.circle_radius:.3f}, "
            item_text += f"偏差={result.deviation:.4f}, "
            item_text += f"角度={math.degrees(result.arc_angle):.1f}°"
            
            if result.is_full_circle:
                item_text += " [完整圆]"
            
            item = QtWidgets.QListWidgetItem(item_text)
            item.setData(QtCore.Qt.UserRole, i)
            self.result_list.addItem(item)
        
        # 更新统计信息
        total_loops = len(self.boundary_loops)
        arc_count = len(self.arc_results)
        self.stats_label.setText(f"总边界环: {total_loops}, 圆弧孔洞: {arc_count}")
    
    def _on_selection_changed(self):
        """选择改变时"""
        selected = self.result_list.selectedItems()
        self.rebuild_btn.setEnabled(len(selected) > 0)
    
    def _rebuild_selected(self):
        """重建选中的孔洞"""
        selected_items = self.result_list.selectedItems()
        if not selected_items:
            return
        
        indices = [item.data(QtCore.Qt.UserRole) for item in selected_items]
        self._rebuild_holes(indices)
    
    def _rebuild_all(self):
        """重建所有圆弧孔洞"""
        indices = list(range(len(self.arc_results)))
        self._rebuild_holes(indices)
    
    def _rebuild_holes(self, indices: List[int]):
        """重建指定的孔洞"""
        if not self.mesh_obj or not self.arc_results:
            return
        
        self._log(f"开始重建 {len(indices)} 个孔洞...")
        
        mesh = self.mesh_obj.Mesh
        rebuilder = HoleRebuilder(mesh)
        
        rebuilt_count = 0
        for idx in indices:
            if idx >= len(self.arc_results):
                continue
            
            result = self.arc_results[idx]
            self._log(f"  重建孔洞 {idx+1}: 半径={result.circle_radius:.3f}")
            
            # 根据类型选择重建方法
            if result.is_full_circle:
                new_mesh = rebuilder.rebuild_circle_hole(result, self.segments_spin.value())
            else:
                new_mesh = rebuilder.rebuild_arc_hole(result)
            
            rebuilt_count += 1
        
        # 更新网格
        if rebuilt_count > 0:
            # 创建新对象
            new_obj = FreeCAD.ActiveDocument.addObject("Mesh::Feature", f"{self.mesh_obj.Name}_repaired")
            new_obj.Mesh = rebuilder.rebuild_circle_hole(self.arc_results[indices[0]], self.segments_spin.value())
            
            FreeCAD.ActiveDocument.recompute()
            self._log(f"重建完成，创建了新对象 '{new_obj.Name}'")
            self.status_bar.showMessage(f"成功重建 {rebuilt_count} 个孔洞")
        else:
            self._log("没有孔洞被重建")
    
    def _log(self, message: str):
        """添加日志"""
        self.log_text.append(message)
        FreeCAD.Console.PrintMessage(f"[HoleRepair] {message}\n")


# ============================================================================
# FreeCAD 命令
# ============================================================================

class HoleRepairCommand:
    """FreeCAD 命令：孔洞修复"""
    
    def GetResources(self):
        return {
            'Pixmap': os.path.join(os.path.dirname(__file__), 'icons', 'hole_repair.svg'),
            'MenuText': 'Mesh 孔洞圆弧检测与重建',
            'ToolTip': '检测 Mesh 中的圆弧形孔洞并重建'
        }
    
    def IsActive(self):
        return FreeCAD.ActiveDocument is not None
    
    def Activated(self):
        dialog = HoleRepairDialog(FreeCADGui.getMainWindow())
        dialog.show()


# ============================================================================
# 注册命令
# ============================================================================

def register():
    """注册 FreeCAD 命令"""
    FreeCADGui.addCommand('HoleRepairCommand', HoleRepairCommand())


def unregister():
    """注销 FreeCAD 命令"""
    FreeCADGui.removeCommand('HoleRepairCommand')


# ============================================================================
# 主入口
# ============================================================================

if __name__ == '__main__':
    # 作为独立脚本运行
    register()
    
    # 创建对话框
    dialog = HoleRepairDialog(FreeCADGui.getMainWindow())
    dialog.show()
