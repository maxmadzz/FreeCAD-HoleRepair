#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeCAD 插件：Mesh 孔洞圆弧检测与重建

功能：
1. 检测 Mesh 中的孔洞边界
2. 识别圆弧形边界（最小二乘法拟合圆）
3. 重建圆弧形孔洞（扇形三角化或调用 FreeCAD 内置 fillupHoles）

依赖：
- FreeCAD 0.20+（内置 numpy/scipy）
"""

import os
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum

import FreeCAD
import FreeCADGui
import Mesh
from FreeCAD import Base
from PySide6 import QtWidgets, QtCore


# ============================================================================
# 数据结构
# ============================================================================

class HoleType(Enum):
    CIRCLE = "circle"
    ARC = "arc"
    POLYGON = "polygon"
    UNKNOWN = "unknown"


@dataclass
class BoundaryLoop:
    """边界环"""
    vertex_indices: List[int]                       # 顶点在 mesh.Topology[0] 中的索引
    points: List[Base.Vector] = field(default_factory=list)  # 3D 坐标
    perimeter: float = 0.0
    area: float = 0.0
    centroid: Base.Vector = field(default_factory=lambda: Base.Vector(0, 0, 0))
    normal: Base.Vector = field(default_factory=lambda: Base.Vector(0, 0, 1))
    hole_type: HoleType = HoleType.UNKNOWN
    fit_center: Optional[Base.Vector] = None
    fit_radius: float = 0.0
    fit_error: float = float('inf')


@dataclass
class ArcDetectionResult:
    """圆弧检测结果"""
    loop: BoundaryLoop
    center: Base.Vector
    radius: float
    normal: Base.Vector
    deviation: float
    is_full_circle: bool
    arc_angle: float


# ============================================================================
# 边界检测
# ============================================================================

class BoundaryDetector:
    """从 Mesh 中提取边界环"""

    def __init__(self, mesh_obj):
        """
        Args:
            mesh_obj: FreeCAD Mesh::Feature 对象（不是 Mesh.Mesh）
        """
        self.mesh_obj = mesh_obj
        self.mesh = mesh_obj.Mesh
        # Topology[0] 是 Base.Vector 列表，Topology[1] 是 (int,int,int) tuple 列表
        self.vertices: List[Base.Vector] = list(self.mesh.Topology[0])
        self.faces: List[Tuple[int, int, int]] = list(self.mesh.Topology[1])

    def find_boundary_loops(self) -> List[BoundaryLoop]:
        """提取所有边界环"""
        edge_face_map: Dict[Tuple[int, int], int] = {}

        # 统计每条边被多少个面共享
        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = (min(v1, v2), max(v1, v2))
                edge_face_map[edge] = edge_face_map.get(edge, 0) + 1

        # 边界边 = 只属于 1 个面
        boundary_edges = {e for e, cnt in edge_face_map.items() if cnt == 1}

        if not boundary_edges:
            return []

        # 构建邻接表（仅边界边）
        adjacency: Dict[int, List[int]] = {}
        for v1, v2 in boundary_edges:
            adjacency.setdefault(v1, []).append(v2)
            adjacency.setdefault(v2, []).append(v1)

        # 遍历所有边界环
        visited_edges = set()
        loops: List[BoundaryLoop] = []

        for start_edge in boundary_edges:
            if start_edge in visited_edges:
                continue

            loop_verts = []
            current, target = start_edge

            while True:
                edge_key = (min(current, target), max(current, target))
                if edge_key in visited_edges:
                    break
                visited_edges.add(edge_key)
                loop_verts.append(current)
                current = target

                # 找下一个未访问邻接点
                nxt = None
                for neighbor in adjacency.get(current, []):
                    ek = (min(current, neighbor), max(current, neighbor))
                    if ek not in visited_edges:
                        nxt = neighbor
                        break
                if nxt is None:
                    break
                target = nxt

            if len(loop_verts) >= 3:
                loops.append(self._build_loop(loop_verts))

        return loops

    def _build_loop(self, vert_indices: List[int]) -> BoundaryLoop:
        pts = [self.vertices[v] for v in vert_indices]

        # 周长
        perim = sum(pts[i].distanceToPoint(pts[(i + 1) % len(pts)])
                    for i in range(len(pts)))

        # 质心
        centroid = Base.Vector(0, 0, 0)
        for p in pts:
            centroid += p
        centroid /= len(pts)

        # 法向量 (Newell method)
        normal = Base.Vector(0, 0, 0)
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            normal.x += (p1.y - p2.y) * (p1.z + p2.z)
            normal.y += (p1.z - p2.z) * (p1.x + p2.x)
            normal.z += (p1.x - p2.x) * (p1.y + p2.y)
        ln = normal.Length
        if ln > 1e-12:
            normal /= ln
        else:
            normal = Base.Vector(0, 0, 1)

        # 面积 (投影到最佳拟合平面后用 Shoelace)
        area = self._projected_area(pts, normal)

        return BoundaryLoop(
            vertex_indices=vert_indices,
            points=pts,
            perimeter=perim,
            area=area,
            centroid=centroid,
            normal=normal,
        )

    @staticmethod
    def _projected_area(pts: List[Base.Vector], normal: Base.Vector) -> float:
        """投影面积（Shoelace 公式）"""
        # 选投影平面
        ax = abs(normal.x)
        ay = abs(normal.y)
        az = abs(normal.z)
        if az >= ax and az >= ay:
            coords = [(p.x, p.y) for p in pts]
        elif ax >= ay:
            coords = [(p.y, p.z) for p in pts]
        else:
            coords = [(p.x, p.z) for p in pts]

        n = len(coords)
        a = 0.0
        for i in range(n):
            j = (i + 1) % n
            a += coords[i][0] * coords[j][1] - coords[j][0] * coords[i][1]
        return abs(a) / 2.0


# ============================================================================
# 圆弧检测（纯 Python + numpy，仿 AnalysisSitus FitCircle 思路）
# ============================================================================

class ArcDetector:
    """
    检测边界环是否为圆弧。
    算法：取 3 点构造圆，再用 20+ 采样点验证偏差。
    与 AnalysisSitus asiAlgo_RecognizeCanonical::FitCircle 同源思路。
    """

    def __init__(self, tolerance: float = 0.1, min_vertices: int = 6):
        self.tolerance = tolerance
        self.min_vertices = min_vertices

    def detect(self, loop: BoundaryLoop) -> Optional[ArcDetectionResult]:
        if len(loop.vertex_indices) < self.min_vertices:
            return None

        pts = loop.points
        n = len(pts)

        # Kasa 全点最小二乘拟合圆（与 AnalysisSitus FitCircle 同源思路）
        fit_result = self._fit_circle_kasa(pts)
        if fit_result is None:
            return None

        center, radius, normal, max_dev = fit_result

        # 验证偏差
        if max_dev > self.tolerance:
            return None

        # 计算圆弧角度
        arc_angle = self._arc_angle(pts, center, normal)
        is_full = abs(arc_angle - 2 * math.pi) < 0.15

        # 更新 loop 信息
        loop.hole_type = HoleType.CIRCLE if is_full else HoleType.ARC
        loop.fit_center = center
        loop.fit_radius = radius
        loop.fit_error = max_dev

        return ArcDetectionResult(
            loop=loop,
            center=center,
            radius=radius,
            normal=normal,
            deviation=max_dev,
            is_full_circle=is_full,
            arc_angle=arc_angle,
        )

    @staticmethod
    def _circle_from_3points(p0: Base.Vector, p1: Base.Vector, p2: Base.Vector):
        """三点定圆，返回 (center, radius, normal) 或 None"""
        eps = 1e-9
        d01 = p0.distanceToPoint(p1)
        d02 = p0.distanceToPoint(p2)
        if d01 < eps or d02 < eps:
            return None

        v01 = p1 - p0
        v02 = p2 - p0
        cross = v01.cross(v02)
        cross_sq = cross.x ** 2 + cross.y ** 2 + cross.z ** 2
        if cross_sq < d01 * d02 * eps:
            return None  # 三点共线

        # 使用 OCCT 的 gce_MakeCirc 等价算法
        # 中垂线法
        m1 = (p0 + p1) / 2.0
        m2 = (p0 + p2) / 2.0
        n = cross.normalize()

        # 两条中垂线方向
        d1 = v01.cross(n)
        d2 = v02.cross(n)

        # 解交点 (最小二乘)
        A = np.array([[d1.x, -d2.x],
                       [d1.y, -d2.y],
                       [d1.z, -d2.z]])
        b = np.array([m2.x - m1.x, m2.y - m1.y, m2.z - m1.z])
        result = np.linalg.lstsq(A, b, rcond=None)
        t = result[0][0]
        center = m1 + d1 * t
        radius = p0.distanceToPoint(center)

        if radius < eps or radius > 1e6:
            return None

        return center, radius, n

    @staticmethod
    def _fit_circle_kasa(points: List[Base.Vector]):
        """
        Kasa 最小二乘圆拟合。
        x^2 + y^2 + Dx + Ey + F = 0
        返回 (center, radius, normal, max_deviation) 或 None。
        """
        n = len(points)
        if n < 6:
            return None

        coords_3d = np.array([(p.x, p.y, p.z) for p in points])
        centroid = coords_3d.mean(axis=0)
        centered = coords_3d - centroid

        # SVD 找拟合平面法向量
        _, _, vh = np.linalg.svd(centered)
        normal_vec = vh[2]
        u_vec, v_vec = vh[0], vh[1]

        # 投影到 2D
        coords_2d = centered @ np.column_stack([u_vec, v_vec])
        x, y = coords_2d[:, 0], coords_2d[:, 1]

        # Kasa 拟合
        A = np.column_stack([x, y, np.ones(n)])
        b_arr = -(x**2 + y**2)
        try:
            D, E, F = np.linalg.lstsq(A, b_arr, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        cx_2d, cy_2d = -D / 2, -E / 2
        rad_sq = cx_2d**2 + cy_2d**2 - F
        if rad_sq < 1e-12:
            return None
        radius = math.sqrt(rad_sq)
        if radius < 1e-6 or radius > 1e6:
            return None

        # 转回 3D
        center_3d = centroid + u_vec * cx_2d + v_vec * cy_2d
        center = Base.Vector(center_3d[0], center_3d[1], center_3d[2])
        normal = Base.Vector(normal_vec[0], normal_vec[1], normal_vec[2])

        devs = [abs(p.distanceToPoint(center) - radius) for p in points]
        return center, radius, normal, max(devs)

    @staticmethod
    def _arc_angle(pts: List[Base.Vector], center: Base.Vector, normal: Base.Vector) -> float:
        """计算点序列在圆上的总弧度"""
        n = len(pts)
        if n < 2:
            return 0.0

        vectors = [(p - center).normalize() for p in pts]
        total = 0.0
        for i in range(n):
            v1 = vectors[i]
            v2 = vectors[(i + 1) % n]
            dot = max(-1.0, min(1.0, v1.dot(v2)))
            angle = math.acos(dot)
            cross = v1.cross(v2)
            if cross.dot(normal) < 0:
                angle = -angle
            total += angle
        return abs(total)


# ============================================================================
# 孔洞重建
# ============================================================================

class HoleRebuilder:
    """孔洞重建器"""

    @staticmethod
    def rebuild_circle(mesh_obj, result: ArcDetectionResult, segments: int = 32) -> Mesh.Mesh:
        """
        重建圆形孔洞：用圆心 + 边界点做扇形三角化。
        返回一个新的 Mesh 对象（不修改原 mesh）。
        """
        loop = result.loop
        pts = loop.points
        center = result.center

        new_mesh = Mesh.Mesh()

        # 扇形三角化：center -> pt[i] -> pt[i+1]
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            new_mesh.addFacet(center, pts[i], pts[j])

        return new_mesh

    @staticmethod
    def rebuild_generic(mesh_obj, loop: BoundaryLoop) -> Mesh.Mesh:
        """
        通用孔洞重建：用质心做扇形三角化。
        """
        pts = loop.points
        centroid = loop.centroid

        new_mesh = Mesh.Mesh()
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            new_mesh.addFacet(centroid, pts[i], pts[j])

        return new_mesh

    @staticmethod
    def rebuild_builtin(mesh_obj, max_radius: int = 100, tolerance: float = 0.1):
        """
        调用 FreeCAD 内置 fillupHoles 方法。
        注意：maxRadius 必须是 int，tolerance 是 float。
        """
        mesh_obj.Mesh.fillupHoles(max_radius, int(tolerance * 1000))


# ============================================================================
# GUI
# ============================================================================

class HoleRepairDialog(QtWidgets.QDialog):
    """孔洞修复对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mesh 孔洞圆弧检测与重建")
        self.setMinimumSize(560, 620)

        self.mesh_obj = None
        self.loops: List[BoundaryLoop] = []
        self.arc_results: List[ArcDetectionResult] = []

        self._build_ui()
        self._connect()
        self._refresh_mesh_list()

    # ---------- UI ----------

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)

        # --- 网格选择 ---
        grp_mesh = QtWidgets.QGroupBox("网格选择")
        gl = QtWidgets.QHBoxLayout()
        self.mesh_combo = QtWidgets.QComboBox()
        gl.addWidget(QtWidgets.QLabel("对象:"))
        gl.addWidget(self.mesh_combo, 1)
        btn_refresh = QtWidgets.QPushButton("刷新")
        gl.addWidget(btn_refresh)
        grp_mesh.setLayout(gl)
        lay.addWidget(grp_mesh)

        # --- 参数 ---
        grp_param = QtWidgets.QGroupBox("检测参数")
        fl = QtWidgets.QFormLayout()

        self.spin_tol = QtWidgets.QDoubleSpinBox()
        self.spin_tol.setRange(0.001, 10.0)
        self.spin_tol.setValue(0.1)
        self.spin_tol.setSingleStep(0.01)
        fl.addRow("圆弧容差 (mm):", self.spin_tol)

        self.spin_min = QtWidgets.QSpinBox()
        self.spin_min.setRange(3, 200)
        self.spin_min.setValue(6)
        fl.addRow("最小顶点数:", self.spin_min)

        self.spin_seg = QtWidgets.QSpinBox()
        self.spin_seg.setRange(8, 128)
        self.spin_seg.setValue(32)
        fl.addRow("重建细分段:", self.spin_seg)

        grp_param.setLayout(fl)
        lay.addWidget(grp_param)

        # --- 按钮 ---
        bl = QtWidgets.QHBoxLayout()
        self.btn_detect = QtWidgets.QPushButton("检测孔洞")
        self.btn_detect.setStyleSheet("QPushButton{background:#4CAF50;color:white;padding:6px}")
        bl.addWidget(self.btn_detect)

        self.btn_rebuild_sel = QtWidgets.QPushButton("重建选中")
        self.btn_rebuild_sel.setEnabled(False)
        self.btn_rebuild_sel.setStyleSheet("QPushButton{background:#2196F3;color:white;padding:6px}")
        bl.addWidget(self.btn_rebuild_sel)

        self.btn_rebuild_all = QtWidgets.QPushButton("重建所有圆弧")
        self.btn_rebuild_all.setEnabled(False)
        self.btn_rebuild_all.setStyleSheet("QPushButton{background:#FF9800;color:white;padding:6px}")
        bl.addWidget(self.btn_rebuild_all)

        self.btn_fill_builtin = QtWidgets.QPushButton("FreeCAD 填充")
        self.btn_fill_builtin.setStyleSheet("QPushButton{background:#9C27B0;color:white;padding:6px}")
        bl.addWidget(self.btn_fill_builtin)
        lay.addLayout(bl)

        # --- 结果列表 ---
        grp_res = QtWidgets.QGroupBox("检测结果")
        rl = QtWidgets.QVBoxLayout()
        self.result_list = QtWidgets.QListWidget()
        self.result_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        rl.addWidget(self.result_list)
        self.lbl_stats = QtWidgets.QLabel("未检测")
        rl.addWidget(self.lbl_stats)
        grp_res.setLayout(rl)
        lay.addWidget(grp_res)

        # --- 日志 ---
        grp_log = QtWidgets.QGroupBox("日志")
        ll = QtWidgets.QVBoxLayout()
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        ll.addWidget(self.log_text)
        grp_log.setLayout(ll)
        lay.addWidget(grp_log)

    def _connect(self):
        self.mesh_combo.currentIndexChanged.connect(self._on_mesh_changed)
        self.btn_detect.clicked.connect(self._detect)
        self.btn_rebuild_sel.clicked.connect(self._rebuild_selected)
        self.btn_rebuild_all.clicked.connect(self._rebuild_all)
        self.btn_fill_builtin.clicked.connect(self._fill_builtin)
        self.result_list.itemSelectionChanged.connect(self._on_sel_changed)

    # ---------- 逻辑 ----------

    def _refresh_mesh_list(self):
        self.mesh_combo.blockSignals(True)
        self.mesh_combo.clear()
        if FreeCAD.ActiveDocument:
            for obj in FreeCAD.ActiveDocument.Objects:
                if obj.TypeId == "Mesh::Feature":
                    self.mesh_combo.addItem(obj.Name)
        self.mesh_combo.blockSignals(False)
        self._on_mesh_changed()

    def _on_mesh_changed(self):
        name = self.mesh_combo.currentText()
        self.mesh_obj = None
        if FreeCAD.ActiveDocument and name:
            obj = FreeCAD.ActiveDocument.getObject(name)
            if obj and obj.TypeId == "Mesh::Feature":
                self.mesh_obj = obj
        self._log(f"选中网格: {name or '(无)'}")

    def _detect(self):
        if not self.mesh_obj:
            self._log("错误: 请先选择网格对象")
            return

        mesh = self.mesh_obj.Mesh
        self._log(f"检测 '{self.mesh_obj.Name}': {mesh.CountPoints} 顶点, {mesh.CountFacets} 面")

        # 边界检测
        detector = BoundaryDetector(self.mesh_obj)
        self.loops = detector.find_boundary_loops()
        self._log(f"找到 {len(self.loops)} 个边界环")

        # 圆弧检测
        arc_det = ArcDetector(
            tolerance=self.spin_tol.value(),
            min_vertices=self.spin_min.value(),
        )
        self.arc_results = []
        for loop in self.loops:
            result = arc_det.detect(loop)
            if result is not None:
                self.arc_results.append(result)

        self._log(f"其中 {len(self.arc_results)} 个是圆弧形")

        self._update_list()
        self.btn_rebuild_all.setEnabled(len(self.arc_results) > 0)
        self.lbl_stats.setText(f"边界环: {len(self.loops)}, 圆弧孔洞: {len(self.arc_results)}")

    def _update_list(self):
        self.result_list.clear()
        for i, r in enumerate(self.arc_results):
            c = r.center
            tag = "[整圆]" if r.is_full_circle else f"[弧 {math.degrees(r.arc_angle):.0f}°]"
            text = (f"#{i+1}  中心=({c.x:.2f},{c.y:.2f},{c.z:.2f})  "
                    f"R={r.radius:.3f}  偏差={r.deviation:.4f}  {tag}")
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, i)
            self.result_list.addItem(item)

    def _on_sel_changed(self):
        self.btn_rebuild_sel.setEnabled(len(self.result_list.selectedItems()) > 0)

    def _rebuild_selected(self):
        indices = [item.data(QtCore.Qt.UserRole) for item in self.result_list.selectedItems()]
        self._do_rebuild(indices)

    def _rebuild_all(self):
        self._do_rebuild(list(range(len(self.arc_results))))

    def _do_rebuild(self, indices: List[int]):
        if not self.mesh_obj or not self.arc_results:
            return

        self._log(f"重建 {len(indices)} 个孔洞...")

        combined = Mesh.Mesh()
        for idx in indices:
            r = self.arc_results[idx]
            patch = HoleRebuilder.rebuild_circle(self.mesh_obj, r, self.spin_seg.value())
            combined.addMesh(patch)
            self._log(f"  #{idx+1}: R={r.radius:.3f}, {patch.CountFacets} 面片")

        # 创建新对象
        doc = FreeCAD.ActiveDocument
        new_name = f"{self.mesh_obj.Name}_repaired"
        new_obj = doc.addObject("Mesh::Feature", new_name)
        new_obj.Mesh = combined
        doc.recompute()

        # 将补丁合并到原 mesh
        self.mesh_obj.Mesh.addMesh(combined)
        doc.recompute()

        self._log(f"完成！补丁对象: {new_name}（已合并到原网格）")

    def _fill_builtin(self):
        """调用 FreeCAD 内置 fillupHoles"""
        if not self.mesh_obj:
            self._log("错误: 请先选择网格对象")
            return
        before = self.mesh_obj.Mesh.CountFacets
        # fillupHoles(maxRadius: int, tolerance: int)
        self.mesh_obj.Mesh.fillupHoles(100, 100)
        FreeCAD.ActiveDocument.recompute()
        after = self.mesh_obj.Mesh.CountFacets
        self._log(f"FreeCAD 内置填充: {before} → {after} 面片 (增加 {after - before})")

    def _log(self, msg: str):
        self.log_text.append(msg)
        FreeCAD.Console.PrintMessage(f"[HoleRepair] {msg}\n")


# ============================================================================
# FreeCAD 命令
# ============================================================================

class HoleRepairCommand:
    def GetResources(self):
        return {
            'MenuText': 'Mesh 孔洞圆弧检测与重建',
            'ToolTip': '检测 Mesh 中的圆弧形孔洞并重建',
        }

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None

    def Activated(self):
        self.dialog = HoleRepairDialog(FreeCADGui.getMainWindow())
        self.dialog.show()


# ============================================================================
# 注册 / 入口
# ============================================================================

def register():
    FreeCADGui.addCommand('HoleRepairCommand', HoleRepairCommand())

def unregister():
    FreeCADGui.removeCommand('HoleRepairCommand')


if __name__ == '__main__':
    register()
    dialog = HoleRepairDialog(FreeCADGui.getMainWindow())
    dialog.show()
