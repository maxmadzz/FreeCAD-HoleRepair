#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeCAD 插件：B-Rep 多边形孔 → 圆孔重建

支持两种检测模式（参考 AnalysisSitus asiAlgo_RecognizeCanonical）：

模式 A — 内孔 Wire 检测（参数化 B-Rep）
  遍历每个 Face 的 inner Wire，Kasa 圆拟合。

模式 B — 面聚类检测（mesh→B-Rep 转换后的碎面）
  构建面邻接图 → 小平面聚类 → SVD 轴线检测 → 径向一致性检查。
  参考 AnalysisSitus CheckIsCylindrical：curvature analysis + SVD axis。

自动选择：有 inner wire → 模式 A；全是单 wire 小平面 → 模式 B。

依赖：FreeCAD 1.0+（内置 numpy）
"""

import os
import math
import numpy as np
from typing import List, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import FreeCAD
import FreeCADGui
import Part
from FreeCAD import Base
from PySide6 import QtWidgets, QtCore


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class HoleInfo:
    """孔洞信息（模式 A：inner wire 检测）"""
    face_index: int
    face: object = None
    wire_index: int = 0
    wire: object = None
    n_edges: int = 0
    points: list = field(default_factory=list)
    is_circular: bool = False
    fit_center: Optional[Base.Vector] = None
    fit_radius: float = 0.0
    fit_normal: Optional[Base.Vector] = None
    fit_max_dev: float = float('inf')
    fit_rel_dev: float = float('inf')
    detection_mode: str = "wire"


@dataclass
class ClusterInfo:
    """聚类信息（模式 B：面聚类检测）"""
    cluster_id: int
    face_indices: list = field(default_factory=list)
    face_areas: list = field(default_factory=list)
    is_cylindrical: bool = False
    cyl_axis_dir: Optional[np.ndarray] = None
    cyl_axis_point: Optional[np.ndarray] = None
    cyl_radius: float = 0.0
    cyl_center: Optional[Base.Vector] = None
    cyl_normal: Optional[Base.Vector] = None
    radius_std: float = float('inf')
    normal_alignment: float = 0.0
    detection_mode: str = "cluster"


# ============================================================================
# Kasa 最小二乘圆拟合
# ============================================================================

def fit_circle_kasa(points) -> Optional[Tuple]:
    """
    Kasa 最小二乘圆拟合。
    x² + y² + Dx + Ey + F = 0
    返回 (center_3d_Vector, radius, normal_Vector, max_deviation) 或 None。
    """
    n = len(points)
    if n < 6:
        return None

    coords_3d = np.array([(p.x, p.y, p.z) for p in points])
    centroid = coords_3d.mean(axis=0)
    centered = coords_3d - centroid

    # SVD 找拟合平面
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

    center_3d = centroid + u_vec * cx_2d + v_vec * cy_2d
    center = Base.Vector(center_3d[0], center_3d[1], center_3d[2])
    normal = Base.Vector(normal_vec[0], normal_vec[1], normal_vec[2])

    devs = [abs(p.distanceToPoint(center) - radius) for p in points]
    return center, radius, normal, max(devs)


# ============================================================================
# 模式 A：inner Wire 检测（参数化 B-Rep）
# ============================================================================

class HoleDetector:
    """
    检测 B-Rep Face 中的多边形孔洞（inner Wire）。
    适用于参数化 CAD 模型（面上有内孔 Wire）。
    """

    def __init__(self, rel_tolerance=30.0, min_edges=6, max_radius=20.0):
        self.rel_tolerance = rel_tolerance
        self.min_edges = min_edges
        self.max_radius = max_radius

    def detect(self, shape) -> List[HoleInfo]:
        results = []
        for fi, face in enumerate(shape.Faces):
            wires = face.Wires
            if len(wires) < 2:
                continue
            for wi in range(1, len(wires)):
                wire = wires[wi]
                edges = wire.Edges
                if len(edges) < self.min_edges:
                    continue
                pts = [v.Point for v in wire.Vertexes]
                if len(pts) < self.min_edges:
                    continue
                fit = fit_circle_kasa(pts)
                if fit is None:
                    continue
                center, radius, normal, max_dev = fit
                rel_dev = max_dev / radius * 100 if radius > 0 else float('inf')
                is_circ = rel_dev < self.rel_tolerance and radius <= self.max_radius
                results.append(HoleInfo(
                    face_index=fi, face=face, wire_index=wi, wire=wire,
                    n_edges=len(edges), points=pts, is_circular=is_circ,
                    fit_center=center, fit_radius=radius, fit_normal=normal,
                    fit_max_dev=max_dev, fit_rel_dev=rel_dev,
                    detection_mode="wire",
                ))
        return results


# ============================================================================
# 模式 B：面聚类检测（mesh→B-Rep 碎面）
# 参考 AnalysisSitus:
#   - CheckIsCylindrical: curvature analysis (k1=0, k2=1/R)
#   - AAG: face adjacency graph
#   - RecognizeDrillHoles: connected component matching
# ============================================================================

class FaceClusterDetector:
    """
    检测 mesh→B-Rep 转换后由大量小平面组成的圆柱孔。

    算法（参考 AnalysisSitus CheckIsCylindrical + AAG）：
    1. 构建面邻接图（共享边的面相连）
    2. BFS 聚类相连的小平面
    3. 对每个聚类做 SVD 轴线检测
    4. 检查径向一致性（std(r)/mean(r) < 阈值）
    5. 检查法线对齐（法线 ⊥ 轴线）
    """

    def __init__(self,
                 max_face_area=5.0,
                 radius_tolerance=0.3,
                 normal_tolerance=0.5,
                 min_cluster_size=8):
        self.max_face_area = max_face_area      # 小于此面积视为"小平面"
        self.radius_tolerance = radius_tolerance # std(r)/mean(r) < 此值
        self.normal_tolerance = normal_tolerance # cos(angle) < 此值表示法线 ⊥ 轴线
        self.min_cluster_size = min_cluster_size # 最少面数

    def _build_adjacency(self, shape):
        """构建面邻接图：共享边的面相连。"""
        # 边 → 面列表
        edge_to_faces = defaultdict(list)
        for fi, face in enumerate(shape.Faces):
            for edge in face.Edges:
                key = edge.hashCode()
                edge_to_faces[key].append(fi)

        # 面 → 邻接面集合
        adj = defaultdict(set)
        for edge_hash, faces in edge_to_faces.items():
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    adj[faces[i]].add(faces[j])
                    adj[faces[j]].add(faces[i])
        return adj

    def _find_clusters(self, shape, adj):
        """BFS 聚类小平面。"""
        visited = set()
        clusters = []

        # 筛选小平面
        small_faces = set()
        for fi, face in enumerate(shape.Faces):
            if face.Area < self.max_face_area:
                small_faces.add(fi)

        for fi in small_faces:
            if fi in visited:
                continue
            # BFS
            cluster = []
            queue = [fi]
            visited.add(fi)
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited and neighbor in small_faces:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        return clusters

    def _check_cylindrical(self, shape, face_indices):
        """
        检查一组面是否构成圆柱孔。
        参考 AnalysisSitus CheckIsCylindrical:
        - 圆柱面上 k1 ≈ 0, k2 = 1/R（一个方向曲率为0，另一个方向曲率为常数）
        - 对碎面：法线方向为径向，轴线方向为法线均值的 SVD 最小分量
        """
        # 收集面质心和法线
        centroids = []
        normals = []
        areas = []
        for fi in face_indices:
            face = shape.Faces[fi]
            cg = face.CenterOfGravity
            centroids.append([cg.x, cg.y, cg.z])
            # 面法线（取参数域中点）
            u_mid = (face.ParameterRange[0] + face.ParameterRange[1]) / 2
            v_mid = (face.ParameterRange[2] + face.ParameterRange[3]) / 2
            n = face.normalAt(u_mid, v_mid)
            normals.append([n.x, n.y, n.z])
            areas.append(face.Area)

        centroids = np.array(centroids)
        normals = np.array(normals)
        areas = np.array(areas)

        if len(centroids) < self.min_cluster_size:
            return None

        # === SVD 轴线检测 ===
        # 对质心做 SVD，最小奇异值对应的方向是轴线方向
        centroid_mean = centroids.mean(axis=0)
        centered = centroids - centroid_mean
        _, s, vh = np.linalg.svd(centered)

        # 最小奇异值对应的方向 = 圆柱轴线
        axis_dir = vh[2]

        # === 径向一致性检查 ===
        # 投影到 ⊥ 轴线的平面
        proj = centered - np.outer(centered @ axis_dir, axis_dir)
        radii = np.linalg.norm(proj, axis=1)

        r_mean = radii.mean()
        r_std = radii.std()
        radius_cv = r_std / r_mean if r_mean > 1e-6 else float('inf')

        if radius_cv > self.radius_tolerance:
            return None

        # === 法线对齐检查 ===
        # 圆柱面上的法线应该垂直于轴线
        # cos(angle) = |normal · axis|，应该 ≈ 0
        dot_products = np.abs(normals @ axis_dir)
        mean_dot = dot_products.mean()

        if mean_dot > self.normal_tolerance:
            return None

        # 计算圆心（投影到 ⊥ 轴线平面上的质心均值）
        proj_center = centroid_mean - np.dot(centroid_mean - centroids.mean(axis=0), axis_dir) * axis_dir
        # 更精确：用加权平均
        weights = areas / areas.sum()
        weighted_centroid = (centroids * weights[:, np.newaxis]).sum(axis=0)
        proj_weighted = weighted_centroid - np.dot(weighted_centroid, axis_dir) * axis_dir

        center = Base.Vector(proj_weighted[0], proj_weighted[1], proj_weighted[2])
        normal = Base.Vector(axis_dir[0], axis_dir[1], axis_dir[2])

        return {
            'axis_dir': axis_dir,
            'axis_point': centroid_mean,
            'radius': r_mean,
            'center': center,
            'normal': normal,
            'radius_cv': radius_cv,
            'normal_alignment': mean_dot,
        }

    def detect(self, shape) -> List[ClusterInfo]:
        """检测所有圆柱孔聚类。"""
        adj = self._build_adjacency(shape)
        clusters = self._find_clusters(shape, adj)

        results = []
        for ci, cluster in enumerate(clusters):
            result = self._check_cylindrical(shape, cluster)
            if result is None:
                continue
            results.append(ClusterInfo(
                cluster_id=ci,
                face_indices=cluster,
                face_areas=[shape.Faces[fi].Area for fi in cluster],
                is_cylindrical=True,
                cyl_axis_dir=result['axis_dir'],
                cyl_axis_point=result['axis_point'],
                cyl_radius=result['radius'],
                cyl_center=result['center'],
                cyl_normal=result['normal'],
                radius_std=result['radius_cv'],
                normal_alignment=result['normal_alignment'],
                detection_mode="cluster",
            ))

        return results


# ============================================================================
# 自动模式选择
# ============================================================================

def detect_holes(shape, rel_tolerance=30.0, min_edges=6,
                 max_radius=20.0, max_face_area=5.0, radius_tolerance=0.3):
    """
    自动选择检测模式：
    - 有 inner wire 的面 → 模式 A
    - 全是单 wire 小平面 → 模式 B
    两种模式都运行，合并结果。
    """
    results = []

    # 模式 A
    detector_a = HoleDetector(rel_tolerance=rel_tolerance, min_edges=min_edges, max_radius=max_radius)
    wire_results = detector_a.detect(shape)
    results.extend(wire_results)

    # 模式 B
    detector_b = FaceClusterDetector(
        max_face_area=max_face_area,
        radius_tolerance=radius_tolerance,
        min_cluster_size=min_edges,
    )
    cluster_results = detector_b.detect(shape)
    results.extend(cluster_results)

    return results


# ============================================================================
# GUI 对话框
# ============================================================================

class HoleRepairDialog(QtWidgets.QDialog):
    """B-Rep 多边形孔 → 圆孔重建对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("B-Rep 多边形孔 → 圆孔重建")
        self.setMinimumSize(700, 700)

        self.shape_obj = None
        self.results = []

        self._build_ui()
        self._connect()
        self._refresh_list()

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)

        # 对象选择
        grp_obj = QtWidgets.QGroupBox("对象选择")
        gl = QtWidgets.QHBoxLayout()
        self.combo = QtWidgets.QComboBox()
        gl.addWidget(QtWidgets.QLabel("B-Rep 对象:"))
        gl.addWidget(self.combo, 1)
        btn_refresh = QtWidgets.QPushButton("刷新")
        btn_refresh.clicked.connect(self._refresh_list)
        gl.addWidget(btn_refresh)
        grp_obj.setLayout(gl)
        lay.addWidget(grp_obj)

        # 参数
        grp_param = QtWidgets.QGroupBox("检测参数")
        fl = QtWidgets.QFormLayout()

        self.spin_tol = QtWidgets.QDoubleSpinBox()
        self.spin_tol.setRange(1.0, 100.0)
        self.spin_tol.setValue(30.0)
        self.spin_tol.setSingleStep(5.0)
        self.spin_tol.setSuffix("%")
        fl.addRow("圆拟合相对偏差容差:", self.spin_tol)

        self.spin_min = QtWidgets.QSpinBox()
        self.spin_min.setRange(3, 200)
        self.spin_min.setValue(6)
        fl.addRow("最小边数/聚类面数:", self.spin_min)

        self.spin_max_r = QtWidgets.QDoubleSpinBox()
        self.spin_max_r.setRange(0.1, 10000.0)
        self.spin_max_r.setValue(20.0)
        self.spin_max_r.setSuffix(" mm")
        fl.addRow("最大圆弧半径:", self.spin_max_r)

        self.spin_area = QtWidgets.QDoubleSpinBox()
        self.spin_area.setRange(0.1, 1000.0)
        self.spin_area.setValue(5.0)
        self.spin_area.setSuffix(" mm²")
        fl.addRow("小平面面积阈值 (模式B):", self.spin_area)

        self.spin_rtol = QtWidgets.QDoubleSpinBox()
        self.spin_rtol.setRange(0.01, 1.0)
        self.spin_rtol.setValue(0.3)
        self.spin_rtol.setSingleStep(0.05)
        fl.addRow("半径变异系数阈值 (模式B):", self.spin_rtol)

        grp_param.setLayout(fl)
        lay.addWidget(grp_param)

        # 按钮
        bl = QtWidgets.QHBoxLayout()
        self.btn_detect = QtWidgets.QPushButton("检测孔洞")
        self.btn_detect.setStyleSheet("QPushButton{background:#4CAF50;color:white;padding:6px}")
        bl.addWidget(self.btn_detect)

        self.btn_rebuild = QtWidgets.QPushButton("重建选中")
        self.btn_rebuild.setEnabled(False)
        self.btn_rebuild.setStyleSheet("QPushButton{background:#2196F3;color:white;padding:6px}")
        bl.addWidget(self.btn_rebuild)

        self.btn_rebuild_all = QtWidgets.QPushButton("重建所有")
        self.btn_rebuild_all.setEnabled(False)
        self.btn_rebuild_all.setStyleSheet("QPushButton{background:#FF9800;color:white;padding:6px}")
        bl.addWidget(self.btn_rebuild_all)
        lay.addLayout(bl)

        # 结果
        grp_res = QtWidgets.QGroupBox("检测结果")
        rl = QtWidgets.QVBoxLayout()
        self.result_list = QtWidgets.QListWidget()
        self.result_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        rl.addWidget(self.result_list)
        self.lbl_stats = QtWidgets.QLabel("未检测")
        rl.addWidget(self.lbl_stats)
        grp_res.setLayout(rl)
        lay.addWidget(grp_res)

        # 日志
        grp_log = QtWidgets.QGroupBox("日志")
        ll = QtWidgets.QVBoxLayout()
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        ll.addWidget(self.log_text)
        grp_log.setLayout(ll)
        lay.addWidget(grp_log)

    def _connect(self):
        self.combo.currentIndexChanged.connect(self._on_select)
        self.btn_detect.clicked.connect(self._detect)
        self.btn_rebuild.clicked.connect(self._rebuild_sel)
        self.btn_rebuild_all.clicked.connect(self._rebuild_all)
        self.result_list.itemSelectionChanged.connect(self._on_sel)

    def _refresh_list(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        if FreeCAD.ActiveDocument:
            for obj in FreeCAD.ActiveDocument.Objects:
                if hasattr(obj, 'Shape'):
                    self.combo.addItem(obj.Name)
        self.combo.blockSignals(False)
        self._on_select()

    def _on_select(self):
        name = self.combo.currentText()
        self.shape_obj = None
        if FreeCAD.ActiveDocument and name:
            obj = FreeCAD.ActiveDocument.getObject(name)
            if obj and hasattr(obj, 'Shape'):
                self.shape_obj = obj
        self._log("选中: %s" % (name or "(无)"))

    def _detect(self):
        if not self.shape_obj:
            self._log("错误: 请选择 B-Rep 对象")
            return

        shape = self.shape_obj.Shape
        n_faces = len(shape.Faces)
        n_edges = len(shape.Edges)
        n_solids = len(shape.Solids)
        self._log("Shape: %d Faces, %d Edges, %d Solids" % (n_faces, n_edges, n_solids))

        self.results = detect_holes(
            shape,
            rel_tolerance=self.spin_tol.value(),
            min_edges=self.spin_min.value(),
            max_radius=self.spin_max_r.value(),
            max_face_area=self.spin_area.value(),
            radius_tolerance=self.spin_rtol.value(),
        )

        # 更新 UI
        self.result_list.clear()
        circ_count = 0
        for i, info in enumerate(self.results):
            if info.detection_mode == "wire":
                if info.is_circular:
                    circ_count += 1
                    c = info.fit_center
                    text = "[圆弧孔] Face%d/Wire%d  R=%.3f  中心=(%.1f,%.1f,%.1f)  偏差=%.1f%%  %d边" % (
                        info.face_index, info.wire_index,
                        info.fit_radius, c.x, c.y, c.z,
                        info.fit_rel_dev, info.n_edges)
                else:
                    text = "[非圆弧] Face%d/Wire%d  %d边  偏差=%.1f%%" % (
                        info.face_index, info.wire_index,
                        info.n_edges, info.fit_rel_dev)
            elif info.detection_mode == "cluster":
                if info.is_cylindrical:
                    circ_count += 1
                    c = info.cyl_center
                    text = "[圆柱孔] 聚类%d  %d面  R=%.3f  中心=(%.1f,%.1f,%.1f)  CV=%.3f" % (
                        info.cluster_id, len(info.face_indices),
                        info.cyl_radius, c.x, c.y, c.z,
                        info.radius_std)
                else:
                    text = "[非圆柱] 聚类%d  %d面" % (
                        info.cluster_id, len(info.face_indices))
            else:
                text = "[未知] %s" % info.detection_mode

            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, i)
            if getattr(info, 'is_circular', False) or getattr(info, 'is_cylindrical', False):
                item.setForeground(QtCore.Qt.darkGreen)
            self.result_list.addItem(item)

        self.lbl_stats.setText("检测到 %d 个结果, %d 个匹配" % (len(self.results), circ_count))
        self.btn_rebuild_all.setEnabled(circ_count > 0)
        self._log("检测完成: %d 个结果, %d 个匹配" % (len(self.results), circ_count))

    def _on_sel(self):
        sel = self.result_list.selectedItems()
        has_circular = False
        for item in sel:
            idx = item.data(QtCore.Qt.UserRole)
            info = self.results[idx]
            if getattr(info, 'is_circular', False) or getattr(info, 'is_cylindrical', False):
                has_circular = True
                break
        self.btn_rebuild.setEnabled(has_circular)

    def _rebuild_sel(self):
        indices = [item.data(QtCore.Qt.UserRole) for item in self.result_list.selectedItems()]
        self._do_rebuild(indices)

    def _rebuild_all(self):
        indices = []
        for i, info in enumerate(self.results):
            if getattr(info, 'is_circular', False) or getattr(info, 'is_cylindrical', False):
                indices.append(i)
        self._do_rebuild(indices)

    def _do_rebuild(self, indices):
        """Wire 替换法重建圆孔：直接将多边形 inner wire 替换为圆 wire。

        与 Cover+Pocket 的区别：
        - 不创建贯穿模型的圆柱
        - 不影响周围几何（槽、凸台等）
        - BB 完全不变，面数不变
        """
        if not self.shape_obj or not indices:
            return

        doc = FreeCAD.ActiveDocument
        all_faces = list(self.shape_obj.Shape.Faces)
        count = 0

        for idx in indices:
            info = self.results[idx]

            if info.detection_mode == "wire" and info.is_circular:
                fi = info.face_index
                face = all_faces[fi]

                # 拟合圆（Kasa 最小二乘，只用顶点）
                pts = [v.Point for v in info.wire.Vertexes]
                coords = np.array([(p.x, p.y, p.z) for p in pts])
                centroid = coords.mean(axis=0)
                centered = coords - centroid
                _, _, vh = np.linalg.svd(centered)
                normal_vec = vh[2]
                u_vec, v_vec = vh[0], vh[1]
                coords_2d = centered @ np.column_stack([u_vec, v_vec])
                x, y = coords_2d[:, 0], coords_2d[:, 1]
                A = np.column_stack([x, y, np.ones(len(x))])
                b_arr = -(x**2 + y**2)
                D, E, F = np.linalg.lstsq(A, b_arr, rcond=None)[0]
                cx_2d, cy_2d = -D / 2, -E / 2
                radius = math.sqrt(cx_2d**2 + cy_2d**2 - F)
                center_3d = centroid + u_vec * cx_2d + v_vec * cy_2d

                # 创建圆形 inner wire
                circ = Part.Circle()
                circ.Center = Base.Vector(center_3d[0], center_3d[1], center_3d[2])
                circ.Axis = Base.Vector(normal_vec[0], normal_vec[1], normal_vec[2])
                circ.Radius = radius
                circle_wire = Part.Wire(circ.toShape())

                # 替换：保留 outer wire + 其他 inner wire，只替换当前孔
                outer_wire = face.Wires[0]
                other_inner = [face.Wires[i] for i in range(2, len(face.Wires))]
                try:
                    new_face = Part.Face([outer_wire, circle_wire] + other_inner)
                    all_faces[fi] = new_face
                    count += 1
                    self._log("  -> Wire 替换完成 R=%.6f" % radius)
                except Exception as ex:
                    self._log("  -> Wire 替换失败: %s" % ex)

            elif info.detection_mode == "cluster" and info.is_cylindrical:
                # 面聚类检测结果 — 暂不支持 wire 替换
                self._log("  -> 聚类模式暂不支持 wire 替换，跳过")

        if count == 0:
            self._log("没有可重建的孔洞")
            return

        # 用新面重建 Solid
        try:
            result = Part.Solid(Part.Shell(all_faces))
            new_name = self.shape_obj.Name + "_repaired"
            new_obj = doc.addObject("Part::Feature", new_name)
            new_obj.Shape = result
            doc.recompute()
            self._log("完成: %s (%d 个孔洞已重建)" % (new_name, count))
        except Exception as ex:
            self._log("重建 Solid 失败: %s" % ex)

    def _log(self, msg):
        self.log_text.append(msg)
        FreeCAD.Console.PrintMessage("[HoleRepair] %s\n" % msg)


# ============================================================================
# FreeCAD 命令
# ============================================================================

class HoleRepairCommand:
    def GetResources(self):
        return {
            'MenuText': 'B-Rep 多边形孔 → 圆孔重建',
            'ToolTip': '检测 B-Rep 中的多边形孔洞（支持 mesh 转换的碎面），重建为真正的圆弧几何',
        }

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None

    def Activated(self):
        self.dialog = HoleRepairDialog(FreeCADGui.getMainWindow())
        self.dialog.show()


def register():
    FreeCADGui.addCommand('HoleRepairCommand', HoleRepairCommand())

def unregister():
    FreeCADGui.removeCommand('HoleRepairCommand')

if __name__ == '__main__':
    register()
    dialog = HoleRepairDialog(FreeCADGui.getMainWindow())
    dialog.show()
