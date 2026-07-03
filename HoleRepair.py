#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeCAD 插件：B-Rep 几何修复（圆弧/圆柱检测与重建）

功能：
1. 边级别：检测碎边直线 → 重建为 Geom_Circle 圆弧
2. 面级别：检测碎面平面 → 重建为 Geom_CylindricalSurface 圆柱面
3. Wire 级别：检测开放边界圆弧孔洞 → 重建为闭合圆

算法参考 AnalysisSitus:
- asiAlgo_RecognizeCanonical::FitCircle → Kasa 最小二乘圆拟合
- asiAlgo_RecognizeCanonical::CheckIsCylindrical → 圆柱面检测
- asiAlgo_RecognizeCanonical::CheckType → 曲面类型识别

依赖：FreeCAD 1.0+（内置 OCCT 7.8、numpy）
"""

import os
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
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
class WireInfo:
    """边界 wire 信息"""
    edges: list
    points: list = field(default_factory=list)
    perimeter: float = 0.0
    centroid: Base.Vector = field(default_factory=lambda: Base.Vector(0, 0, 0))
    is_circular: bool = False
    fit_center: Optional[Base.Vector] = None
    fit_radius: float = 0.0
    fit_normal: Optional[Base.Vector] = None
    fit_deviation: float = float('inf')
    replacement_edge: object = None


@dataclass
class FaceInfo:
    """面级别分析结果"""
    face_index: int
    face: object = None
    original_type: str = ""         # Plane, BSpline, Cylinder 等
    is_cylindrical: bool = False
    fit_axis_point: Optional[Base.Vector] = None
    fit_axis_dir: Optional[Base.Vector] = None
    fit_radius: float = 0.0
    fit_deviation: float = float('inf')
    replacement_face: object = None


@dataclass
class EdgeInfo:
    """边级别分析结果"""
    edge_index: int
    edge: object = None
    original_type: str = ""         # Line, BSplineCurve 等
    is_circular: bool = False
    fit_center: Optional[Base.Vector] = None
    fit_radius: float = 0.0
    fit_normal: Optional[Base.Vector] = None
    fit_deviation: float = float('inf')
    replacement_edge: object = None


# ============================================================================
# Kasa 圆拟合核心（与 AnalysisSitus FitCircle 同源思路）
# ============================================================================

def fit_circle_kasa(points):
    """
    Kasa 最小二乘圆拟合。
    x^2 + y^2 + Dx + Ey + F = 0
    返回 (center_3d_Vector, radius, normal_Vector, max_deviation) 或 None。
    """
    n = len(points)
    if n < 6:
        return None

    coords_3d = np.array([(p.x, p.y, p.z) for p in points])
    centroid = coords_3d.mean(axis=0)
    centered = coords_3d - centroid

    _, _, vh = np.linalg.svd(centered)
    normal_vec = vh[2]
    u_vec, v_vec = vh[0], vh[1]

    coords_2d = centered @ np.column_stack([u_vec, v_vec])
    x, y = coords_2d[:, 0], coords_2d[:, 1]

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


def fit_cylinder(points_3d, normals=None):
    """
    圆柱面拟合（参考 AnalysisSitus CheckIsCylindrical）。
    1. SVD 找主方向（轴方向 = 最小变化方向）
    2. 投影到垂直平面，Kasa 拟合圆
    3. 验证法向量一致性
    返回 (axis_point, axis_dir, radius, max_deviation) 或 None。
    """
    n = len(points_3d)
    if n < 10:
        return None

    coords = np.array(points_3d)
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    # SVD 找主方向
    _, s, vh = np.linalg.svd(centered)
    axis_dir = vh[2]  # 最小变化方向 = 圆柱轴

    # 投影到垂直于轴的平面
    proj = centered - np.outer(centered @ axis_dir, axis_dir)
    radii = np.linalg.norm(proj, axis=1)
    r_mean = radii.mean()
    r_std = radii.std()

    if r_mean < 1e-6 or r_mean > 1e6:
        return None

    # 检查半径一致性
    max_dev = abs(radii - r_mean).max()
    if max_dev > r_mean * 0.3:  # 偏差超过30%不是圆柱
        return None

    # 检查法向量是否垂直于轴（圆柱法向量应径向朝外）
    if normals is not None:
        normals_arr = np.array(normals)
        # 法向量与轴的点积应接近0
        dot_with_axis = np.abs(normals_arr @ axis_dir)
        if dot_with_axis.mean() > 0.3:
            return None

    axis_point = Base.Vector(centroid[0], centroid[1], centroid[2])
    axis_d = Base.Vector(axis_dir[0], axis_dir[1], axis_dir[2])

    return axis_point, axis_d, r_mean, max_dev


# ============================================================================
# 边界 Wire 提取
# ============================================================================

class BoundaryWireExtractor:
    """从 B-Rep Shape 中提取开放边界 wires"""

    def __init__(self, shape):
        self.shape = shape

    def extract(self) -> List[List]:
        edge_count = {}
        for face in self.shape.Faces:
            for edge in face.Edges:
                h = edge.hashCode()
                edge_count[h] = edge_count.get(h, 0) + 1

        boundary_edges = []
        seen = set()
        for face in self.shape.Faces:
            for edge in face.Edges:
                h = edge.hashCode()
                if edge_count[h] == 1 and h not in seen:
                    seen.add(h)
                    boundary_edges.append(edge)

        if not boundary_edges:
            return []

        return self._group_into_wires(boundary_edges)

    def _group_into_wires(self, edges: list) -> List[List]:
        wires = []
        used = set()
        for i, e1 in enumerate(edges):
            if i in used:
                continue
            wire = [e1]
            used.add(i)
            changed = True
            while changed:
                changed = False
                for j, e2 in enumerate(edges):
                    if j in used:
                        continue
                    for we in wire:
                        if self._edges_connected(we, e2):
                            wire.append(e2)
                            used.add(j)
                            changed = True
                            break
            if len(wire) >= 3:
                wires.append(wire)
        return wires

    @staticmethod
    def _edges_connected(e1, e2, tol=0.01) -> bool:
        pts1 = [e1.Vertexes[0].Point, e1.Vertexes[-1].Point]
        pts2 = [e2.Vertexes[0].Point, e2.Vertexes[-1].Point]
        for p1 in pts1:
            for p2 in pts2:
                if p1.distanceToPoint(p2) < tol:
                    return True
        return False


# ============================================================================
# Wire 级别圆弧检测
# ============================================================================

class ArcDetector:
    """检测 edge wire 是否为圆弧"""

    def __init__(self, tolerance: float = 0.5, min_edges: int = 6):
        self.tolerance = tolerance
        self.min_edges = min_edges

    def detect(self, edges: list) -> Optional[WireInfo]:
        if len(edges) < self.min_edges:
            return None

        pts = []
        for e in edges:
            pts.extend(e.discretize(20))

        if len(pts) < 10:
            return None

        perimeter = sum(e.Length for e in edges)
        centroid = Base.Vector(0, 0, 0)
        for p in pts:
            centroid += p
        centroid /= len(pts)

        fit = fit_circle_kasa(pts)
        if fit is None:
            return WireInfo(edges=edges, points=pts, perimeter=perimeter, centroid=centroid)

        center, radius, normal, max_dev = fit
        is_circ = max_dev <= self.tolerance

        info = WireInfo(
            edges=edges, points=pts, perimeter=perimeter, centroid=centroid,
            is_circular=is_circ, fit_center=center, fit_radius=radius,
            fit_normal=normal, fit_deviation=max_dev,
        )

        if is_circ:
            circ = Part.Circle()
            circ.Center = center
            circ.Axis = Base.Vector(normal[0], normal[1], normal[2])
            circ.Radius = radius
            info.replacement_edge = circ.toShape()

        return info


# ============================================================================
# 面级别圆柱检测（参考 AnalysisSitus CheckIsCylindrical）
# ============================================================================

class FaceAnalyzer:
    """
    分析每个面是否为碎面圆柱。
    算法：
    1. 采样面上的点和法向量
    2. SVD 找主方向
    3. 检查点到轴的距离是否一致（圆柱特征）
    4. 检查法向量是否径向朝外
    """

    def __init__(self, tolerance: float = 0.3):
        self.tolerance = tolerance

    def analyze(self, face, face_index: int) -> Optional[FaceInfo]:
        surf = face.Surface
        stype = type(surf).__name__

        info = FaceInfo(face_index=face_index, face=face, original_type=stype)

        # 只分析平面和 BSpline 面（已经是圆柱的跳过）
        if stype == "Geom_CylindricalSurface":
            info.is_cylindrical = True
            info.fit_axis_dir = surf.Axis.Direction
            info.fit_radius = surf.Radius
            return info

        if stype not in ("Geom_Plane", "Geom_BSplineSurface"):
            return info

        # 采样点和法向量
        u0, u1, v0, v1 = face.ParameterRange
        pts = []
        normals = []
        for ui in range(8):
            for vi in range(8):
                u = u0 + (u1 - u0) * (ui + 0.5) / 8
                v = v0 + (v1 - v0) * (vi + 0.5) / 8
                try:
                    p = face.valueAt(u, v)
                    n = face.normalAt(u, v)
                    pts.append([p.x, p.y, p.z])
                    normals.append([n.x, n.y, n.z])
                except:
                    continue

        if len(pts) < 10:
            return info

        # 圆柱拟合
        fit = fit_cylinder(pts, normals)
        if fit is None:
            return info

        axis_point, axis_dir, radius, max_dev = fit

        # 偏差检查
        if max_dev > self.tolerance:
            return info

        info.is_cylindrical = True
        info.fit_axis_point = axis_point
        info.fit_axis_dir = axis_dir
        info.fit_radius = radius
        info.fit_deviation = max_dev

        # 创建替换圆柱面
        try:
            cyl = Part.Cylinder()
            cyl.Radius = radius
            cyl.Axis = axis_dir
            cyl.Center = axis_point
            info.replacement_face = cyl.toShape()
        except:
            info.replacement_face = None

        return info


# ============================================================================
# 边级别圆弧检测（参考 AnalysisSitus FitCircle）
# ============================================================================

class EdgeAnalyzer:
    """
    分析每条边是否为碎边圆弧。
    算法：
    1. 离散化边
    2. Kasa 圆拟合
    3. 偏差检查
    """

    def __init__(self, tolerance: float = 0.3, min_points: int = 10):
        self.tolerance = tolerance
        self.min_points = min_points

    def analyze(self, edge, edge_index: int) -> Optional[EdgeInfo]:
        etype = type(edge.Curve).__name__
        info = EdgeInfo(edge_index=edge_index, edge=edge, original_type=etype)

        # 只分析直线和 BSpline（已经是圆的跳过）
        if isinstance(edge.Curve, Part.Circle):
            info.is_circular = True
            info.fit_center = edge.Curve.Center
            info.fit_radius = edge.Curve.Radius
            info.fit_normal = edge.Curve.Axis
            return info

        if etype not in ("Geom_Line", "Geom_BSplineCurve", "Geom_LineSegment"):
            return info

        # 离散化
        n_pts = max(self.min_points, int(edge.Length / 0.5))
        pts = edge.discretize(n_pts)
        if len(pts) < self.min_points:
            return info

        # Kasa 圆拟合
        fit = fit_circle_kasa(pts)
        if fit is None:
            return info

        center, radius, normal, max_dev = fit

        if max_dev > self.tolerance:
            return info

        info.is_circular = True
        info.fit_center = center
        info.fit_radius = radius
        info.fit_normal = normal
        info.fit_deviation = max_dev

        # 创建替换圆弧边
        try:
            circ = Part.Circle()
            circ.Center = center
            circ.Axis = Base.Vector(normal[0], normal[1], normal[2])
            circ.Radius = radius
            # 用原始边的参数范围
            p0 = edge.valueAt(edge.FirstParameter)
            p1 = edge.valueAt(edge.LastParameter)
            u0 = circ.parameter(p0)
            u1 = circ.parameter(p1)
            info.replacement_edge = circ.toShape(u0, u1)
        except:
            try:
                circ = Part.Circle()
                circ.Center = center
                circ.Axis = Base.Vector(normal[0], normal[1], normal[2])
                circ.Radius = radius
                info.replacement_edge = circ.toShape()
            except:
                info.replacement_edge = None

        return info


# ============================================================================
# 重建器
# ============================================================================

class HoleRebuilder:
    """B-Rep 重建器"""

    @staticmethod
    def rebuild_face_from_wire(shape, wire_info: WireInfo):
        circ_edge = wire_info.replacement_edge
        if circ_edge is None:
            return None
        wire = Part.Wire(circ_edge)
        if not wire.isClosed():
            wire = Part.Wire(Part.sortEdges([circ_edge])[0])
        try:
            return Part.Face(wire)
        except:
            return None

    @staticmethod
    def replace_edge(shape, edge_info: EdgeInfo, edge_index: int):
        """替换一条边为圆弧边"""
        if edge_info.replacement_edge is None:
            return None
        return edge_info.replacement_edge

    @staticmethod
    def replace_face(shape, face_info: FaceInfo, face_index: int):
        """替换一个面为圆柱面"""
        if face_info.replacement_face is None:
            return None
        return face_info.replacement_face


# ============================================================================
# GUI 对话框
# ============================================================================

class HoleRepairDialog(QtWidgets.QDialog):
    """B-Rep 几何修复对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("B-Rep 圆弧/圆柱检测与重建")
        self.setMinimumSize(620, 700)

        self.shape_obj = None
        self.wire_infos: List[WireInfo] = []
        self.face_infos: List[FaceInfo] = []
        self.edge_infos: List[EdgeInfo] = []

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
        gl.addWidget(btn_refresh)
        grp_obj.setLayout(gl)
        lay.addWidget(grp_obj)

        # 参数
        grp_param = QtWidgets.QGroupBox("检测参数")
        fl = QtWidgets.QFormLayout()

        self.spin_tol = QtWidgets.QDoubleSpinBox()
        self.spin_tol.setRange(0.01, 10.0)
        self.spin_tol.setValue(0.5)
        self.spin_tol.setSingleStep(0.1)
        fl.addRow("圆弧容差 (mm):", self.spin_tol)

        self.spin_min = QtWidgets.QSpinBox()
        self.spin_min.setRange(3, 200)
        self.spin_min.setValue(6)
        fl.addRow("最小边/点数:", self.spin_min)

        grp_param.setLayout(fl)
        lay.addWidget(grp_param)

        # 检测模式
        grp_mode = QtWidgets.QGroupBox("检测模式")
        ml = QtWidgets.QHBoxLayout()
        self.chk_wire = QtWidgets.QCheckBox("Wire 圆弧孔洞")
        self.chk_wire.setChecked(True)
        ml.addWidget(self.chk_wire)
        self.chk_face = QtWidgets.QCheckBox("面 圆柱检测")
        self.chk_face.setChecked(True)
        ml.addWidget(self.chk_face)
        self.chk_edge = QtWidgets.QCheckBox("边 圆弧检测")
        self.chk_edge.setChecked(True)
        ml.addWidget(self.chk_edge)
        grp_mode.setLayout(ml)
        lay.addWidget(grp_mode)

        # 按钮
        bl = QtWidgets.QHBoxLayout()
        self.btn_detect = QtWidgets.QPushButton("检测")
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
        self.log_text.setMaximumHeight(100)
        ll.addWidget(self.log_text)
        grp_log.setLayout(ll)
        lay.addWidget(grp_log)

        # 存储所有结果（带类型标记）
        self.all_results = []  # list of (type_str, index, info)

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
        tol = self.spin_tol.value()
        min_n = self.spin_min.value()

        self._log("Shape: %d Faces, %d Edges, %d Solids" % (
            len(shape.Faces), len(shape.Edges), len(shape.Solids)))

        self.all_results = []
        circ_count = 0

        # Wire 级别
        if self.chk_wire.isChecked():
            extractor = BoundaryWireExtractor(shape)
            wires = extractor.extract()
            self._log("边界 wires: %d" % len(wires))
            detector = ArcDetector(tolerance=tol, min_edges=min_n)
            self.wire_infos = []
            for wire_edges in wires:
                info = detector.detect(wire_edges)
                if info is not None:
                    self.wire_infos.append(info)
                    if info.is_circular:
                        circ_count += 1
                        self.all_results.append(("wire", len(self.wire_infos)-1, info))

        # 面级别
        if self.chk_face.isChecked():
            analyzer = FaceAnalyzer(tolerance=tol)
            self.face_infos = []
            for i, face in enumerate(shape.Faces):
                info = analyzer.analyze(face, i)
                self.face_infos.append(info)
                if info.is_cylindrical and info.original_type in ("Geom_Plane", "Geom_BSplineSurface"):
                    circ_count += 1
                    self.all_results.append(("face", i, info))

        # 边级别
        if self.chk_edge.isChecked():
            analyzer = EdgeAnalyzer(tolerance=tol, min_points=min_n)
            self.edge_infos = []
            for i, edge in enumerate(shape.Edges):
                info = analyzer.analyze(edge, i)
                self.edge_infos.append(info)
                if info.is_circular and info.original_type in ("Geom_Line", "Geom_BSplineCurve", "Geom_LineSegment"):
                    circ_count += 1
                    self.all_results.append(("edge", i, info))

        # 更新 UI
        self.result_list.clear()
        for rtype, idx, info in self.all_results:
            if rtype == "wire":
                c = info.fit_center
                text = "[Wire 圆弧] R=%.4f 中心=(%.2f,%.2f,%.2f) 偏差=%.4f %d边" % (
                    info.fit_radius, c.x, c.y, c.z, info.fit_deviation, len(info.edges))
            elif rtype == "face":
                text = "[面 圆柱] Face%d 原始=%s R=%.4f 偏差=%.4f" % (
                    idx, info.original_type, info.fit_radius, info.fit_deviation)
            elif rtype == "edge":
                text = "[边 圆弧] Edge%d 原始=%s R=%.4f 偏差=%.4f" % (
                    idx, info.original_type, info.fit_radius, info.fit_deviation)
            else:
                continue

            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, (rtype, idx))
            item.setForeground(QtCore.Qt.darkGreen)
            self.result_list.addItem(item)

        self.lbl_stats.setText("检测到 %d 个可修复的几何" % circ_count)
        self.btn_rebuild_all.setEnabled(circ_count > 0)
        self._log("检测完成: %d 个 Wire 圆弧, %d 个面圆柱, %d 个边圆弧" % (
            len(self.wire_infos) if hasattr(self, 'wire_infos') else 0,
            sum(1 for f in self.face_infos if f.is_cylindrical and f.original_type in ("Geom_Plane", "Geom_BSplineSurface")) if hasattr(self, 'face_infos') else 0,
            sum(1 for e in self.edge_infos if e.is_circular and e.original_type in ("Geom_Line", "Geom_BSplineCurve", "Geom_LineSegment")) if hasattr(self, 'edge_infos') else 0,
        ))

    def _on_sel(self):
        sel = self.result_list.selectedItems()
        self.btn_rebuild.setEnabled(len(sel) > 0)

    def _rebuild_sel(self):
        items = self.result_list.selectedItems()
        self._do_rebuild(items)

    def _rebuild_all(self):
        items = [self.result_list.item(i) for i in range(self.result_list.count())]
        self._do_rebuild(items)

    def _do_rebuild(self, items):
        if not self.shape_obj or not items:
            return

        doc = FreeCAD.ActiveDocument
        shape = self.shape_obj.Shape
        faces = list(shape.Faces)
        modified = False

        for item in items:
            rtype, idx = item.data(QtCore.Qt.UserRole)

            if rtype == "wire":
                info = self.wire_infos[idx]
                if info.is_circular and info.replacement_edge:
                    face = HoleRebuilder.rebuild_face_from_wire(shape, info)
                    if face:
                        faces.append(face)
                        modified = True
                        self._log("重建 Wire: R=%.4f" % info.fit_radius)

            elif rtype == "face":
                info = self.face_infos[idx]
                if info.is_cylindrical and info.replacement_face:
                    faces[idx] = info.replacement_face
                    modified = True
                    self._log("替换 Face%d: %s -> Cylinder R=%.4f" % (
                        idx, info.original_type, info.fit_radius))

            elif rtype == "edge":
                # 边替换需要重建整个 shape，这里先标记
                info = self.edge_infos[idx]
                if info.is_circular and info.replacement_edge:
                    self._log("Edge%d 标记为圆弧 R=%.4f (需要重建拓扑)" % (idx, info.fit_radius))

        if not modified:
            self._log("没有可重建的几何")
            return

        # 重建 shape
        try:
            shell = Part.Shell(faces)
            try:
                result = Part.Solid(shell)
            except:
                result = shell
        except:
            result = shape

        new_name = self.shape_obj.Name + "_repaired"
        new_obj = doc.addObject("Part::Feature", new_name)
        new_obj.Shape = result
        doc.recompute()
        self._log("完成: %s" % new_name)

    def _log(self, msg):
        self.log_text.append(msg)
        FreeCAD.Console.PrintMessage("[HoleRepair] %s\n" % msg)


# ============================================================================
# FreeCAD 命令
# ============================================================================

class HoleRepairCommand:
    def GetResources(self):
        return {
            'MenuText': 'B-Rep 圆弧/圆柱检测与重建',
            'ToolTip': '检测碎边圆弧、碎面圆柱、开放边界圆弧孔洞并重建为真正的几何',
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
