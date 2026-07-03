#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeCAD 插件：STL B-Rep 多边形孔 → 圆弧重建

工作流程（参考 AnalysisSitus asiAlgo_RecognizeCanonical::FitCircle）：
1. 从 B-Rep 中提取边界边（只属于 1 个面的边）
2. 按端点连接分组为闭合环
3. 对每个环做 Kasa 最小二乘圆拟合
4. 相对偏差 < 容差 → 判定为多边形圆弧孔
5. 用 Part.Circle 重建为真正的圆弧 Face

适用场景：STL mesh → B-Rep 转换后，圆孔变成多边形碎边
依赖：FreeCAD 1.0+（内置 numpy）
"""

import os
import math
import numpy as np
from typing import List, Optional, Tuple
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
class BoundaryLoop:
    """边界环信息"""
    edges: list                          # Part.Edge 列表
    points: list = field(default_factory=list)  # 离散点
    n_edges: int = 0
    is_circular: bool = False
    fit_center: Optional[Base.Vector] = None
    fit_radius: float = 0.0
    fit_normal: Optional[Base.Vector] = None
    fit_max_dev: float = float('inf')
    fit_rel_dev: float = float('inf')    # 相对偏差 (%)
    replacement_face: object = None


# ============================================================================
# 边界提取（参考 AnalysisSitus asiAlgo_CompleteEdgeLoop）
# ============================================================================

def extract_boundary_loops(shape, min_edges=3) -> List[List]:
    """
    从 B-Rep Shape 提取开放边界 edge，按拓扑连接分组为闭合环。
    算法：
    1. 统计每条 edge 被多少个面共享
    2. 只属于 1 个面的 edge = 边界边
    3. 按端点连接分组
    """
    # 1. 统计共享面数
    edge_count = {}
    for face in shape.Faces:
        for edge in face.Edges:
            h = edge.hashCode()
            edge_count[h] = edge_count.get(h, 0) + 1

    # 2. 收集边界边
    boundary = []
    seen = set()
    for face in shape.Faces:
        for edge in face.Edges:
            h = edge.hashCode()
            if edge_count[h] == 1 and h not in seen:
                seen.add(h)
                boundary.append(edge)

    if not boundary:
        return []

    # 3. 按端点连接分组
    def edges_connected(e1, e2, tol=0.01):
        for p1 in [e1.Vertexes[0].Point, e1.Vertexes[-1].Point]:
            for p2 in [e2.Vertexes[0].Point, e2.Vertexes[-1].Point]:
                if p1.distanceToPoint(p2) < tol:
                    return True
        return False

    wires = []
    used = set()
    for i, e1 in enumerate(boundary):
        if i in used:
            continue
        wire = [e1]
        used.add(i)
        changed = True
        while changed:
            changed = False
            for j, e2 in enumerate(boundary):
                if j in used:
                    continue
                for we in wire:
                    if edges_connected(we, e2):
                        wire.append(e2)
                        used.add(j)
                        changed = True
                        break
        if len(wire) >= min_edges:
            wires.append(wire)

    return wires


# ============================================================================
# Kasa 圆拟合（与 AnalysisSitus FitCircle 同源思路）
# ============================================================================

def fit_circle_kasa(points) -> Optional[Tuple]:
    """
    Kasa 最小二乘圆拟合。
    x² + y² + Dx + Ey + F = 0
    返回 (center_3d, radius, normal, max_deviation) 或 None。
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
# 圆弧检测与重建
# ============================================================================

class CircularHoleDetector:
    """
    检测 B-Rep 中的多边形圆弧孔并重建。
    参考 AnalysisSitus asiAlgo_RecognizeCanonical。
    """

    def __init__(self, rel_tolerance=30.0, min_edges=8):
        """
        rel_tolerance: 相对偏差容差 (% of radius)
        min_edges: 最小边数（少于此数不判定为圆弧）
        """
        self.rel_tolerance = rel_tolerance
        self.min_edges = min_edges

    def detect_and_rebuild(self, shape) -> List[BoundaryLoop]:
        """
        检测所有边界环，判定圆弧，创建替换面。
        返回 BoundaryLoop 列表。
        """
        loops = extract_boundary_loops(shape, min_edges=3)
        results = []

        for loop_edges in loops:
            # 收集离散点
            pts = []
            for e in loop_edges:
                pts.extend(e.discretize(20))

            if len(pts) < 10:
                continue

            # Kasa 圆拟合
            fit = fit_circle_kasa(pts)
            if fit is None:
                results.append(BoundaryLoop(
                    edges=loop_edges, points=pts, n_edges=len(loop_edges)
                ))
                continue

            center, radius, normal, max_dev = fit
            rel_dev = max_dev / radius * 100

            info = BoundaryLoop(
                edges=loop_edges,
                points=pts,
                n_edges=len(loop_edges),
                is_circular=(rel_dev < self.rel_tolerance and len(loop_edges) >= self.min_edges),
                fit_center=center,
                fit_radius=radius,
                fit_normal=normal,
                fit_max_dev=max_dev,
                fit_rel_dev=rel_dev,
            )

            # 创建替换面
            if info.is_circular:
                try:
                    circ = Part.Circle()
                    circ.Center = center
                    circ.Axis = normal
                    circ.Radius = radius
                    circ_edge = circ.toShape()
                    wire = Part.Wire(circ_edge)
                    info.replacement_face = Part.Face(wire)
                except:
                    info.replacement_face = None

            results.append(info)

        return results


# ============================================================================
# GUI 对话框
# ============================================================================

class HoleRepairDialog(QtWidgets.QDialog):
    """STL B-Rep 多边形孔圆弧重建对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("STL B-Rep 多边形孔 → 圆弧重建")
        self.setMinimumSize(600, 650)

        self.shape_obj = None
        self.results: List[BoundaryLoop] = []

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
        self.spin_tol.setRange(1.0, 100.0)
        self.spin_tol.setValue(30.0)
        self.spin_tol.setSingleStep(5.0)
        self.spin_tol.setSuffix("%")
        fl.addRow("相对偏差容差:", self.spin_tol)

        self.spin_min = QtWidgets.QSpinBox()
        self.spin_min.setRange(3, 200)
        self.spin_min.setValue(8)
        fl.addRow("最小边数:", self.spin_min)

        grp_param.setLayout(fl)
        lay.addWidget(grp_param)

        # 按钮
        bl = QtWidgets.QHBoxLayout()
        self.btn_detect = QtWidgets.QPushButton("检测多边形圆弧孔")
        self.btn_detect.setStyleSheet("QPushButton{background:#4CAF50;color:white;padding:6px}")
        bl.addWidget(self.btn_detect)

        self.btn_rebuild = QtWidgets.QPushButton("重建选中")
        self.btn_rebuild.setEnabled(False)
        self.btn_rebuild.setStyleSheet("QPushButton{background:#2196F3;color:white;padding:6px}")
        bl.addWidget(self.btn_rebuild)

        self.btn_rebuild_all = QtWidgets.QPushButton("重建所有圆弧孔")
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
        self._log("Shape: %d Faces, %d Edges, %d Solids" % (
            len(shape.Faces), len(shape.Edges), len(shape.Solids)))

        detector = CircularHoleDetector(
            rel_tolerance=self.spin_tol.value(),
            min_edges=self.spin_min.value(),
        )
        self.results = detector.detect_and_rebuild(shape)

        # 更新 UI
        self.result_list.clear()
        circ_count = 0
        for i, info in enumerate(self.results):
            if info.is_circular:
                circ_count += 1
                c = info.fit_center
                text = "[圆弧孔] #%d  R=%.3f  中心=(%.1f,%.1f,%.1f)  偏差=%.1f%%  %d边" % (
                    i+1, info.fit_radius, c.x, c.y, c.z, info.fit_rel_dev, info.n_edges)
            else:
                text = "[非圆弧] #%d  %d边" % (i+1, info.n_edges)

            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, i)
            if info.is_circular:
                item.setForeground(QtCore.Qt.darkGreen)
            self.result_list.addItem(item)

        self.lbl_stats.setText("边界环: %d, 圆弧孔: %d" % (len(self.results), circ_count))
        self.btn_rebuild_all.setEnabled(circ_count > 0)
        self._log("检测完成: %d 个边界环, %d 个圆弧孔" % (len(self.results), circ_count))

    def _on_sel(self):
        sel = self.result_list.selectedItems()
        self.btn_rebuild.setEnabled(any(
            self.results[item.data(QtCore.Qt.UserRole)].is_circular
            for item in sel
        ))

    def _rebuild_sel(self):
        indices = [item.data(QtCore.Qt.UserRole) for item in self.result_list.selectedItems()]
        self._do_rebuild(indices)

    def _rebuild_all(self):
        indices = [i for i, info in enumerate(self.results) if info.is_circular]
        self._do_rebuild(indices)

    def _do_rebuild(self, indices):
        if not self.shape_obj:
            return

        doc = FreeCAD.ActiveDocument
        count = 0

        for idx in indices:
            info = self.results[idx]
            if not info.is_circular or info.replacement_face is None:
                continue

            new_name = "%s_CircleHole_%d" % (self.shape_obj.Name, idx)
            new_obj = doc.addObject("Part::Feature", new_name)
            new_obj.Shape = info.replacement_face
            count += 1
            self._log("重建 #%d: R=%.3f Face" % (idx+1, info.fit_radius))

        if count > 0:
            doc.recompute()
            self._log("完成: %d 个圆弧孔已重建" % count)
        else:
            self._log("没有可重建的圆弧孔")

    def _log(self, msg):
        self.log_text.append(msg)
        FreeCAD.Console.PrintMessage("[HoleRepair] %s\n" % msg)


# ============================================================================
# FreeCAD 命令
# ============================================================================

class HoleRepairCommand:
    def GetResources(self):
        return {
            'MenuText': 'STL B-Rep 多边形孔 → 圆弧重建',
            'ToolTip': '检测 STL 转换的 B-Rep 中的多边形孔，重建为真正的圆弧几何',
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
