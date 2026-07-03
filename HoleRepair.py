#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeCAD 插件：B-Rep 多边形孔 → 圆孔重建

工作流程（参考 AnalysisSitus asiAlgo_RecognizeCanonical::FitCircle）：
1. 遍历 B-Rep 每个 Face
2. 检查 Face 内部 Wire（inner wire = 孔洞边界）
3. 对每个 inner wire 做 Kasa 最小二乘圆拟合
4. 相对偏差 < 容差 → 判定为多边形圆弧孔
5. 用 Part.Circle 替换 inner wire，重建 Face

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
class HoleInfo:
    """孔洞信息"""
    face_index: int                     # 所在面索引
    face: object = None                 # Part.Face
    wire_index: int = 0                 # inner wire 索引
    wire: object = None                 # Part.Wire
    n_edges: int = 0
    points: list = field(default_factory=list)
    is_circular: bool = False
    fit_center: Optional[Base.Vector] = None
    fit_radius: float = 0.0
    fit_normal: Optional[Base.Vector] = None
    fit_max_dev: float = float('inf')
    fit_rel_dev: float = float('inf')   # 相对偏差 (%)
    replacement_face: object = None     # 重建后的面


# ============================================================================
# Kasa 最小二乘圆拟合（与 AnalysisSitus FitCircle 同源思路）
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
# 孔洞检测：遍历 Face 的 inner Wire
# ============================================================================

class HoleDetector:
    """
    检测 B-Rep Face 中的多边形孔洞（inner Wire）。
    参考 AnalysisSitus asiAlgo_RecognizeCanonical。
    """

    def __init__(self, rel_tolerance=30.0, min_edges=6):
        self.rel_tolerance = rel_tolerance
        self.min_edges = min_edges

    def detect(self, shape) -> List[HoleInfo]:
        """
        遍历所有 Face，检查 inner Wire。
        返回 HoleInfo 列表。
        """
        results = []

        for fi, face in enumerate(shape.Faces):
            wires = face.Wires
            if len(wires) < 2:
                continue  # 没有 inner wire

            # Wire[0] 是外边界，Wire[1:] 是内孔
            for wi in range(1, len(wires)):
                wire = wires[wi]
                edges = wire.Edges
                if len(edges) < self.min_edges:
                    continue

                # 收集离散点
                # 只用顶点（圆应过顶点）
                pts = []
                for v in wire.Vertexes:
                    pts.append(v.Point)

                if len(pts) < self.min_edges:
                    continue

                # Kasa 圆拟合
                fit = fit_circle_kasa(pts)
                if fit is None:
                    continue

                center, radius, normal, max_dev = fit
                rel_dev = max_dev / radius * 100 if radius > 0 else float('inf')

                is_circ = rel_dev < self.rel_tolerance

                info = HoleInfo(
                    face_index=fi,
                    face=face,
                    wire_index=wi,
                    wire=wire,
                    n_edges=len(edges),
                    points=pts,
                    is_circular=is_circ,
                    fit_center=center,
                    fit_radius=radius,
                    fit_normal=normal,
                    fit_max_dev=max_dev,
                    fit_rel_dev=rel_dev,
                )

                # 创建替换面
                if is_circ:
                    try:
                        circ = Part.Circle()
                        circ.Center = center
                        circ.Axis = Base.Vector(normal[0], normal[1], normal[2])
                        circ.Radius = radius
                        circ_edge = circ.toShape()
                        new_wire = Part.Wire(circ_edge)

                        # 创建带圆孔的新面
                        outer_wire = face.Wires[0]
                        if len(face.Wires) > 2:
                            # 多个内孔：保留其他内孔
                            other_inner = [face.Wires[j] for j in range(1, len(face.Wires)) if j != wi]
                            new_face = Part.Face([outer_wire, new_wire] + other_inner)
                        else:
                            new_face = Part.Face([outer_wire, new_wire])
                        info.replacement_face = new_face
                    except Exception:
                        info.replacement_face = None

                results.append(info)

        return results


# ============================================================================
# GUI 对话框
# ============================================================================

class HoleRepairDialog(QtWidgets.QDialog):
    """B-Rep 多边形孔 → 圆孔重建对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("B-Rep 多边形孔 → 圆孔重建")
        self.setMinimumSize(600, 650)

        self.shape_obj = None
        self.results: List[HoleInfo] = []

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
        fl.addRow("相对偏差容差:", self.spin_tol)

        self.spin_min = QtWidgets.QSpinBox()
        self.spin_min.setRange(3, 200)
        self.spin_min.setValue(6)
        fl.addRow("最小边数:", self.spin_min)

        grp_param.setLayout(fl)
        lay.addWidget(grp_param)

        # 按钮
        bl = QtWidgets.QHBoxLayout()
        self.btn_detect = QtWidgets.QPushButton("检测多边形孔")
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

        detector = HoleDetector(
            rel_tolerance=self.spin_tol.value(),
            min_edges=self.spin_min.value(),
        )
        self.results = detector.detect(shape)

        # 更新 UI
        self.result_list.clear()
        circ_count = 0
        for i, info in enumerate(self.results):
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

            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, i)
            if info.is_circular:
                item.setForeground(QtCore.Qt.darkGreen)
            self.result_list.addItem(item)

        self.lbl_stats.setText("检测到 %d 个内孔, %d 个圆弧" % (len(self.results), circ_count))
        self.btn_rebuild_all.setEnabled(circ_count > 0)
        self._log("检测完成: %d 个内孔, %d 个圆弧孔" % (len(self.results), circ_count))

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
        if not self.shape_obj or not indices:
            return

        doc = FreeCAD.ActiveDocument
        shape = self.shape_obj.Shape
        faces = list(shape.Faces)
        count = 0

        for idx in indices:
            info = self.results[idx]
            if not info.is_circular or info.replacement_face is None:
                continue

            # 替换原 face
            faces[info.face_index] = info.replacement_face
            count += 1
            self._log("重建 Face%d/Wire%d: R=%.3f" % (
                info.face_index, info.wire_index, info.fit_radius))

        if count == 0:
            self._log("没有可重建的圆弧孔")
            return

        # 从新 faces 重建 shape
        try:
            shell = Part.Shell(faces)
            try:
                result = Part.Solid(shell)
            except Exception:
                result = shell
        except Exception:
            result = shape

        new_name = self.shape_obj.Name + "_repaired"
        new_obj = doc.addObject("Part::Feature", new_name)
        new_obj.Shape = result
        doc.recompute()

        self._log("完成: %s (%d 个圆弧孔已重建)" % (new_name, count))

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
            'ToolTip': '检测 B-Rep 中的多边形孔洞，重建为真正的圆弧几何',
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
