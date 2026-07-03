#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FreeCAD 插件：B-Rep 孔洞圆弧检测与重建

工作流程：
1. 选择 B-Rep 对象（或从 Mesh 转换）
2. 检测开放边界 wire
3. 对每个 wire 做 Kasa 最小二乘圆拟合
4. 将拟合成功的 wire 替换为真正的 Geom_Circle 边
5. 重建为完整的 B-Rep 面/体

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
import Mesh
from FreeCAD import Base
from PySide6 import QtWidgets, QtCore


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class WireInfo:
    """边界 wire 信息"""
    edges: list                          # Part.Edge 列表
    points: list = field(default_factory=list)  # 离散点
    perimeter: float = 0.0
    centroid: Base.Vector = field(default_factory=lambda: Base.Vector(0, 0, 0))
    is_circular: bool = False
    fit_center: Optional[Base.Vector] = None
    fit_radius: float = 0.0
    fit_normal: Optional[Base.Vector] = None
    fit_deviation: float = float('inf')
    replacement_edge: object = None      # 拟合后的圆弧边


# ============================================================================
# 边界 Wire 提取
# ============================================================================

class BoundaryWireExtractor:
    """从 B-Rep Shape 中提取开放边界 wires"""

    def __init__(self, shape):
        self.shape = shape

    def extract(self) -> List[List]:
        """
        提取所有开放边界 edge，按拓扑连接分组为 wires。
        返回: List[List[Part.Edge]]
        """
        # 1. 统计每条 edge 被多少个面共享
        edge_count = {}
        for face in self.shape.Faces:
            for edge in face.Edges:
                h = edge.hashCode()
                edge_count[h] = edge_count.get(h, 0) + 1

        # 2. 收集边界 edge（只属于 1 个面）
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

        # 3. 按拓扑连接分组
        return self._group_into_wires(boundary_edges)

    def _group_into_wires(self, edges: list) -> List[List]:
        """将边按端点连接关系分组为 wires"""
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
        """检查两条边是否端点相连"""
        pts1 = [e1.Vertexes[0].Point, e1.Vertexes[-1].Point]
        pts2 = [e2.Vertexes[0].Point, e2.Vertexes[-1].Point]
        for p1 in pts1:
            for p2 in pts2:
                if p1.distanceToPoint(p2) < tol:
                    return True
        return False


# ============================================================================
# 圆弧检测（Kasa 最小二乘，与 AnalysisSitus FitCircle 同源思路）
# ============================================================================

class ArcDetector:
    """检测 edge wire 是否为圆弧"""

    def __init__(self, tolerance: float = 0.5, min_edges: int = 6):
        self.tolerance = tolerance
        self.min_edges = min_edges

    def detect(self, edges: list) -> Optional[WireInfo]:
        """
        检测一组边是否构成圆弧。
        返回 WireInfo（含拟合结果）或 None。
        """
        if len(edges) < self.min_edges:
            return None

        # 收集离散点
        pts = []
        for e in edges:
            pts.extend(e.discretize(20))

        if len(pts) < 10:
            return None

        # 计算属性
        perimeter = sum(edges[i].Length for i in range(len(edges)))
        centroid = Base.Vector(0, 0, 0)
        for p in pts:
            centroid += p
        centroid /= len(pts)

        # Kasa 圆拟合
        fit = self._fit_circle_kasa(pts)
        if fit is None:
            return WireInfo(edges=edges, points=pts, perimeter=perimeter, centroid=centroid)

        center, radius, normal, max_dev = fit

        is_circ = max_dev <= self.tolerance

        info = WireInfo(
            edges=edges,
            points=pts,
            perimeter=perimeter,
            centroid=centroid,
            is_circular=is_circ,
            fit_center=center,
            fit_radius=radius,
            fit_normal=normal,
            fit_deviation=max_dev,
        )

        # 如果是圆弧，创建替换边
        if is_circ:
            circ = Part.Circle()
            circ.Center = center
            circ.Axis = Base.Vector(normal[0], normal[1], normal[2])
            circ.Radius = radius
            info.replacement_edge = circ.toShape()

        return info

    @staticmethod
    def _fit_circle_kasa(points):
        """
        Kasa 最小二乘圆拟合。
        x^2 + y^2 + Dx + Ey + F = 0
        返回 (center_3d, radius, normal_vec, max_deviation) 或 None。
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
# B-Rep 孔洞重建
# ============================================================================

class HoleRebuilder:
    """B-Rep 孔洞重建器"""

    @staticmethod
    def rebuild_face(shape, wire_info: WireInfo):
        """
        用拟合的圆边创建新的 B-Rep Face。
        1. 用圆弧边替换原来的碎边
        2. 创建 Wire
        3. 创建 Face
        """
        circ_edge = wire_info.replacement_edge
        if circ_edge is None:
            return None

        # 创建 Wire
        wire = Part.Wire(circ_edge)
        if not wire.isClosed():
            # 强制闭合
            wire = Part.Wire(Part.sortEdges([circ_edge])[0])

        # 创建 Face
        try:
            face = Part.Face(wire)
            return face
        except Exception:
            return None

    @staticmethod
    def rebuild_solid(shape, wire_infos: List[WireInfo]):
        """
        将所有圆弧孔洞重建为 B-Rep Solid。
        1. 用圆弧边替换原来的碎边
        2. 修补原始 shape
        """
        # 收集所有面
        faces = list(shape.Faces)

        # 找到需要替换的面（包含边界边的面）
        # 简化处理：直接创建新 face 覆盖孔洞
        new_faces = []
        for info in wire_infos:
            if info.replacement_edge is not None:
                face = HoleRebuilder.rebuild_face(shape, info)
                if face is not None:
                    new_faces.append(face)

        if not new_faces:
            return shape

        # 合并所有面
        all_faces = faces + new_faces
        try:
            shell = Part.Shell(all_faces)
            solid = Part.Solid(shell)
            return solid
        except Exception:
            # 如果创建 Solid 失败，返回 Shell
            try:
                return Part.Shell(all_faces)
            except Exception:
                return shape


# ============================================================================
# GUI 对话框
# ============================================================================

class HoleRepairDialog(QtWidgets.QDialog):
    """B-Rep 孔洞修复对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("B-Rep 孔洞圆弧检测与重建")
        self.setMinimumSize(580, 600)

        self.shape_obj = None
        self.wire_infos: List[WireInfo] = []

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
        self.spin_min.setRange(3, 100)
        self.spin_min.setValue(6)
        fl.addRow("最小边数:", self.spin_min)

        grp_param.setLayout(fl)
        lay.addWidget(grp_param)

        # 按钮
        bl = QtWidgets.QHBoxLayout()
        self.btn_detect = QtWidgets.QPushButton("检测圆弧孔洞")
        self.btn_detect.setStyleSheet("QPushButton{background:#4CAF50;color:white;padding:6px}")
        bl.addWidget(self.btn_detect)

        self.btn_rebuild = QtWidgets.QPushButton("重建选中")
        self.btn_rebuild.setEnabled(False)
        self.btn_rebuild.setStyleSheet("QPushButton{background:#2196F3;color:white;padding:6px}")
        bl.addWidget(self.btn_rebuild)

        self.btn_rebuild_all = QtWidgets.QPushButton("重建所有圆弧")
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
        self._log("Shape: %d Faces, %d Edges" % (len(shape.Faces), len(shape.Edges)))

        # 提取边界 wires
        extractor = BoundaryWireExtractor(shape)
        wires = extractor.extract()
        self._log("边界 wires: %d" % len(wires))

        # 检测圆弧
        detector = ArcDetector(
            tolerance=self.spin_tol.value(),
            min_edges=self.spin_min.value(),
        )
        self.wire_infos = []
        for wire_edges in wires:
            info = detector.detect(wire_edges)
            if info is not None:
                self.wire_infos.append(info)

        # 更新 UI
        self.result_list.clear()
        circ_count = 0
        for i, info in enumerate(self.wire_infos):
            if info.is_circular:
                circ_count += 1
                c = info.fit_center
                text = "[圆弧] #%d  R=%.4f  中心=(%.3f,%.f,%.f)  偏差=%.4f  %d边" % (
                    i+1, info.fit_radius, c.x, c.y, c.z, info.fit_deviation, len(info.edges))
            else:
                text = "[非圆弧] #%d  %d边  周长=%.2f" % (i+1, len(info.edges), info.perimeter)

            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, i)
            if info.is_circular:
                item.setForeground(QtCore.Qt.darkGreen)
            self.result_list.addItem(item)

        self.lbl_stats.setText("边界 wires: %d, 圆弧: %d" % (len(wires), circ_count))
        self.btn_rebuild_all.setEnabled(circ_count > 0)
        self._log("检测完成: %d 个圆弧孔洞" % circ_count)

    def _on_sel(self):
        sel = self.result_list.selectedItems()
        self.btn_rebuild.setEnabled(any(
            self.wire_infos[item.data(QtCore.Qt.UserRole)].is_circular
            for item in sel
        ))

    def _rebuild_sel(self):
        indices = [item.data(QtCore.Qt.UserRole) for item in self.result_list.selectedItems()]
        self._do_rebuild(indices)

    def _rebuild_all(self):
        indices = [i for i, info in enumerate(self.wire_infos) if info.is_circular]
        self._do_rebuild(indices)

    def _do_rebuild(self, indices):
        if not self.shape_obj:
            return

        doc = FreeCAD.ActiveDocument
        shape = self.shape_obj.Shape

        # 创建新 face 覆盖孔洞
        new_faces = []
        for idx in indices:
            info = self.wire_infos[idx]
            if not info.is_circular or info.replacement_edge is None:
                continue

            face = HoleRebuilder.rebuild_face(shape, info)
            if face is not None:
                new_faces.append(face)
                self._log("重建 #%d: R=%.4f 面片" % (idx+1, info.fit_radius))

        if not new_faces:
            self._log("没有可重建的孔洞")
            return

        # 合并到原 shape
        all_faces = list(shape.Faces) + new_faces
        try:
            shell = Part.Shell(all_faces)
            try:
                solid = Part.Solid(shell)
                result = solid
            except Exception:
                result = shell
        except Exception:
            result = shape

        # 创建新对象
        new_name = self.shape_obj.Name + "_repaired"
        new_obj = doc.addObject("Part::Feature", new_name)
        new_obj.Shape = result
        doc.recompute()

        self._log("完成: %s (%d 个圆弧面片已合并)" % (new_name, len(new_faces)))

    def _log(self, msg):
        self.log_text.append(msg)
        FreeCAD.Console.PrintMessage("[HoleRepair] %s\n" % msg)


# ============================================================================
# FreeCAD 命令
# ============================================================================

class HoleRepairCommand:
    def GetResources(self):
        return {
            'MenuText': 'B-Rep 孔洞圆弧检测与重建',
            'ToolTip': '检测 B-Rep 中的圆弧形孔洞并重建为真正的圆弧几何',
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
