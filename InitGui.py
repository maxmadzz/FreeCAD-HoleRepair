# -*- coding: utf-8 -*-
"""
FreeCAD Workbench InitGui
B-Rep 孔洞圆弧检测与重建插件
"""
import FreeCADGui


class HoleRepairWorkbench(FreeCADGui.Workbench):
    """FreeCAD 工作台：孔洞修复"""

    def GetClassName(self):
        return "Gui::PythonWorkbench"

    def GetName(self):
        return "HoleRepair"

    def IsActive(self):
        return True

    def GetIcon(self):
        import os
        icon_path = os.path.join(os.path.dirname(__file__), 'icons', 'hole_repair.svg')
        if os.path.exists(icon_path):
            return icon_path
        return ""

    def Initialize(self):
        """设置工作台"""
        from HoleRepair import HoleRepairCommand
        FreeCADGui.addCommand('HoleRepairCommand', HoleRepairCommand())

        self.appendMenu("孔洞修复", ["HoleRepairCommand"])
        self.appendToolbar("孔洞修复", ["HoleRepairCommand"])

        FreeCAD.Console.PrintMessage("孔洞修复工作台已加载\n")


# 注册工作台
FreeCADGui.addWorkbench(HoleRepairWorkbench())
