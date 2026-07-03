# -*- coding: utf-8 -*-
"""
FreeCAD Workbench InitGui
Mesh 孔洞圆弧检测与重建插件
"""

class HoleRepairWorkbench:
    """FreeCAD 工作台：孔洞修复"""
    
    def __init__(self):
        pass
    
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
    
    def Setup(self):
        """设置工作台"""
        import FreeCADGui
        
        # 创建菜单
        self.appendMenu(
            "孔洞修复",
            ["HoleRepairCommand"]
        )
        
        # 创建工具栏
        self.appendToolbar(
            "孔洞修复",
            ["HoleRepairCommand"]
        )
        
        FreeCAD.Console.PrintMessage("孔洞修复工作台已加载\n")


# 注册工作台
wb = HoleRepairWorkbench()
FreeCADGui.addWorkbench(wb)
