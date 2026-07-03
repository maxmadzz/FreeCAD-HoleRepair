# FreeCAD Mesh Hole Repair Plugin

## 功能

检测 Mesh 中的圆弧形孔洞并自动重建。

### 主要特性

1. **边界检测** - 自动识别 Mesh 中的边界边和边界环
2. **圆弧识别** - 使用最小二乘法拟合圆，检测圆弧形孔洞
3. **孔洞重建** - 对圆形孔洞进行扇形三角化填充

## 安装

### 方法 1: 作为 FreeCAD 宏

1. 将 `HoleRepair.py` 复制到 FreeCAD 宏目录：
   - Windows: `%APPDATA%\FreeCAD\Macro\`
   - macOS: `~/Library/Preferences/FreeCAD/Macro/`
   - Linux: `~/.FreeCAD/Macro/`

2. 在 FreeCAD 中运行宏：`宏` -> `宏` -> 选择 `HoleRepair`

### 方法 2: 作为工作台插件

1. 将整个 `FreeCAD-HoleRepair` 目录复制到 FreeCAD Mod 目录：
   - Windows: `%APPDATA%\FreeCAD\Mod\`
   - macOS: `~/Library/Preferences/FreeCAD/Mod/`
   - Linux: `~/.FreeCAD/Mod/`

2. 重启 FreeCAD，在工作台列表中选择 "HoleRepair"

## 使用方法

1. 打开包含 Mesh 对象的 FreeCAD 文档
2. 运行插件（宏或工作台命令）
3. 在对话框中选择要检测的 Mesh 对象
4. 设置检测参数：
   - **圆弧检测容差**: 圆弧拟合的最大允许偏差（默认 0.1mm）
   - **最小顶点数**: 忽略顶点数少于此值的边界环（默认 6）
   - **重建细分段数**: 圆形孔洞重建时的细分段数（默认 32）
5. 点击 "检测孔洞" 按钮
6. 在结果列表中选择要重建的孔洞
7. 点击 "重建选中孔洞" 或 "重建所有圆弧孔洞"

## 算法说明

### 边界检测

- 构建边到面的映射
- 只属于一个面的边为边界边
- 将边界边连接成闭合环

### 圆弧检测

- 对边界环进行 3D 圆拟合（最小二乘法）
- 计算所有顶点到拟合圆的距离
- 如果最大偏差小于容差，则判定为圆弧

### 孔洞重建

- **圆形孔洞**: 使用圆心进行扇形三角化
- **圆弧孔洞**: 使用质心进行扇形三角化

## 依赖

- FreeCAD 0.20+
- numpy

## 文件结构

```
FreeCAD-HoleRepair/
├── __init__.py          # 包初始化
├── Init.py              # FreeCAD 初始化
├── InitGui.py           # FreeCAD 工作台注册
├── HoleRepair.py        # 主插件代码
├── README.md            # 说明文档
└── icons/               # 图标目录
    └── hole_repair.svg  # 插件图标
```

## 许可证

MIT License
