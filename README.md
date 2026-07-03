# FreeCAD B-Rep Hole Repair Plugin

## 功能

检测 B-Rep 模型中的圆弧形孔洞，并重建为真正的 `Geom_Circle` 几何。

### 工作流程

1. **Mesh → B-Rep 转换**后，边界边全是 Line/BSpline 碎边
2. **边界 Wire 提取**：按拓扑连接分组开放边
3. **Kasa 圆拟合**：与 AnalysisSitus `asiAlgo_RecognizeCanonical::FitCircle` 同源思路
4. **B-Rep 重建**：将碎边替换为真正的 `Geom_Circle` 边，创建完整 Face

### 核心算法

| 步骤 | 方法 | 说明 |
|------|------|------|
| 边界提取 | `BoundaryWireExtractor` | 按 `hashCode` 统计共享面数，只取 1 面边；按端点连接分组 |
| 圆拟合 | `ArcDetector._fit_circle_kasa` | SVD 找拟合平面 → 2D 投影 → Kasa 最小二乘 `x²+y²+Dx+Ey+F=0` |
| 偏差验证 | `max_deviation ≤ tolerance` | 与 AnalysisSitus 的 20 点验证一致 |
| 重建 | `HoleRebuilder.rebuild_face` | `Part.Circle` → `toShape()` → `Part.Wire` → `Part.Face` |

## 安装

```bash
# 复制到 FreeCAD Mod 目录
cp -r FreeCAD-HoleRepair ~/Library/Preferences/FreeCAD/Mod/
# 重启 FreeCAD，选择 "HoleRepair" 工作台
```

## 使用

1. 打开包含 B-Rep 对象的文档
2. 运行插件
3. 选择对象 → 设置容差 → 点击 "检测圆弧孔洞"
4. 选中结果 → 点击 "重建选中" 或 "重建所有圆弧"

## 依赖

- FreeCAD 1.0+（内置 OCCT 7.8、numpy）
- 无需额外安装

## 许可证

MIT License
