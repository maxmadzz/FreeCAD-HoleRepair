# FreeCAD B-Rep Hole Repair Plugin

## 功能

检测 B-Rep 模型中的圆弧形孔洞（参数化内环 & mesh→B-Rep 碎面聚类），
并重建为 `Geom_Circle` 真圆几何。

### 工作流程

1. **双重检测模式**：
   - **模式 A（Wire 检测）**：遍历每个 Face 的 inner Wire，使用 Kasa 最小二乘圆拟合。
   - **模式 B（面聚类检测）**：对 mesh→B-Rep 碎面构建邻接图 → BFS 聚类 → SVD 径向一致性 → 法线方向判别。
2. **精确轴线提取**（参考 AnalysisSitus `CheckIsCylindrical`）：
   - 有限差分求曲面主曲率（k₁≈0, k₂=1/R）
   - 法线叉积求圆柱轴线（`dSu × dSv` 法线平面 → SVD 最小方差方向）
   - 内孔 vs 外圆判别（参考 `isInternal`）：面法线指向轴线 → 内孔
3. **Ring+Pocket 重建**：
   - 创建环形（R×1.2 − R×0.8）融合填充多边形间隙
   - 创建精确圆柱（R×1.0）切割真圆孔
   - 支持任意轴向（非 Z 轴孔洞）
   - 零 margin，`Part.RefineShape` 清理拓扑

### 核心算法

| 步骤 | 方法 | 参考 |
|------|------|------|
| 内环圆拟合 | `fit_circle_kasa`（SVD 平面 + Kasa） | `asiAlgo_RecognizeCanonical::FitCircle` |
| 曲率分析 | `_principal_curvatures`（FFF+SFF 有限差分） | `CheckIsCylindrical::EvaluateCurvature` |
| 法线轴线 | `_cylinder_axis_from_normals`（法线叉积） | `CheckIsCylindrical: axis = isCurvedU ? D1v : D1u` |
| 面聚类 | `FaceClusterDetector`（邻接图 + BFS） | `asiAlgo_AAG`（Attribute Adjacency Graph） |
| 内孔判别 | `_check_is_bore`（面法线·径向方向） | `asisAlgo_RecognizeDrillHoles::isInternal` |
| 轴线精化 | 法线 SVD（面法线最小方差方向） | 法线平面法向量 = 轴线 |
| 深度检测 | `_get_hole_extent`（邻接面顶点沿轴投影） | `visitNeighborCylinders` + `gp_Ax1` |

## 安装

```bash
# 复制到 FreeCAD Mod 目录
cp -r FreeCAD-HoleRepair ~/Library/Preferences/FreeCAD/Mod/
# 重启 FreeCAD，选择 "HoleRepair" 工作台
```

## 使用

1. 打开包含 B-Rep 对象的文档（或从 STL → 网格 → ShapeBuilder 转换）
2. 从工具栏点击 **B-Rep 多边形孔 → 圆孔重建**
3. 选择对象 → 设置检测参数
4. 点击 **检测孔洞** → 选中结果 → 点击 **重建选中** 或 **重建所有**

### 检测参数

| 参数 | 默认 | 说明 |
|------|------|------|
| 圆拟合相对偏差容差 | 30% | Kasa 拟合最大偏差/半径 |
| 最小边数/聚类面数 | 6 | 过滤噪声面 |
| 最大圆弧半径 | 20 mm | 跳过大型功能开口 |
| 小平面面积阈值 (模式 B) | 5 mm² | 区分碎面与主体面 |
| 半径变异系数阈值 (模式 B) | 0.3 | 圆柱径向一致性 |

## 依赖

- FreeCAD 1.0+（内置 OCCT 7.8、numpy）
- 无需额外安装

## 许可证

MIT License
