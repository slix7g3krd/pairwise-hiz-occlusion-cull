# pairwise-hiz-occlusion-cull

## 领域

游戏、图形与多媒体

## 主题

Hi-Z 遮挡剔除

## 说明

金字塔重建、异步回读与保守判定。

初始状态下 `graphics/hiz_occlusion_cull.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
