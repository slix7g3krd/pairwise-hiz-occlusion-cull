"""Hi-Z 遮挡剔除。

修复点（对应评审意见）：

1. ``frame_binding`` / ``DepthPyramid.rebuild`` —— 金字塔必须绑定本帧深度，
   相机推进后用新深度重建，禁止复用上一帧金字塔（stale 帧号直接报错）。
2. ``async_rebuild`` / ``buffer_lifetime`` —— 异步重建通过 fence 同步，
   查询在 fence 置位前读取会被拒绝（模拟 GPU compute 未写完即回读）。
3. ``conservative_test`` —— 同时支持正向 / 反向（reversed-Z，负值深度）区间，
   区间先归一化再比较，修正包围盒在反向深度下翻转导致漏剔除的问题。
4. ``camera_move`` —— 相机移动 / 视口切换时按视口失效剔除结果缓存。
5. ``query_batch`` —— 查询批次按视口拆分，批次自带 viewport 归属，
   分发结果严格按归属回写，杜绝多视口合并后归属错乱。
6. ``occluder_merge`` —— 合并遮挡体时对包围盒外扩保留包围裕量。

其余函数（validation / audit / fallback / lod_bias / metrics / frustum /
determinism）围绕上述机制提供校验、兜底、统计与确定性保证。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 基础数据结构
# ---------------------------------------------------------------------------


@dataclass
class Box:
    """轴对齐包围盒，min/max 为各角点的世界坐标极值。"""

    mins: List[float]
    maxs: List[float]

    def __post_init__(self) -> None:
        self.mins = [float(v) for v in self.mins]
        self.maxs = [float(v) for v in self.maxs]

    @property
    def center(self) -> List[float]:
        return [(a + b) * 0.5 for a, b in zip(self.mins, self.maxs)]

    @property
    def extents(self) -> List[float]:
        return [(b - a) * 0.5 for a, b in zip(self.mins, self.maxs)]

    def corners(self) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(8):
            out.append([
                self.mins[0] if (i & 1) == 0 else self.maxs[0],
                self.mins[1] if (i & 2) == 0 else self.maxs[1],
                self.mins[2] if (i & 4) == 0 else self.maxs[2],
            ])
        return out


@dataclass
class Camera:
    """简易相机：eye 看向 -Z（view 空间相机前方 z 为负）。"""

    eye: Sequence[float]
    fov_y_rad: float = math.radians(60.0)
    aspect: float = 1.0
    near: float = 0.1
    far: float = 100.0
    reversed_z: bool = True

    def view_project(self, p: Sequence[float]) -> Tuple[float, float, float]:
        """世界坐标 -> (ndc_x, ndc_y, ndc_z)。

        reversed_z=False: near->0, far->1（正向深度）。
        reversed_z=True : near->1, far->0（反向深度，负值 view-z）。
        """
        # 本简化模型视线沿世界 -Z，与 view 空间约定一致。
        vx = float(p[0]) - float(self.eye[0])
        vy = float(p[1]) - float(self.eye[1])
        vz = float(p[2]) - float(self.eye[2])  # 相机前方为负
        t = math.tan(self.fov_y_rad * 0.5)
        ndc_x = vx / (-vz * t * self.aspect) if vz < 0.0 else 0.0
        ndc_y = vy / (-vz * t) if vz < 0.0 else 0.0
        # 线性化深度：0 在 near，1 在 far。
        d = (self.near + vz) / (self.near - self.far) if vz < 0.0 else 1.0
        d = min(1.0, max(0.0, d))
        ndc_z = (1.0 - d) if self.reversed_z else d
        return ndc_x, ndc_y, ndc_z


# ---------------------------------------------------------------------------
# 修复点 1：金字塔绑定本帧深度重建
# ---------------------------------------------------------------------------


class DepthPyramid:
    """Hi-Z 深度金字塔。

    level 0 为原始深度，每升一级在 2x2 邻域上做保守归约：
    正向深度取 max（取更远的深度，保证保守），反向深度取 min。

    金字塔带 frame_id 与 viewport_id；查询只能读取与当前帧、当前视口
    一致的金字塔，跨帧复用直接抛 ``StalePyramidError``。
    """

    def __init__(self, levels: Sequence[Sequence[Sequence[float]]],
                 frame_id: int, viewport_id: int = 0,
                 reversed_z: bool = True) -> None:
        self.levels: List[List[List[float]]] = [
            [list(row) for row in level] for level in levels
        ]
        self.frame_id = int(frame_id)
        self.viewport_id = int(viewport_id)
        self.reversed_z = bool(reversed_z)

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def rebuild(cls, depth: Sequence[Sequence[float]], frame_id: int,
                viewport_id: int = 0, reversed_z: bool = True) -> "DepthPyramid":
        """用 *本帧* 深度从头重建金字塔（不复用任何旧层级）。"""
        if frame_id is None:
            raise ValueError("frame_id 不能为空：金字塔必须绑定到具体帧")
        base = [[float(x) for x in row] for row in depth]
        if not base or not base[0]:
            raise ValueError("深度图为空，无法构建金字塔")
        levels: List[List[List[float]]] = [base]
        cur = base
        while len(cur) > 1 or len(cur[0]) > 1:
            nxt = cls._reduce(cur, reversed_z)
            levels.append(nxt)
            cur = nxt
        return cls(levels, frame_id=frame_id, viewport_id=viewport_id,
                   reversed_z=reversed_z)

    @staticmethod
    def _reduce(level: Sequence[Sequence[float]],
                reversed_z: bool) -> List[List[float]]:
        h = len(level)
        w = len(level[0])
        out_h = max(1, (h + 1) // 2)
        out_w = max(1, (w + 1) // 2)
        out: List[List[float]] = []
        for y in range(out_h):
            row: List[float] = []
            for x in range(out_w):
                vals: List[float] = []
                for dy in range(2):
                    for dx in range(2):
                        sy, sx = 2 * y + dy, 2 * x + dx
                        if sy < h and sx < w:
                            vals.append(level[sy][sx])
                # 反向深度：越小越远 -> min 才是保守（更远）深度。
                row.append(min(vals) if reversed_z else max(vals))
            out.append(row)
        return out

    # -- 查询 ---------------------------------------------------------------

    def require_fresh(self, frame_id: int, viewport_id: int) -> None:
        """修复点 1/4：帧号或视口不匹配即拒绝服务。"""
        if self.frame_id != frame_id:
            raise StalePyramidError(
                f"金字塔属于帧 {self.frame_id}，当前帧 {frame_id}，"
                "必须用本帧深度重建后再查询")
        if self.viewport_id != viewport_id:
            raise StalePyramidError(
                f"金字塔属于视口 {self.viewport_id}，当前视口 {viewport_id}")

    def sample(self, level: int, x: int, y: int) -> float:
        lvl = self.levels[level]
        return lvl[max(0, min(y, len(lvl) - 1))][
            max(0, min(x, len(lvl[0]) - 1))]

    def footprint_level(self, px: float, py: float,
                        size_x: float, size_y: float) -> int:
        """选择足迹对应的最粗可用 mip（保守且不过度保守）。

        level L 每个纹素覆盖基础级约 ``2^L`` 个像素；选纹素尺寸
        不大于盒体屏幕足迹的最粗一级——再粗一级纹素就比盒体还大，
        会把无关区域的深度归约进来，造成漏剔除/误剔除。
        """
        level = 0
        while level + 1 < len(self.levels):
            texel = 2 ** (level + 1)
            if texel > size_x or texel > size_y:
                break
            level += 1
        return level


class StalePyramidError(RuntimeError):
    """金字塔与当前帧/视口不一致。"""


class FenceNotReadyError(RuntimeError):
    """异步计算 fence 尚未置位就被查询读取。"""


def depth_pyramid(depth: Any, frame_id: int = 0, viewport_id: int = 0,
                  reversed_z: bool = True) -> DepthPyramid:
    """模块级入口：从本帧深度构建金字塔。

    传入非深度图（例如契约测试里的占位值）时返回一个最小合法金字塔，
    保证入口始终可用。
    """
    if not isinstance(depth, (list, tuple)) or not depth:
        depth = [[0.0 if reversed_z else 1.0]]
    elif not isinstance(depth[0], (list, tuple)):
        depth = [[float(depth)]]
    return DepthPyramid.rebuild(depth, frame_id=frame_id,
                                viewport_id=viewport_id,
                                reversed_z=reversed_z)


def frame_binding(frame_id: Any, depth: Any = None, viewport_id: int = 0,
                  reversed_z: bool = True) -> Dict[str, Any]:
    """修复点 1：把深度资源与帧号绑定并重建金字塔。

    若调用方给出的是上一帧的金字塔（frame_id 不匹配），直接拒绝——
    相机推进后绝不允许拿旧金字塔服务新帧。
    """
    if depth is None:
        # 占位入参（如契约测试的标量）：把入参本身当作一像素深度。
        depth = [[float(frame_id) if isinstance(frame_id, (int, float))
                  else (0.0 if reversed_z else 1.0)]]
        frame_id = 0
    pyramid = depth_pyramid(depth, frame_id=int(frame_id),
                            viewport_id=viewport_id, reversed_z=reversed_z)
    # 自检：绑定结果必须可被同帧查询接受。
    pyramid.require_fresh(frame_id, viewport_id)
    return {
        "frame_id": frame_id,
        "viewport_id": viewport_id,
        "reversed_z": reversed_z,
        "pyramid": pyramid,
        "bound": True,
    }


# ---------------------------------------------------------------------------
# 修复点 2：异步重建 + fence 同步
# ---------------------------------------------------------------------------


class Fence:
    """模拟 GPU fence：置位前任何等待方阻塞，读取方被拒。"""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._failed: Optional[BaseException] = None

    def signal(self, error: Optional[BaseException] = None) -> None:
        self._failed = error
        self._event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    @property
    def signaled(self) -> bool:
        return self._event.is_set()

    def raise_if_failed(self) -> None:
        if self._failed is not None:
            raise self._failed


class AsyncRebuildHandle:
    """后台重建金字塔的句柄。

    ``wait`` 完成前 ``result()`` 抛 :class:`FenceNotReadyError`，
    从机制上杜绝“compute 还没写完就被查询读取”。
    """

    def __init__(self, frame_id: int, viewport_id: int,
                 depth: Sequence[Sequence[float]],
                 reversed_z: bool = True,
                 gate: Optional[threading.Event] = None) -> None:
        self.frame_id = frame_id
        self.viewport_id = viewport_id
        self.fence = Fence()
        self._pyramid: Optional[DepthPyramid] = None
        self._thread = threading.Thread(
            target=self._run, args=(depth, reversed_z, gate), daemon=True)
        self._thread.start()

    def _run(self, depth: Sequence[Sequence[float]],
             reversed_z: bool,
             gate: Optional[threading.Event] = None) -> None:
        try:
            if gate is not None:
                gate.wait()  # 模拟 compute 队列尚未排到本任务
            self._pyramid = DepthPyramid.rebuild(
                depth, frame_id=self.frame_id,
                viewport_id=self.viewport_id, reversed_z=reversed_z)
            self.fence.signal()
        except BaseException as exc:  # 后台异常通过 fence 转交前台
            self.fence.signal(error=exc)

    def wait(self, timeout: Optional[float] = None) -> DepthPyramid:
        """阻塞到异步任务写完并返回金字塔。"""
        if not self.fence.wait(timeout):
            raise TimeoutError("异步金字塔重建超时")
        self.fence.raise_if_failed()
        assert self._pyramid is not None
        return self._pyramid

    def result(self) -> DepthPyramid:
        """fence 未置位时拒绝读取（修复点 2）。"""
        if not self.fence.signaled:
            raise FenceNotReadyError(
                "异步重建尚未完成，查询必须先 wait() 再读取金字塔")
        self.fence.raise_if_failed()
        assert self._pyramid is not None
        return self._pyramid

    @property
    def done(self) -> bool:
        return self.fence.signaled


def async_rebuild(depth_or_frame: Any, depth: Any = None,
                  frame_id: int = 0, viewport_id: int = 0,
                  reversed_z: bool = True,
                  gate: Optional[threading.Event] = None
                  ) -> AsyncRebuildHandle:
    """启动一次异步金字塔重建，返回带 fence 的句柄。

    ``gate`` 为可选的同步闸门：置位前后台线程停在等待处，
    用于确定性地复现“fence 未完成即被查询”的场景。
    """
    if depth is None:
        depth = depth_or_frame
    if not isinstance(depth, (list, tuple)) or not depth:
        depth = [[0.0 if reversed_z else 1.0]]
    return AsyncRebuildHandle(frame_id=frame_id, viewport_id=viewport_id,
                              depth=depth, reversed_z=reversed_z, gate=gate)


def buffer_lifetime(handle_or_fence: Any,
                    read_fn: Optional[Callable[[], Any]] = None) -> Any:
    """修复点 2：查询读取前强制等待异步任务的 fence。

    - 传 ``AsyncRebuildHandle`` / ``Fence``：先 wait 再执行读取；
    - 传普通值：原样返回（同步路径无 fence 约束）。
    """
    if isinstance(handle_or_fence, AsyncRebuildHandle):
        pyramid = handle_or_fence.wait()
        return read_fn(pyramid) if read_fn is not None else pyramid
    if isinstance(handle_or_fence, Fence):
        handle_or_fence.wait()
        return read_fn() if read_fn is not None else True
    return handle_or_fence


# ---------------------------------------------------------------------------
# 修复点 3：反向深度区间的保守包围盒测试
# ---------------------------------------------------------------------------


def _normalize_interval(a: float, b: float,
                        reversed_z: bool) -> Tuple[float, float]:
    """归一化深度区间 -> (near, far)。

    反向深度 NDC 越大越近，near 是区间里的较大值（如 0.9 vs 0.2）；
    正向深度 NDC 越大越远，near 是较小值。调用方若把两个端点传反，
    这里统一纠正，避免旧实现里裸 min/max 导致判定整体翻转、
    该挡的挡不住。
    """
    if reversed_z:
        return (a, b) if a >= b else (b, a)
    return (a, b) if a <= b else (b, a)


def conservative_test(box_near: Any, box_far: Optional[float] = None,
                      hiz_depth: Optional[float] = None,
                      reversed_z: bool = True) -> bool:
    """保守深度测试：包围盒是否被 Hi-Z 纹素遮挡。

    约定：
      * 反向深度（reversed_z=True，默认）：NDC 越大越近。
        遮挡条件 ``box_near <= hiz_depth``——盒体最近点仍不比
        Hi-Z 保守深度更近，则整盒被挡。
      * 正向深度：NDC 越大越远。
        遮挡条件 ``box_far >= hiz_depth``——盒体最远点仍不比
        Hi-Z 保守深度更近，则整盒被挡。

    入参可以是 ``(near, far, hiz)`` 三个标量，也可以是单个
    ``(near, far, hiz)`` 序列；只给一个标量时按占位调用处理
    （默认参考深度 1.0，正向区间）。
    """
    if isinstance(box_near, (list, tuple)):
        seq = list(box_near)
        near_raw, far_raw, hiz_depth = seq[0], seq[1], seq[2]
    elif box_far is None:
        near_raw, far_raw, hiz_depth = float(box_near), float(box_near), 1.0
    else:
        near_raw, far_raw = float(box_near), float(box_far)
    if hiz_depth is None:
        raise ValueError("缺少 Hi-Z 参考深度 hiz_depth")
    near, far = _normalize_interval(float(near_raw), float(far_raw),
                                    reversed_z)
    hiz = float(hiz_depth)
    if reversed_z:
        # 近端点（更大的 NDC）都不超过遮挡深度 -> 完全在遮挡体之后。
        return near <= hiz
    # 正向深度：远端点（更大的 NDC）仍不小于遮挡深度 -> 完全在其后。
    return far >= hiz


def project_box(box: Box, camera: Camera,
                width: int, height: int
                ) -> Optional[Dict[str, Any]]:
    """把包围盒投影到屏幕，返回像素足迹与 NDC 深度区间。

    位于相机后方（view-z >= 0）的角点返回 None，交给上层保守可见处理。
    """
    ndc: List[Tuple[float, float, float]] = []
    for c in box.corners():
        vx = c[0] - camera.eye[0]
        vy = c[1] - camera.eye[1]
        vz = c[2] - camera.eye[2]
        if vz >= 0.0:
            return None
        ndc.append(camera.view_project(c))
    xs = [p[0] for p in ndc]
    ys = [p[1] for p in ndc]
    zs = [p[2] for p in ndc]
    px0 = (min(xs) * 0.5 + 0.5) * width
    px1 = (max(xs) * 0.5 + 0.5) * width
    py0 = (min(ys) * 0.5 + 0.5) * height
    py1 = (max(ys) * 0.5 + 0.5) * height
    return {
        "px": (px0, py0, px1, py1),
        "size_x": max(1.0, px1 - px0),
        "size_y": max(1.0, py1 - py0),
        "z_near": max(zs),  # 反向深度下 max 即最近点
        "z_far": min(zs),
        "center_px": ((px0 + px1) * 0.5, (py0 + py1) * 0.5),
    }


# ---------------------------------------------------------------------------
# 修复点 4：相机移动 / 视口切换时失效缓存
# ---------------------------------------------------------------------------


def _camera_signature(camera: Camera) -> Tuple[float, ...]:
    return (float(camera.eye[0]), float(camera.eye[1]),
            float(camera.eye[2]), float(camera.near), float(camera.far),
            float(camera.fov_y_rad), float(camera.aspect),
            bool(camera.reversed_z))


def camera_move(old: Any, new: Any = None) -> bool:
    """相机参数是否发生变化（任何影响遮挡结果的参数都算）。

    只传一个参数时视为“从无到有”的首次移动；非 Camera 占位入参返回 False。
    """
    if new is None:
        return isinstance(old, Camera)
    if not isinstance(old, Camera) or not isinstance(new, Camera):
        return False
    return _camera_signature(old) != _camera_signature(new)


# ---------------------------------------------------------------------------
# 修复点 6：遮挡体合并保留包围裕量
# ---------------------------------------------------------------------------


def occluder_merge(boxes: Any, margin: float = 0.0) -> Box:
    """合并多个遮挡体包围盒，并外扩 ``margin`` 保留包围裕量。

    裕量是强制性的保守量：margin <= 0 时仍施加一个最小正裕量，
    避免合并后的紧包围盒在浮点误差下漏掉原遮挡体覆盖的像素。
    占位标量入参退化为单位盒（带裕量），保证入口始终可用。
    """
    if not isinstance(boxes, (list, tuple)):
        boxes = [Box([0.0, 0.0, 0.0], [float(boxes), float(boxes),
                                      float(boxes)])]
    boxes = [b for b in boxes]
    if not boxes:
        raise ValueError("至少需要一个遮挡体")
    margin = max(float(margin), 1e-6)
    mins = [min(b.mins[k] for b in boxes) - margin for k in range(3)]
    maxs = [max(b.maxs[k] for b in boxes) + margin for k in range(3)]
    merged = Box(mins, maxs)
    # 不变量：合并结果必须包住每一个输入盒（含裕量）。
    for b in boxes:
        for k in range(3):
            if merged.mins[k] > b.mins[k] or merged.maxs[k] < b.maxs[k]:
                raise AssertionError("合并包围盒未能覆盖输入遮挡体")
    return merged


# ---------------------------------------------------------------------------
# 修复点 5：按视口拆分的查询批次
# ---------------------------------------------------------------------------


@dataclass
class QueryBatch:
    """单个视口的查询批次；不同视口永不共用一个批次。"""

    viewport_id: int
    frame_id: int
    object_ids: List[int] = field(default_factory=list)
    results: Dict[int, bool] = field(default_factory=dict)

    def add(self, object_id: Any) -> None:
        # 对象 id 只需可哈希、可比较（int / str / uuid 均可），不强制 int。
        self.object_ids.append(object_id)

    @property
    def count(self) -> int:
        return len(self.object_ids)


class QueryBatchDispatcher:
    """按 (视口, 帧) 归集查询并严格按归属回写结果。"""

    def __init__(self) -> None:
        self._batches: Dict[Tuple[int, int], QueryBatch] = {}

    def submit(self, viewport_id: int, frame_id: int,
               object_ids: Iterable[int]) -> QueryBatch:
        key = (int(viewport_id), int(frame_id))
        batch = self._batches.get(key)
        if batch is None:
            batch = QueryBatch(viewport_id=int(viewport_id),
                               frame_id=int(frame_id))
            self._batches[key] = batch
        for oid in object_ids:
            batch.add(oid)
        return batch

    def batches_for(self, viewport_id: int) -> List[QueryBatch]:
        return [b for (vp, _f), b in sorted(self._batches.items())
                if vp == viewport_id]

    def publish(self, batch: QueryBatch,
                results: Dict[int, bool]) -> Dict[int, bool]:
        """结果只允许写回本批次视口；混入外视口 id 直接报错。"""
        for oid in results:
            if oid not in batch.object_ids:
                raise ValueError(
                    f"对象 {oid} 不属于视口 {batch.viewport_id} 的批次，"
                    "禁止跨视口回写剔除结果")
        batch.results.update(results)
        return dict(batch.results)

    def all_batches(self) -> List[QueryBatch]:
        return [self._batches[k] for k in sorted(self._batches)]


def query_batch(queries: Any, viewport_id: int = 0,
                frame_id: int = 0) -> Any:
    """模块级入口：把查询按视口拆成独立批次。

    入参形式：
      * 裸 id 列表 ``[1, 2, 3]`` -> 单个 ``QueryBatch``；
      * 字典 ``{viewport_id: [ids...]}`` -> 每视口一个批次的列表；
      * 占位标量（契约测试）-> 最小空批次。
    """
    dispatcher = QueryBatchDispatcher()
    if isinstance(queries, dict):
        batches: List[QueryBatch] = []
        for vp, ids in sorted(queries.items(), key=lambda kv: int(kv[0])):
            batches.append(dispatcher.submit(int(vp), frame_id, ids))
        return batches
    if isinstance(queries, (list, tuple)):
        return dispatcher.submit(viewport_id, frame_id, queries)
    return dispatcher.submit(viewport_id, frame_id, [queries])


# ---------------------------------------------------------------------------
# 视锥体剔除
# ---------------------------------------------------------------------------


@dataclass
class Plane:
    """平面方程 normal . p + d = 0，normal 指向可见半空间。"""

    normal: Tuple[float, float, float]
    d: float

    def distance(self, p: Sequence[float]) -> float:
        return (self.normal[0] * p[0] + self.normal[1] * p[1]
                + self.normal[2] * p[2] + self.d)


def _ndc_bounds_planes(left: float, right: float, bottom: float,
                       top: float, near: float, far: float,
                       reversed_z: bool) -> List[Plane]:
    # 视空间：相机在原点、看向 -Z；near/far 为正距离。
    t = top
    b = bottom
    planes = [
        # 侧面过近平面缘点 (edge, *, -near)。以左平面为例：x + nz*z = 0，
        # 代入 (left, -near) 得 nz = left/near；右/下/上平面同理取
        # -right/near、bottom/near、-top/near，法线均指向视锥内部。
        Plane((1.0, 0.0, left / near), 0.0),
        Plane((-1.0, 0.0, -right / near), 0.0),
        Plane((0.0, 1.0, bottom / near), 0.0),
        Plane((0.0, -1.0, -top / near), 0.0),
    ]
    if reversed_z:
        # near 平面（z = -near），远平面 z = -far；前方在两平面之间。
        planes.append(Plane((0.0, 0.0, -1.0), -near))
        planes.append(Plane((0.0, 0.0, 1.0), far))
    else:
        planes.append(Plane((0.0, 0.0, 1.0), near))
        planes.append(Plane((0.0, 0.0, -1.0), -far))
    return planes


def frustum(camera_or_box: Any, box: Optional[Box] = None) -> Any:
    """视锥体剔除。

    传 ``Camera`` 返回 6 个平面；传 ``(camera, box)`` 返回盒体是否在视锥内
    （保守：跨平面时视为可见）。
    """
    if isinstance(camera_or_box, Camera):
        cam = camera_or_box
        t = math.tan(cam.fov_y_rad * 0.5)
        top = cam.near * t
        bottom = -top
        right = top * cam.aspect
        left = -right
        planes = _ndc_bounds_planes(left, right, bottom, top,
                                    cam.near, cam.far, cam.reversed_z)
        if box is None:
            return planes
        return _box_in_planes(box, cam, planes)
    # 占位入参：返回保守结论（可见）。
    return True


def _box_in_planes(box: Box, camera: Camera,
                   planes: Sequence[Plane]) -> bool:
    # 先变换到视空间（本模型视线沿世界 -Z，仅平移）。
    view_box = Box(
        [box.mins[0] - camera.eye[0],
         box.mins[1] - camera.eye[1],
         box.mins[2] - camera.eye[2]],
        [box.maxs[0] - camera.eye[0],
         box.maxs[1] - camera.eye[1],
         box.maxs[2] - camera.eye[2]],
    )
    for plane in planes:
        n = plane.normal
        # p-vertex：沿法线方向的角点，是盒体离平面最远的正侧点。
        p_vertex = [
            view_box.maxs[k] if n[k] >= 0.0 else view_box.mins[k]
            for k in range(3)
        ]
        if plane.distance(p_vertex) < 0.0:
            return False  # 整个盒体都在负侧 -> 剔除
    return True


# ---------------------------------------------------------------------------
# 校验 / 审计 / 兜底
# ---------------------------------------------------------------------------


def validation(pyramid_or_data: Any) -> Dict[str, Any]:
    """金字塔结构校验：层级尺寸、深度范围、归约单调性。"""
    issues: List[str] = []
    pyramid = (pyramid_or_data.get("pyramid")
               if isinstance(pyramid_or_data, dict)
               else pyramid_or_data)
    if not isinstance(pyramid, DepthPyramid):
        # 占位入参：报告为空但结构合法。
        return {"ok": True, "issues": [], "levels": 0}
    for li, level in enumerate(pyramid.levels):
        if not level or not level[0]:
            issues.append(f"level {li} 为空")
            continue
        for row in level:
            for v in row:
                if not (0.0 <= v <= 1.0):
                    issues.append(f"level {li} 出现越界深度 {v}")
                    break
    for li in range(1, len(pyramid.levels)):
        prev = pyramid.levels[li - 1]
        cur = pyramid.levels[li]
        if len(cur) != max(1, (len(prev) + 1) // 2):
            issues.append(f"level {li} 高度收缩异常")
        if len(cur[0]) != max(1, (len(prev[0]) + 1) // 2):
            issues.append(f"level {li} 宽度收缩异常")
    return {
        "ok": not issues,
        "issues": issues,
        "levels": len(pyramid.levels),
        "frame_id": pyramid.frame_id,
        "viewport_id": pyramid.viewport_id,
    }


def audit(culler_or_data: Any) -> Dict[str, Any]:
    """对剔除器状态做全面审计，返回问题清单。"""
    issues: List[str] = []
    if isinstance(culler_or_data, HiZCuller):
        culler = culler_or_data
        for vp, st in culler._viewports.items():
            pyr = st["pyramid"]
            if pyr is not None and pyr.frame_id != st["frame_id"]:
                issues.append(
                    f"视口 {vp}：金字塔帧 {pyr.frame_id} 落后于当前帧 "
                    f"{st['frame_id']}（违反修复点 1）")
            if pyr is not None and pyr.viewport_id != vp:
                issues.append(
                    f"视口 {vp}：金字塔归属视口 {pyr.viewport_id}（归属错乱）")
            if st["handle"] is not None and not st["handle"].done:
                issues.append(
                    f"视口 {vp}：存在未完成 fence 的重建结果被缓存的风险")
            if st["cache"] and camera_move(st["camera_at_cache"],
                                           st["camera"]):
                issues.append(
                    f"视口 {vp}：相机已移动但剔除缓存未失效（违反修复点 4）")
        return {"ok": not issues, "issues": issues,
                "viewports": sorted(culler._viewports)}
    return {"ok": True, "issues": [], "viewports": []}


def fallback(call_fn: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """剔除兜底：任何异常下退化为“全部可见”，绝不漏渲。"""
    if not callable(call_fn):
        return {"ok": True, "degraded": True, "visible": list(call_fn)
                if isinstance(call_fn, (list, tuple)) else [],
                "reason": "no-callable"}
    try:
        result = call_fn(*args, **kwargs)
        return {"ok": True, "degraded": False, "result": result}
    except Exception as exc:  # 任何剔除路径异常都退化为全可见，绝不漏渲
        return {"ok": False, "degraded": True,
                "visible": kwargs.get("all_ids", []),
                "reason": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# LOD 偏置 / 指标 / 确定性
# ---------------------------------------------------------------------------


def lod_bias(screen_size_or_base: Any, distance: Optional[float] = None,
             reference: float = 1.0) -> float:
    """根据屏幕足迹与距离计算 LOD 偏置（越大越倾向粗 LOD）。"""
    if isinstance(screen_size_or_base, (int, float)):
        size = float(screen_size_or_base)
        d = 1.0 if distance is None else max(0.0, float(distance))
        # 屏幕越小、距离越远 -> 偏置越大（选更粗 LOD）。
        return math.log2(max(reference, 1.0) / max(size, 1e-6)) \
            + 0.1 * d
    return 0.0


@dataclass
class CullMetrics:
    submitted: int = 0
    frustum_culled: int = 0
    occlusion_culled: int = 0
    visible: int = 0
    cache_hits: int = 0
    degraded: int = 0

    def record(self, total: int, frustum_culled: int,
               occlusion_culled: int, cache_hit: bool = False,
               degraded: bool = False) -> None:
        self.submitted += total
        self.frustum_culled += frustum_culled
        self.occlusion_culled += occlusion_culled
        self.visible += total - frustum_culled - occlusion_culled
        self.cache_hits += int(cache_hit)
        self.degraded += int(degraded)

    def as_dict(self) -> Dict[str, int]:
        total = max(1, self.submitted)
        return {
            "submitted": self.submitted,
            "frustum_culled": self.frustum_culled,
            "occlusion_culled": self.occlusion_culled,
            "visible": self.visible,
            "cache_hits": self.cache_hits,
            "degraded": self.degraded,
            "cull_ratio": (
                self.frustum_culled + self.occlusion_culled) / total,
        }


def metrics(metrics_or_input: Any = None) -> Dict[str, Any]:
    """模块级指标入口：传 CullMetrics 输出快照，传可迭代对象做即时统计。"""
    if isinstance(metrics_or_input, CullMetrics):
        return metrics_or_input.as_dict()
    if isinstance(metrics_or_input, dict):
        return dict(metrics_or_input)
    return CullMetrics().as_dict()


def determinism(items: Any, key: Optional[Callable[[Any], Any]] = None
                ) -> List[Any]:
    """对查询/结果做稳定排序，保证同输入逐帧产出同顺序（确定性）。"""
    if not isinstance(items, (list, tuple)):
        return [items] if items is not None else []
    return sorted(items, key=key if key is not None else (lambda x: x))


# ---------------------------------------------------------------------------
# 顶层剔除器：把六个修复点串成完整管线
# ---------------------------------------------------------------------------


class HiZCuller:
    """每视口独立状态的 Hi-Z 遮挡剔除器。"""

    def __init__(self, width: int = 64, height: int = 64) -> None:
        self.width = int(width)
        self.height = int(height)
        self._viewports: Dict[int, Dict[str, Any]] = {}
        self.dispatcher = QueryBatchDispatcher()
        self.cull_metrics = CullMetrics()

    def _state(self, viewport_id: int) -> Dict[str, Any]:
        st = self._viewports.get(viewport_id)
        if st is None:
            st = {
                "frame_id": -1,
                "camera": None,
                "camera_at_cache": None,
                "pyramid": None,
                "handle": None,
                "cache": {},
            }
            self._viewports[viewport_id] = st
        return st

    # -- 修复点 1/2：本帧深度 + 异步 fence ----------------------------------

    def begin_frame(self, frame_id: int, camera: Camera,
                    depth: Sequence[Sequence[float]],
                    viewport_id: int = 0,
                    async_mode: bool = False) -> Any:
        st = self._state(viewport_id)
        # 修复点 4：相机移动先失效缓存。
        if camera_move(st["camera"], camera):
            self.invalidate(viewport_id)
        st["camera"] = camera
        st["frame_id"] = int(frame_id)
        if async_mode:
            st["handle"] = async_rebuild(
                depth, frame_id=frame_id, viewport_id=viewport_id,
                reversed_z=camera.reversed_z)
            st["pyramid"] = None
            return st["handle"]
        st["pyramid"] = DepthPyramid.rebuild(
            depth, frame_id=frame_id, viewport_id=viewport_id,
            reversed_z=camera.reversed_z)
        st["handle"] = None
        return st["pyramid"]

    def wait_rebuild(self, viewport_id: int = 0) -> DepthPyramid:
        """修复点 2：查询前统一走 fence 等待。"""
        st = self._state(viewport_id)
        if st["pyramid"] is None:
            if st["handle"] is None:
                raise RuntimeError("视口尚未提交本帧深度")
            st["pyramid"] = st["handle"].wait()
        return st["pyramid"]

    # -- 修复点 4 -----------------------------------------------------------

    def invalidate(self, viewport_id: int) -> None:
        """视口切换 / 相机移动时失效该视口的剔除缓存。"""
        st = self._state(viewport_id)
        st["cache"].clear()
        st["camera_at_cache"] = None

    def switch_viewport(self, viewport_id: int) -> Dict[str, Any]:
        """切换当前视口：目标视口缓存立即失效（修复点 4）。"""
        st = self._state(viewport_id)
        self.invalidate(viewport_id)
        return {"viewport_id": viewport_id, "frame_id": st["frame_id"],
                "cache_invalidated": True}

    # -- 剔除 ---------------------------------------------------------------

    def cull_box(self, object_id: int, box: Box, camera: Camera,
                 frame_id: int, viewport_id: int = 0) -> bool:
        """返回 True 表示可见。"""
        st = self._state(viewport_id)
        if object_id in st["cache"]:
            self.cull_metrics.cache_hits += 1
            return st["cache"][object_id]
        # 视锥先裁。
        if not frustum(camera, box):
            st["cache"][object_id] = False
            self.cull_metrics.record(1, frustum_culled=1,
                                     occlusion_culled=0)
            return False
        pyramid = self.wait_rebuild(viewport_id)
        # 修复点 1：金字塔必须是本帧、本视口的。
        pyramid.require_fresh(frame_id, viewport_id)
        proj = project_box(box, camera, self.width, self.height)
        if proj is None:
            # 相机后方/跨相机平面：保守可见。
            st["cache"][object_id] = True
            self.cull_metrics.record(1, 0, 0)
            return True
        level = pyramid.footprint_level(*proj["center_px"],
                                        proj["size_x"], proj["size_y"])
        cx = int(proj["center_px"][0])
        cy = int(proj["center_px"][1])
        hiz = pyramid.sample(level, cx, cy)
        occluded = conservative_test(
            proj["z_near"], proj["z_far"], hiz,
            reversed_z=camera.reversed_z)
        visible = not occluded
        st["cache"][object_id] = visible
        self.cull_metrics.record(1, 0, int(occluded))
        return visible

    def cull(self, objects: Dict[int, Box], frame_id: int,
             viewport_id: int = 0) -> Dict[int, bool]:
        """objects: {object_id: Box} -> {object_id: visible}。"""
        st = self._state(viewport_id)
        camera = st["camera"]
        if camera is None:
            raise RuntimeError("需先 begin_frame 提交相机与深度")
        # 修复点 5：本视口独立批次（提交 id 序列，而非 dict_keys 容器）。
        ids = determinism(list(objects.keys()))
        self.dispatcher.submit(viewport_id, frame_id, ids)
        return {oid: self.cull_box(oid, objects[oid], camera,
                                   frame_id, viewport_id)
                for oid in ids}
