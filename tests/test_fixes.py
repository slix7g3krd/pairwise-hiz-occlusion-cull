"""六个修复点的行为测试 + 周边函数的单元测试。"""

import threading

import pytest

import graphics.hiz_occlusion_cull as m
from graphics.hiz_occlusion_cull import (
    Box,
    Camera,
    CullMetrics,
    DepthPyramid,
    Fence,
    FenceNotReadyError,
    HiZCuller,
    QueryBatchDispatcher,
    StalePyramidError,
)


# ---------------------------------------------------------------------------
# 修复点 1：金字塔跟着本帧深度重建
# ---------------------------------------------------------------------------


def test_pyramid_rebuild_uses_current_frame_depth():
    depth_f0 = [[0.2, 0.4], [0.6, 0.8]]
    depth_f1 = [[0.9, 0.9], [0.9, 0.9]]
    p0 = DepthPyramid.rebuild(depth_f0, frame_id=0)
    p1 = DepthPyramid.rebuild(depth_f1, frame_id=1)

    # 新帧的 level0 必须来自新深度，而不是复用旧金字塔。
    assert p0.levels[0] == [[0.2, 0.4], [0.6, 0.8]]
    assert p1.levels[0] == [[0.9, 0.9], [0.9, 0.9]]
    assert p1.frame_id == 1

    # 反向深度归约取 min（更远处的保守深度）。
    assert p0.levels[1] == [[0.2]]
    # 正向深度归约取 max。
    pf = DepthPyramid.rebuild(depth_f0, frame_id=0, reversed_z=False)
    assert pf.levels[1] == [[0.8]]


def test_stale_pyramid_rejected_on_new_frame():
    pyramid = DepthPyramid.rebuild([[0.5, 0.5], [0.5, 0.5]], frame_id=7)
    # 同帧同视口可用。
    pyramid.require_fresh(7, 0)
    # 相机推进后仍拿上一帧金字塔 -> 拒绝。
    with pytest.raises(StalePyramidError):
        pyramid.require_fresh(8, 0)
    # 视口不匹配同样拒绝。
    with pytest.raises(StalePyramidError):
        pyramid.require_fresh(7, 2)


def test_frame_binding_rebinds_each_frame():
    b0 = m.frame_binding(0, [[0.1, 0.2], [0.3, 0.4]], viewport_id=0)
    b1 = m.frame_binding(1, [[0.8, 0.8], [0.8, 0.8]], viewport_id=0)
    assert b0["pyramid"].frame_id == 0
    assert b1["pyramid"].frame_id == 1
    assert b1["pyramid"].levels[0][0][0] == 0.8


# ---------------------------------------------------------------------------
# 修复点 2：查询等异步任务完成再读
# ---------------------------------------------------------------------------


def test_async_rebuild_blocks_read_until_fence_signaled():
    gate = threading.Event()
    handle = m.async_rebuild([[0.3, 0.3], [0.3, 0.3]], frame_id=0,
                             gate=gate)
    # 闸门未放开 -> compute 没写完，直接 result() 必须被拒。
    assert not handle.done
    with pytest.raises(FenceNotReadyError):
        handle.result()
    gate.set()
    # buffer_lifetime 负责等待 fence，之后读取成功。
    pyramid = m.buffer_lifetime(handle)
    assert isinstance(pyramid, DepthPyramid)
    assert handle.done
    assert handle.result().levels[0][0][0] == 0.3


def test_buffer_lifetime_runs_read_after_wait():
    handle = m.async_rebuild([[0.6]], frame_id=3)
    called = {}

    def read(pyramid):
        called["frame"] = pyramid.frame_id
        return pyramid.frame_id

    frame = m.buffer_lifetime(handle, read)
    assert frame == 3 and called == {"frame": 3}


# ---------------------------------------------------------------------------
# 修复点 3：反向深度区间的包围盒测试
# ---------------------------------------------------------------------------


def test_conservative_test_reversed_z_negative_depth():
    # 反向深度：NDC 越大越近。区间 (0.9, 0.2) 中 0.9 是最近点。
    # 盒体最近点 0.9 <= Hi-Z 0.95 -> 整盒在遮挡体之后 -> 被挡。
    assert m.conservative_test(0.9, 0.2, 0.95, reversed_z=True) is True
    # 最近点 0.9 > Hi-Z 0.5 -> 有部分比遮挡体更近 -> 不可剔除。
    assert m.conservative_test(0.9, 0.2, 0.5, reversed_z=True) is False


def test_conservative_test_argument_order_robustness():
    # 入参顺序翻转（模拟反向深度下 min/max 写反的旧 bug）也应得到一致结论。
    a = m.conservative_test(0.9, 0.2, 0.95, reversed_z=True)
    b = m.conservative_test(0.2, 0.9, 0.95, reversed_z=True)
    assert a is True and a == b
    c = m.conservative_test(0.9, 0.2, 0.5, reversed_z=True)
    d = m.conservative_test(0.2, 0.9, 0.5, reversed_z=True)
    assert c is False and c == d


def test_conservative_test_forward_z():
    # 正向深度：NDC 越大越远。盒体 (near=0.3, far=0.8)，最远点 0.8。
    # Hi-Z 保守深度 0.7：最远点 0.8 >= 0.7 -> 整盒在遮挡体之后 -> 被挡。
    assert m.conservative_test(0.8, 0.3, 0.7, reversed_z=False) is True
    # 最远点 0.6 < Hi-Z 0.7 -> 盒体比遮挡体更近 -> 可见。
    assert m.conservative_test(0.6, 0.3, 0.7, reversed_z=False) is False
    # 遮挡体更远（Hi-Z 0.9）时任何近于它的盒体都可见。
    assert m.conservative_test(0.8, 0.3, 0.9, reversed_z=False) is False


def test_conservative_test_sequence_form():
    assert m.conservative_test([0.9, 0.2, 0.95], reversed_z=True) is True
    assert m.conservative_test([0.9, 0.2, 0.5], reversed_z=True) is False


# ---------------------------------------------------------------------------
# 修复点 4：视口切换 / 相机移动时缓存失效
# ---------------------------------------------------------------------------


def test_camera_move_detection():
    cam_a = Camera(eye=(0, 0, 0))
    cam_b = Camera(eye=(0, 0, 5))  # 推进
    assert m.camera_move(cam_a, cam_b) is True
    assert m.camera_move(cam_a, Camera(eye=(0, 0, 0))) is False
    # 投影参数变化也算移动。
    assert m.camera_move(cam_a, Camera(eye=(0, 0, 0), near=1.0)) is True


def test_cull_cache_invalidated_on_camera_push():
    culler = HiZCuller(width=32, height=32)
    cam = Camera(eye=(0, 0, 0), reversed_z=True)
    # 全 0.97 的背景深度：遮挡墙 NDC 0.97，比盒体最近点 0.951 更近。
    depth = [[0.97] * 32 for _ in range(32)]
    box = Box([-0.5, -0.5, -6], [0.5, 0.5, -5])

    culler.begin_frame(0, cam, depth)
    v0 = culler.cull({1: box}, frame_id=0)[1]
    assert v0 is False  # 被墙挡住
    assert 1 in culler._viewports[0]["cache"]

    # 相机推进越过遮挡墙，缓存必须失效并按新深度重判。
    cam2 = Camera(eye=(0, 0, 8), reversed_z=True)
    depth2 = [[0.05] * 32 for _ in range(32)]  # 近处已无遮挡
    culler.begin_frame(1, cam2, depth2)
    assert culler._viewports[0]["cache"] == {}
    v1 = culler.cull({1: box}, frame_id=1)[1]
    assert v1 is True


def test_viewport_switch_invalidates_cache():
    culler = HiZCuller()
    culler.begin_frame(0, Camera(eye=(0, 0, 0)),
                       [[0.5] * 8 for _ in range(8)], viewport_id=1)
    culler._state(1)["cache"][99] = True
    out = culler.switch_viewport(1)
    assert out["cache_invalidated"] is True
    assert culler._viewports[1]["cache"] == {}


# ---------------------------------------------------------------------------
# 修复点 5：按视口拆分查询批次
# ---------------------------------------------------------------------------


def test_query_batch_split_by_viewport():
    batches = m.query_batch({0: [1, 2], 1: [3, 4, 5]}, frame_id=0)
    assert len(batches) == 2
    by_vp = {b.viewport_id: b for b in batches}
    assert by_vp[0].object_ids == [1, 2]
    assert by_vp[1].object_ids == [3, 4, 5]
    # 批次互不共享存储。
    assert by_vp[0].results is not by_vp[1].results


def test_dispatcher_rejects_cross_viewport_publish():
    d = QueryBatchDispatcher()
    b0 = d.submit(viewport_id=0, frame_id=0, object_ids=[1, 2])
    d.submit(viewport_id=1, frame_id=0, object_ids=[3])
    with pytest.raises(ValueError):
        d.publish(b0, {3: False})  # 视口 1 的对象不能写回视口 0 的批次
    d.publish(b0, {1: True, 2: False})
    assert b0.results == {1: True, 2: False}


def test_culler_batches_carry_viewport_ownership():
    culler = HiZCuller()
    for vp in (0, 1):
        culler.begin_frame(0, Camera(eye=(0, 0, 0)),
                           [[0.2] * 8 for _ in range(8)], viewport_id=vp)
        culler.cull({10 + vp: Box([-1, -1, -4], [1, 1, -3])},
                    frame_id=0, viewport_id=vp)
    batches = culler.dispatcher.all_batches()
    assert {b.viewport_id for b in batches} == {0, 1}
    assert [b.object_ids for b in
            sorted(batches, key=lambda b: b.viewport_id)] == [[10], [11]]


# ---------------------------------------------------------------------------
# 修复点 6：遮挡体合并保留包围裕量
# ---------------------------------------------------------------------------


def test_occluder_merge_keeps_margin():
    a = Box([0, 0, 0], [1, 1, 1])
    b = Box([2, 2, 2], [3, 3, 3])
    merged = m.occluder_merge([a, b], margin=0.25)
    assert merged.mins == [-0.25, -0.25, -0.25]
    assert merged.maxs == [3.25, 3.25, 3.25]


def test_occluder_merge_always_conservative():
    a = Box([0.5, -2.0, 4.0], [1.0, 0.0, 9.0])
    merged = m.occluder_merge([a], margin=0.0)  # 零裕量也强制最小正裕量
    for k in range(3):
        assert merged.mins[k] < a.mins[k]
        assert merged.maxs[k] > a.maxs[k]


# ---------------------------------------------------------------------------
# 端到端：六个修复点协同
# ---------------------------------------------------------------------------


def test_end_to_end_async_multi_viewport_pipeline():
    culler = HiZCuller(width=16, height=16)
    cam = Camera(eye=(0, 0, 0), reversed_z=True)
    # 遮挡墙 NDC 0.97，近于 wall_box 的最近点 0.951。
    depth = [[0.97] * 16 for _ in range(16)]
    wall_box = Box([-0.5, -0.5, -6], [0.5, 0.5, -5])
    near_box = Box([-0.3, -0.3, -2], [0.3, 0.3, -1.5])

    # 异步重建路径：fence 未完成前 cull 内部必须先等待。
    culler.begin_frame(0, cam, depth, viewport_id=0, async_mode=True)
    results = culler.cull({"a": wall_box, "b": near_box},
                          frame_id=0, viewport_id=0)
    assert results == {"a": False, "b": True}
    snap = m.metrics(culler.cull_metrics)
    assert snap["submitted"] == 2 and snap["occlusion_culled"] == 1


def test_fallback_degrades_to_all_visible():
    def boom():
        raise FenceNotReadyError("not ready")

    out = m.fallback(boom, all_ids=[1, 2, 3])
    assert out["degraded"] is True and out["visible"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# 周边函数
# ---------------------------------------------------------------------------


def test_validation_and_audit():
    pyramid = DepthPyramid.rebuild([[0.1, 0.2], [0.3, 0.4]], frame_id=0)
    report = m.validation(pyramid)
    assert report["ok"] is True and report["levels"] == 2

    culler = HiZCuller()
    culler.begin_frame(0, Camera(eye=(0, 0, 0)),
                       [[0.5] * 8 for _ in range(8)])
    assert m.audit(culler)["ok"] is True


def test_frustum_culls_box_behind_camera():
    cam = Camera(eye=(0, 0, 0), near=0.1, far=100.0)
    behind = Box([-1, -1, 5], [1, 1, 6])  # 相机后方（+Z）
    front = Box([-1, -1, -10], [1, 1, -5])
    assert m.frustum(cam, behind) is False
    assert m.frustum(cam, front) is True
    assert len(m.frustum(cam)) == 6


def test_lod_bias_and_determinism():
    assert m.lod_bias(100.0, distance=1.0) < m.lod_bias(2.0, distance=50.0)
    data = [3, 1, 2, 5, 4]
    assert m.determinism(data) == [1, 2, 3, 4, 5]
    objs = [{"id": 2}, {"id": 1}]
    assert m.determinism(objs, key=lambda o: o["id"])[0]["id"] == 1


def test_metrics_snapshot():
    mt = CullMetrics()
    mt.record(total=10, frustum_culled=2, occlusion_culled=3)
    snap = m.metrics(mt)
    assert snap["visible"] == 5
    assert snap["cull_ratio"] == pytest.approx(0.5)
