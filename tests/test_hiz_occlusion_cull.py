import pytest
import graphics.hiz_occlusion_cull as mod

CASES = [
    ("depth_pyramid", (1,)),
    ("async_rebuild", (1,)),
    ("conservative_test", (1,)),
    ("frame_binding", (1,)),
    ("camera_move", (1,)),
    ("buffer_lifetime", (1,)),
    ("occluder_merge", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("lod_bias", (1,)),
    ("metrics", (1,)),
    ("frustum", (1,)),
    ("query_batch", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
