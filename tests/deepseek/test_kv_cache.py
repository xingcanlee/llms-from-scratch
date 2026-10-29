import torch

from llms_from_scratch.deepseek.kv_cache import (
    E2M1_GRID,
    fp4_e4m3_roundtrip,
    kv_latent_bound,
    max_representable_without_global_scale,
    mxfp4_roundtrip,
    persistent_cache_bytes,
    round_to_e2m1,
    swa_replay_tokens,
)


def test_e2m1_grid_rounding():
    x = torch.tensor([0.2, 0.3, 1.2, 2.6, 5.1, 7.0, -3.4])
    assert round_to_e2m1(x).tolist() == [0.0, 0.5, 1.0, 3.0, 6.0, 6.0, -3.0]
    assert len(E2M1_GRID) == 8  # 3 位编码幅值 + 1 位符号


def test_fp4_formats_values_and_error():
    torch.manual_seed(0)
    x = torch.randn(64, 512)
    nv, mx = fp4_e4m3_roundtrip(x, 16), mxfp4_roundtrip(x, 32)
    for y, block in ((nv, 16), (mx, 32)):
        tiles = (y - x).unflatten(-1, (-1, block))
        amax = x.unflatten(-1, (-1, block)).abs().amax(-1, keepdim=True)
        assert (tiles.abs() <= amax / 3 + 1e-6).all()  # 最大格距 2（4→6），量化步长 ≤ amax/3
    # 更小的块 + 有尾数的缩放：主 KV 格式误差更小，这也是 V4.1 选它的原因之一
    assert (nv - x).norm() < (mx - x).norm()


def test_dynamic_range_argument_from_report():
    assert max_representable_without_global_scale() == 2688
    assert 22 < kv_latent_bound(512) < 23  # √512 ≈ 22.6，远小于 2688


def test_replay_and_persistent_cache_accounting():
    assert swa_replay_tokens(43, 128, bounded=False) == 43 * 128
    assert swa_replay_tokens(40, 128, bounded=True) == 128
    v4 = persistent_cache_bytes(4096, 3508.8, 583, 43, 128, swa_points=2)
    v41 = persistent_cache_bytes(4096, 890, 583, 40, 128, swa_points=0)
    assert v4 / v41 > 4  # 短会话中滑窗快照占比大：去掉它之后差距超过全局 KV 的 4 倍
