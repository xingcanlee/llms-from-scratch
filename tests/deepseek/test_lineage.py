from llms_from_scratch.deepseek.lineage import (
    RELEASES,
    fp4_bytes,
    global_kv_bytes_per_token,
    v41_bytes,
)


def test_v41_flash_global_kv_is_890_bytes():
    # 主 KV：512 维 FP4 + 每 16 维一个 E4M3 缩放 = 288 字节；索引器键：128 维 MXFP4 = 68 字节
    assert fp4_bytes(512, 16) == 288 and fp4_bytes(128, 32) == 68
    # 编码器 3 个 Full 层（m = 2）+ 解码器 1 个 Full 层（m = 1）
    assert v41_bytes([0, 0] + [2] * 18 + [1] * 20, [2, 8, 14, 20]) == 3 * 356 / 2 + 356 == 890


def test_generation_ratios_match_official_claims():
    kv = global_kv_bytes_per_token()
    # 模型卡：相对 V4-Flash 约 4 倍、相对 DeepSeek-V1 约 437 倍
    assert round(kv["DeepSeek-V1 67B"] / kv["DeepSeek-V4.1-Flash"]) == 437
    assert 3.5 < kv["DeepSeek-V4-Flash"] / kv["DeepSeek-V4.1-Flash"] < 4.5
    # V3：61 层 × (512 + 64) × 2 字节 = 68.6 KiB
    assert kv["DeepSeek-V3"] / 1024 == 68.625


def test_releases_are_chronological_and_unique():
    dates = [r.date for r in RELEASES]
    assert dates == sorted(dates)
    assert len({r.repo for r in RELEASES}) == len(RELEASES)
