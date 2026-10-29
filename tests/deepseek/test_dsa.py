import torch

from llms_from_scratch.deepseek.dsa import (
    SparseMLA,
    fp8_roundtrip,
    hadamard_matrix,
    indexer_distill_target,
    rotate_activation,
)
from llms_from_scratch.deepseek.v3 import MLA, V3Args


def test_hadamard_is_orthogonal_and_spreads_outliers():
    H = hadamard_matrix(16)
    torch.testing.assert_close(H @ H.T, torch.eye(16))
    q, k = torch.randn(5, 16), torch.randn(7, 16)
    torch.testing.assert_close(rotate_activation(q) @ rotate_activation(k).T, q @ k.T,
                               atol=1e-5, rtol=1e-5)
    spike = torch.zeros(16)
    spike[3] = 8.0  # 一个离群通道：旋转后能量均匀摊到 16 个通道
    assert rotate_activation(spike).abs().max() == 2.0


def test_rotation_flattens_channel_distribution():
    # 量化器偏爱“峰均比”小的向量：旋转后最大值相对 RMS 大幅下降，而点积不变
    torch.manual_seed(0)
    x = torch.randn(64, 128) * 0.1
    x[:, 5] = 50.0

    def crest(t):
        return (t.abs().amax(-1) / t.square().mean(-1).sqrt()).mean()

    assert crest(rotate_activation(x)) < crest(x) / 4
    torch.testing.assert_close(fp8_roundtrip(x).shape, x.shape)


def test_mha_and_mqa_modes_agree():
    torch.manual_seed(0)
    m = SparseMLA(V3Args(), index_topk=5)
    x = torch.randn(2, 12, 64)
    torch.testing.assert_close(m(x, "mha"), m(x, "mqa"), atol=1e-5, rtol=1e-4)


def test_topk_covering_all_tokens_is_dense_mla():
    torch.manual_seed(0)
    args = V3Args()
    sparse = SparseMLA(args, index_topk=64)
    dense = MLA(args, "naive")
    dense.load_state_dict({k: v for k, v in sparse.state_dict().items()
                           if not k.startswith("indexer")}, strict=False)
    x = torch.randn(2, 12, 64)
    torch.testing.assert_close(sparse(x), dense(x), atol=1e-5, rtol=1e-4)


def test_distill_target_is_a_distribution():
    p = torch.rand(2, 4, 6, 6).softmax(-1)
    t = indexer_distill_target(p)
    torch.testing.assert_close(t.sum(-1), torch.ones(2, 6))
