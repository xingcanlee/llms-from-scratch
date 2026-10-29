import torch

from llms_from_scratch.deepseek.optim import (
    headwise_orthogonalize,
    sinkhorn_balance,
    sinkhorn_balanced_update,
    update_modal_bias,
)
from llms_from_scratch.pretraining.optimizers import newton_schulz


def test_sinkhorn_balance_equalizes_row_and_column_rms():
    torch.manual_seed(0)
    m, n = 512, 32
    G = torch.randn(m, n) * torch.rand(m, 1) * 10 * torch.rand(1, n)  # 行、列尺度都很不均匀
    G[:3] = 1e-9  # 几乎没收到梯度的行（低频 token）
    U = sinkhorn_balance(G)
    rows, cols = U[3:].square().mean(1), U.square().mean(0)
    torch.testing.assert_close(rows, torch.ones_like(rows), atol=1e-4, rtol=0)
    assert (cols - 1).abs().max() < 0.1 and (U[:3] == 0).all()


def test_update_step_is_bounded_and_uses_nesterov_momentum():
    torch.manual_seed(0)
    W, g, mom = torch.zeros(64, 16), torch.randn(64, 16), torch.zeros(64, 16)
    sinkhorn_balanced_update(W, g, mom, lr=1.0)
    torch.testing.assert_close(mom, 0.05 * g)
    assert abs(W.square().mean().sqrt().item() - 0.18) < 0.02  # 更新的 RMS ≈ γ


def test_headwise_orthogonalization():
    torch.manual_seed(0)
    G = torch.randn(4 * 8, 32)
    out = headwise_orthogonalize(G, 4)
    torch.testing.assert_close(out[8:16], newton_schulz(G[8:16]))
    assert not torch.allclose(out, newton_schulz(G))


def test_modal_bias_balances_each_modality_separately():
    bias = {"text": torch.zeros(4), "image": torch.zeros(4)}
    idx = torch.tensor([[0], [0], [0], [1], [2], [2], [2], [3]])
    img = torch.tensor([False, False, False, False, True, True, True, True])
    update_modal_bias(bias, idx, img, speed=0.1)
    assert bias["text"][0] < 0 < bias["text"][2]  # 文本只用了 0、1：0 过载，2 空闲
    assert bias["image"][2] < 0 < bias["image"][0]  # 图像集中在 2


def test_effort_penalty_decays_exponentially_and_groups_by_effort():
    import math

    from llms_from_scratch.deepseek.optim import effort_penalty_coef, effort_rewards
    k = [effort_penalty_coef(b, 1.0, 50, 25) for b in (50, 75, 100)]
    assert abs(k[1] / k[0] - math.exp(-1)) < 1e-12 and abs(k[2] / k[1] - math.exp(-1)) < 1e-12
    task = torch.tensor([1.0, 1.0, 1.0, 1.0])
    lengths = torch.tensor([1000.0, 4000.0, 1000.0, 4000.0])
    effort = torch.tensor([50.0, 50.0, 100.0, 100.0])
    adv = effort_rewards(task, lengths, effort, k0=0.5, b_min=50, tau=25, l_norm=4000, c_max=1.0)
    # 同样答对：低强度下短回答的优势更大；高强度下长度几乎不受罚
    assert adv[0] > 0 > adv[1] and adv[0] - adv[1] > adv[2] - adv[3] > 0
    torch.testing.assert_close(adv[:2].sum(), torch.tensor(0.0))
