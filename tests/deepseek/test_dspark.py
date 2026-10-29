import torch

from llms_from_scratch.deepseek.dspark import (
    expected_tokens,
    markov_draft,
    prefix_survival,
    schedule_verify_lengths,
)


def test_markov_head_adds_dependency_on_previous_draft():
    torch.manual_seed(0)
    V, r, k = 10, 3, 4
    base = torch.zeros(1, k, V)
    base[0, :, 2] = 1.0  # 并行头对每个位置都偏好 token 2
    emb, head = torch.zeros(V, r), torch.zeros(V, r)
    emb[2, 0] = 1.0
    head[5, 0] = 5.0  # “上一个是 2” → 强烈偏好 5
    draft, _ = markov_draft(base, torch.tensor([7]), emb, head)
    assert draft.tolist() == [[2, 5, 2, 5]]


def test_expected_tokens_matches_monte_carlo():
    torch.manual_seed(0)
    conf = torch.tensor([0.9, 0.8, 0.6, 0.3])
    trials = (torch.rand(200_000, 4) < conf).int().cumprod(-1).sum(-1).float() + 1
    for L in range(5):
        mc = (torch.minimum(trials - 1, torch.tensor(float(L))) + 1).mean().item()
        assert abs(mc - expected_tokens(conf, L)) < 0.01
    torch.testing.assert_close(prefix_survival(conf), torch.tensor([0.9, 0.72, 0.432, 0.1296]))


def test_scheduler_shortens_verification_under_load():
    confs = [torch.tensor([0.9, 0.8, 0.5, 0.2, 0.1]), torch.tensor([0.6, 0.3, 0.1, 0.05, 0.0])]

    def light(n):  # 批很小：多验证几乎不增加耗时（访存受限）
        return 1.0 + 0.001 * n

    def heavy(n):  # 批已经很大：每多一个 token 都要付计算时间
        return 1.0 + 0.3 * n

    long, short = schedule_verify_lengths(confs, light), schedule_verify_lengths(confs, heavy)
    assert sum(short) < sum(long)
    assert long[0] >= long[1] and short[0] >= short[1]  # 置信度高的请求分到更长的验证


def test_scheduler_matches_bruteforce():
    import itertools
    confs = [torch.tensor([0.95, 0.7, 0.4]), torch.tensor([0.5, 0.2, 0.1]),
             torch.tensor([0.8, 0.8, 0.8])]

    def cost(n):
        return 1.0 + 0.12 * n

    def rate(ls):
        return sum(expected_tokens(c, L) for c, L in zip(confs, ls)) / cost(len(ls) + sum(ls))

    best = max(itertools.product(range(4), repeat=3), key=rate)
    assert abs(rate(schedule_verify_lengths(confs, cost)) - rate(best)) < 1e-9
