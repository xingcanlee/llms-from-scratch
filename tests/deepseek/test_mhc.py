import torch

from llms_from_scratch.deepseek.mhc import activation_traffic, hc_split_sinkhorn, run_sublayers


def test_split_sinkhorn_ranges():
    torch.manual_seed(0)
    hc = 4
    pre, post, comb = hc_split_sinkhorn(torch.randn(10, (2 + hc) * hc), torch.ones(3),
                                        torch.zeros((2 + hc) * hc), hc)
    assert (pre > 0).all() and (pre < 1 + 1e-5).all() and (post > 0).all() and (post < 2).all()
    torch.testing.assert_close(comb.sum(-2), torch.ones(10, hc), atol=1e-5, rtol=0)
    torch.testing.assert_close(comb.sum(-1), torch.ones(10, hc), atol=1e-3, rtol=0)


def test_single_pass_input_does_not_depend_on_own_pre_coefficients():
    torch.manual_seed(0)
    hc, d, n_sub = 4, 8, 3
    x = torch.randn(5, hc, d)
    params = [(torch.randn((2 + hc) * hc, hc * d) * 0.1, torch.ones(3),
               torch.randn((2 + hc) * hc)) for _ in range(n_sub)]
    W = torch.randn(d, d) * 0.3

    def inputs_seen(params, single_pass):
        seen = []
        subs = [lambda h: (seen.append(h), torch.tanh(h @ W))[1]] * n_sub
        run_sublayers(x, subs, params, hc, single_pass)
        return seen

    # 只改第 2 个子层 pre 系数对应的投影行
    perturbed = [p for p in params]
    fn, s, b = params[1]
    fn = fn.clone()
    fn[:hc] += 1.0
    perturbed[1] = (fn, s, b)
    for single_pass, same in ((True, True), (False, False)):
        a, c = inputs_seen(params, single_pass), inputs_seen(perturbed, single_pass)
        assert torch.allclose(a[1], c[1]) == same  # 单遍：子层 1 读入不受自己的 A 影响
    # 单遍时被推迟的 A_1 改在子层 2 生效
    assert not torch.allclose(inputs_seen(params, True)[2], inputs_seen(perturbed, True)[2])


def test_traffic_formulas():
    n, d = 4, 5120
    assert activation_traffic(n, d, "three_kernels") == 2 * activation_traffic(n, d, "lower_bound")
    assert activation_traffic(n, d, "single_pass") == activation_traffic(n, d, "lower_bound")
    assert activation_traffic(n, d, "two_pass") - activation_traffic(n, d, "single_pass") == n * d
