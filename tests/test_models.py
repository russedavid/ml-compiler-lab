import torch
from compiler_lab.models import make_case


def test_zero_weights_is_residual_relu():
    model, x = make_case(channels=7, height=13, width=17)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        torch.testing.assert_close(model(x), x.relu(), rtol=0, atol=0)


def test_case_reproducible_and_does_not_mutate_global_rng():
    state = torch.random.get_rng_state().clone()
    a, x = make_case(channels=3, height=5, width=7)
    b, y = make_case(channels=3, height=5, width=7)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_explicit_gru_matches_independent_cell():
    from compiler_lab.models import FixedGRU
    torch.manual_seed(2718)
    model=FixedGRU().eval();x=torch.randn(2,3,16);h=torch.randn(2,16)
    expected=h
    with torch.no_grad():
        for step in range(3):expected=model.cell(x[:,step],expected)
        torch.testing.assert_close(model(x,h),expected,rtol=1e-5,atol=1e-6)
