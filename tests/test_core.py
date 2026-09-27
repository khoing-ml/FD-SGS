import math

import pytest
import torch

from fd_sgs.core import fd_direction, flow_grpo_step, relative_correction, rms, stein_field, stochastic_flow_step


def test_fd_reward_orientation_and_zero_difference():
    a = torch.zeros(2, 1, 2, 2)
    b = torch.ones_like(a)
    result = fd_direction(a, b, torch.tensor([0., 2.]), torch.tensor([1., 1.]))
    torch.testing.assert_close(result[0], b[0])
    torch.testing.assert_close(result[1], -b[1])
    assert not fd_direction(a, a, torch.zeros(2), torch.ones(2)).any()


def test_rbf_source_derivative_matches_autograd():
    torch.manual_seed(7)
    states, directions = torch.randn(3, 5), torch.randn(3, 5)
    d2 = torch.cdist(states, states).square()
    h = d2[d2 > 0].median() / math.log(4)
    expected = torch.zeros_like(states)
    for i in range(3):
        for j in range(3):
            source = states[j].clone().requires_grad_()
            kernel = torch.exp(-(states[i] - source).square().sum() / h)
            derivative, = torch.autograd.grad(kernel, source)
            expected[i] += (kernel.detach() * directions[j] + .2 * derivative) / 3
    torch.testing.assert_close(stein_field(states, directions, .2), expected)


def test_repulsion_points_outward_and_collapsed_particles_are_finite():
    x = torch.tensor([[-1.], [1.]])
    field = stein_field(x, torch.zeros_like(x), 1.)
    assert field[0] < 0 < field[1]
    assert torch.isfinite(stein_field(torch.zeros(4, 3), torch.ones(4, 3))).all()


def test_trust_region_and_zero_field():
    base, field = torch.randn(4, 5), torch.randn(4, 5)
    c = relative_correction(field, base, rho=.5, trust_ratio=.1)
    torch.testing.assert_close(rms(c) / rms(base), torch.full((4, 1), .1))
    assert not relative_correction(torch.zeros_like(field), base).any()


@pytest.mark.parametrize("target", [0., .2, .7])
def test_exploration_matches_plan_formula(target):
    x, v, noise = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)
    gamma, sigma = .0025, .9
    tilde = target / (1 + gamma * (1 - target))
    expected = (x + (tilde - sigma) * v + tilde * math.sqrt(gamma**2 + 2*gamma) * noise) / (1 + gamma*tilde)
    torch.testing.assert_close(stochastic_flow_step(x, v, sigma, target, gamma, noise), expected)


def test_zero_exploration_is_euler():
    x, v = torch.randn(4, 8), torch.randn(4, 8)
    torch.testing.assert_close(stochastic_flow_step(x, v, .8, .4, 0, x), x - .4*v)


@pytest.mark.parametrize("sigma,target", [(1., .9), (.7, .3), (.2, 0.), (0., 0.)])
def test_flow_grpo_zero_noise_is_euler(sigma, target):
    x, v = torch.randn(2, 4), torch.randn(2, 4)
    actual = flow_grpo_step(x, v, sigma, target, 0., torch.ones_like(x), sigma_cap=.9)
    torch.testing.assert_close(actual, x + (target - sigma) * v)


def test_flow_grpo_transition_moments():
    # At s=.5, g²=.1²; dt=-.1. With x=2 and v=0, drift correction
    # shifts the mean to 1.998 and transition variance is .001.
    noise = torch.randn(100000, generator=torch.Generator().manual_seed(3))
    samples = flow_grpo_step(torch.full_like(noise, 2.), torch.zeros_like(noise),
                             .5, .4, .1, noise, sigma_cap=.9)
    assert abs(float(samples.mean()) - 1.998) < .0005
    assert abs(float(samples.var()) - .001) < .00002


def test_flow_grpo_endpoints_are_finite_and_float32():
    x = torch.ones(2, 3, dtype=torch.bfloat16)
    for sigma, target in [(1., .95), (.2, 0.), (0., 0.)]:
        result = flow_grpo_step(x, x, sigma, target, .05, x, sigma_cap=.95)
        assert torch.isfinite(result).all()
        assert result.dtype == torch.float32


@pytest.mark.parametrize("kwargs", [{"noise_level": -1}, {"noise_level": float("nan")},
                                   {"sigma_cap": 1}, {"sigma_next": 1.1}])
def test_flow_grpo_invalid_parameters(kwargs):
    params = dict(sigma=1., sigma_next=.9, noise_level=.05, sigma_cap=.9)
    params.update(kwargs)
    with pytest.raises(ValueError):
        flow_grpo_step(torch.ones(2), torch.ones(2), noise=torch.ones(2), **params)
