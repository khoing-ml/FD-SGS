# Improving Stein-Guided Sampling with Finite-Difference Reward Guidance and Few-Step Sampling

## Working title

**Finite-Difference Stein-Guided Sampling for Black-Box Reward Alignment in Few-Step Diffusion and Flow Models**

Alternative names:

- **FD-SGS: Finite-Difference Stein-Guided Sampling**
- **ZO-SGS: Zeroth-Order Stein-Guided Sampling**
- **Few-Step Stein-Guided Sampling with Pairwise Reward Differences**

---

## 1. Motivation

The current Stein-Guided Sampling (SGS) framework performs test-time alignment by evolving a set of interacting particles under three components:

1. the pretrained diffusion/flow dynamics,
2. a kernel-weighted reward-attraction term,
3. a particle-repulsion term.

In the current formulation, the reward-attraction term requires an intermediate reward gradient
$$
\nabla_x \log h_t(x),
$$
where
$$
h_t(x_t)
=
\mathbb E_{X_1\sim p_{1|t}(\cdot\mid x_t)}
\left[\exp(r(X_1))\right].
$$

This creates two practical limitations:

- the reward must be differentiable, or at least admit a usable gradient estimator;
- the current text-to-image experiments use a large number of inference steps, while modern image generators increasingly operate in a few-step regime.

The goal of this extension is therefore:

> **Replace the explicit reward gradient in SGS with a finite-difference reward-improvement direction, and redesign the Stein correction for 8-NFE and 10-NFE samplers.**

This directly extends SGS toward non-differentiable rewards such as black-box preference models, API-based evaluators, programmatic checks, or human feedback.

---

## 2. Key observation from Finite Difference Flow Optimization

The main inspiration is **Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models** (McAllister et al., 2026).

A crucial point is that their finite-difference construction is **not** the standard coordinate-wise or SPSA estimator
$$
\frac{R(x+\mu v)-R(x-\mu v)}{2\mu}v.
$$

Instead, they generate two nearby trajectories and use their **terminal reward difference** and **terminal image difference**.

For two nearby endpoints \(x_T\) and \(x_T'\),
$$
\Delta R = R(x_T')-R(x_T),
$$
$$
\Delta x = x_T'-x_T.
$$

Their theoretical prototype forms
$$
\widetilde g
=
\frac{\Delta R}{\sigma_c^2}\Delta x.
$$

In their practical implementation, the endpoint displacement is normalized:
$$
\overline{\Delta x}
=
\frac{\Delta x}
{\operatorname{RMS}(\Delta x)+\epsilon},
$$
and the update direction is effectively
$$
g^{\mathrm{FDFO}}
=
\Delta R\,\overline{\Delta x}.
$$

The important interpretation is:

> The difference between two nearby generated samples supplies a direction in output space, while the reward difference determines whether that direction should be reinforced or reversed.

Their analysis relates the expectation of this reward-weighted difference to the gradient of a **smoothed reward**, transformed by the Jacobian of the remaining flow. This makes the construction applicable even when the reward itself is non-differentiable.

FDFO also evaluates a **10-step fast configuration**, showing that the finite-difference idea is compatible with a low-step generation regime.

---

## 3. Proposed extension: Finite-Difference Stein-Guided Sampling

### 3.1 Original SGS interaction

For particles
$$
\{X_t^{(i)}\}_{i=1}^K,
$$
the current SGS interaction has the form
$$
\frac{dX_t^{(i)}}{dt}
=
u_t(X_t^{(i)})
+
\frac{\lambda_t}{K}
\sum_{j=1}^K
\left[
k(X_t^{(j)},X_t^{(i)})
\nabla_{X_t^{(j)}}\log h_t(X_t^{(j)})
+
\gamma_t
\nabla_{X_t^{(j)}}
k(X_t^{(i)},X_t^{(j)})
\right].
$$

The proposed method replaces
$$
\nabla_{X_t^{(j)}}\log h_t(X_t^{(j)})
$$
with a finite-difference reward direction.

---

## 4. FDFO-style reward direction for SGS

For particle \(j\), construct a pair of nearby realizations. Denote their predicted terminal outputs by
$$
\widehat z_t^{(j,a)},
\qquad
\widehat z_t^{(j,b)}.
$$

Evaluate the black-box reward:
$$
r_t^{(j,a)}
=
r\!\left(\widehat z_t^{(j,a)}\right),
$$
$$
r_t^{(j,b)}
=
r\!\left(\widehat z_t^{(j,b)}\right).
$$

Define
$$
\Delta r_t^{(j)}
=
r_t^{(j,b)}-r_t^{(j,a)},
$$
and
$$
\Delta z_t^{(j)}
=
\widehat z_t^{(j,b)}
-
\widehat z_t^{(j,a)}.
$$

The normalized finite-difference reward direction is
$$
\boxed{
g_{t,\mathrm{FD}}^{(j)}
=
\Delta r_t^{(j)}
\frac{
\Delta z_t^{(j)}
}{
\operatorname{RMS}
\left(
\Delta z_t^{(j)}
\right)
+\epsilon
}.
}
$$

The resulting Stein field becomes
$$
\boxed{
\Psi_{t,\mathrm{FD}}^{(i)}
=
\frac{1}{K}
\sum_{j=1}^K
\left[
k(X_t^{(j)},X_t^{(i)})
g_{t,\mathrm{FD}}^{(j)}
+
\gamma_t
\nabla_{X_t^{(j)}}
k(X_t^{(i)},X_t^{(j)})
\right].
}
$$

The dynamics are then
$$
\frac{dX_t^{(i)}}{dt}
=
u_t(X_t^{(i)})
+
\lambda_t
\Psi_{t,\mathrm{FD}}^{(i)}.
$$

---

## 5. Why the combination with Stein interaction is interesting

The finite-difference direction obtained from a single pair can be noisy.

Independent finite-difference guidance would use
$$
X_t^{(i)}
\leftarrow
X_t^{(i)}
+
\eta_t
g_{t,\mathrm{FD}}^{(i)}.
$$

FD-SGS instead uses
$$
X_t^{(i)}
\leftarrow
X_t^{(i)}
+
\eta_t
\frac{1}{K}
\sum_j
k(X_t^{(j)},X_t^{(i)})
g_{t,\mathrm{FD}}^{(j)}
+
\text{repulsion}.
$$

Therefore nearby particles exchange reward-improvement directions.

This suggests a stronger hypothesis than the current SGS paper:

> **Particle interaction can aggregate noisy zeroth-order reward information, improving both exploration and finite-difference query efficiency.**

A particle whose local pair discovers a useful reward-improving direction can transmit that direction to nearby particles through the kernel.

At the same time, the repulsive term prevents all particles from following the same finite-difference direction and collapsing to one reward mode.

---

## 6. Important distinction from FDFO: training-time vs. inference-time

FDFO is a **post-training** method.

It can:

1. sample a full pair of trajectories,
2. evaluate terminal rewards,
3. compute \(\Delta R\Delta x\),
4. update the model parameters,
5. use the improved parameters on future trajectories.

SGS is a **test-time sampling** method.

It needs a reward direction while the current trajectory is still being generated.

Therefore, we cannot directly wait until the final image is generated and then use that direction at an earlier step of the same trajectory.

A causal approximation is required.

---

## 7. Causal finite difference using the predicted clean endpoint

At intermediate time \(t\), let
$$
P_t(x,v_\theta(x,t))
$$
denote the model-dependent reconstruction of the predicted clean endpoint.

For a flow model, this may be obtained from the current state and velocity prediction.  
For a diffusion parameterization, it corresponds to the appropriate \(x_0\)- or clean-sample prediction.

Define
$$
\widehat z_t
=
P_t(X_t,v_\theta(X_t,t)).
$$

For a nearby pair
$$
X_t^{(j,a)},
\qquad
X_t^{(j,b)},
$$
compute
$$
\widehat z_t^{(j,a)}
=
P_t(X_t^{(j,a)},v_\theta(X_t^{(j,a)},t)),
$$
$$
\widehat z_t^{(j,b)}
=
P_t(X_t^{(j,b)},v_\theta(X_t^{(j,b)},t)).
$$

Then use
$$
\Delta r_t^{(j)}
=
r(\widehat z_t^{(j,b)})
-
r(\widehat z_t^{(j,a)}),
$$
$$
\Delta z_t^{(j)}
=
\widehat z_t^{(j,b)}
-
\widehat z_t^{(j,a)}.
$$

This gives the causal estimator
$$
\boxed{
g_{t,\mathrm{FD}}^{(j)}
=
\Delta r_t^{(j)}
\frac{
\Delta z_t^{(j)}
}{
\operatorname{RMS}(\Delta z_t^{(j)})+\epsilon
}.
}
$$

This is FDFO-inspired, but it is usable online during test-time sampling.

---

## 8. How to construct nearby pairs

Several pairing strategies should be tested.

### 8.1 Twin-particle perturbation

For each base particle \(X_t^{(j)}\), create two nearby twins:
$$
X_t^{(j,\pm)}
=
X_t^{(j)}
\pm
\xi_t^{(j)},
$$
where
$$
\xi_t^{(j)}
\sim
\mathcal N(0,\sigma_{\mathrm{FD},t}^2 I).
$$

This is the cleanest finite-difference construction, but approximately doubles the particle batch.

---

### 8.2 Stochastic twin trajectories

Following the practical FDFO philosophy, start a pair from a shared state or shared initial noise and inject weak stochasticity during sampling:
$$
X_{n+1}^{(j,a)}
=
\Phi_n(X_n^{(j,a)};\omega_n^{(a)}),
$$
$$
X_{n+1}^{(j,b)}
=
\Phi_n(X_n^{(j,b)};\omega_n^{(b)}).
$$

The perturbation should be small enough that the two samples remain semantically related but differ in details that can be compared by the reward.

---

### 8.3 Pair existing neighboring particles

To avoid doubling the batch, use existing SGS particles.

For particle \(j\), choose
$$
\pi(j)
=
\arg\max_{\ell\neq j}
k(X_t^{(j)},X_t^{(\ell)}),
$$
and form
$$
\Delta r_t^{(j)}
=
r(\widehat z_t^{(\pi(j))})
-
r(\widehat z_t^{(j)}),
$$
$$
\Delta z_t^{(j)}
=
\widehat z_t^{(\pi(j))}
-
\widehat z_t^{(j)}.
$$

This is cheaper, but the pair may not be sufficiently local for the finite-difference interpretation.

This should therefore be treated as an efficiency ablation rather than the primary formulation.

---

## 9. Few-step formulation

The current SGS text-to-image experiments use many inference steps. For 8-NFE and 10-NFE models, simply applying the same continuous-time correction with a much larger discretization interval may be unstable.

A better formulation is **operator splitting**.

Let
$$
\Phi_n
$$
be one step of the pretrained sampler from \(t_n\) to \(t_{n+1}\).

First perform the pretrained step:
$$
\overline X_{n+1}^{(i)}
=
\Phi_n(X_n^{(i)}).
$$

Then, at selected guidance steps, apply a small Stein correction:
$$
X_{n+1}^{(i)}
=
\overline X_{n+1}^{(i)}
+
\eta_n
\Psi_{n,\mathrm{FD}}^{(i)}.
$$

Thus
$$
\boxed{
\text{pretrained few-step update}
\quad\rightarrow\quad
\text{small FD-Stein correction}.
}
$$

This is preferable to interpreting a large Euler discretization of the original interacting ODE.

---

## 10. Relative correction scaling

The current SGS experiments already indicate that overly aggressive Stein updates degrade performance.

This problem should become more severe as the number of solver steps decreases.

Instead of using the same absolute \(\lambda\) for 100-step and 8-step samplers, scale the correction relative to the pretrained displacement.

Define
$$
\Delta_{n,\mathrm{base}}^{(i)}
=
\overline X_{n+1}^{(i)}
-
X_n^{(i)}.
$$

Then use
$$
\boxed{
X_{n+1}^{(i)}
=
\overline X_{n+1}^{(i)}
+
\rho_n
\frac{
\|\Delta_{n,\mathrm{base}}^{(i)}\|
}{
\|\Psi_{n,\mathrm{FD}}^{(i)}\|+\epsilon
}
\Psi_{n,\mathrm{FD}}^{(i)}.
}
$$

Here \(\rho_n\) directly controls the correction size relative to the base solver step.

Suggested initial sweep:
$$
\rho
\in
\{0.01,0.025,0.05,0.1\}.
$$

---

## 11. Sparse guidance schedule for 10 NFE and 8 NFE

Finite-difference reward guidance need not be applied at every solver step.

Early clean predictions may be unreliable because the sample is still highly noisy.

Very late corrections may not leave enough remaining generative dynamics to restore image quality.

Therefore, begin with middle-to-late guidance schedules.

### 10-NFE candidates

$$
\mathcal G_{10}^{(A)}
=
\{4,6,8\},
$$

$$
\mathcal G_{10}^{(B)}
=
\{5,7,9\},
$$

$$
\mathcal G_{10}^{(C)}
=
\{4,5,6,7,8\}.
$$

### 8-NFE candidates

$$
\mathcal G_{8}^{(A)}
=
\{3,5,6\},
$$

$$
\mathcal G_{8}^{(B)}
=
\{4,6\},
$$

$$
\mathcal G_{8}^{(C)}
=
\{3,4,5,6\}.
$$

These are hypotheses to test, not assumptions.

---

## 12. NFE accounting

It is important to distinguish:

$$
\text{sequential denoiser NFE}
$$
from
$$
\text{total denoiser FLOPs / batch evaluations}
$$
and
$$
\text{reward evaluations}.
$$

If paired states are evaluated in parallel as one batch, an 8-step sampler still has
$$
8
$$
sequential denoiser steps, but the total compute can be approximately doubled if every particle has a twin.

Therefore all few-step experiments should report at least:

- sequential NFE,
- number of particles,
- number of twin trajectories,
- reward-model calls,
- denoiser forward calls measured at the batch-element level,
- wall-clock latency,
- peak memory.

This prevents an apparent "8-NFE" method from hiding a much larger compute budget.

---

## 13. Possible theoretical extension

Let the exact reward score be
$$
g_t(x)
=
\nabla_x \log h_t(x),
$$
and suppose the finite-difference direction satisfies
$$
\widehat g_t(x)
=
g_t(x)+e_t(x).
$$

The FD-SGS field can be written as
$$
\widehat\Psi_t
=
\Psi_t+\Delta_t,
$$
where
$$
\Delta_t(x)
=
\mathbb E_{Y\sim\rho_t}
\left[
k(Y,x)e_t(Y)
\right].
$$

The current SGS local KL-descent result has an approximation term that depends on mismatch from the pretrained marginal and the repulsion contribution.

A natural extension is a bound of the schematic form
$$
\frac{d}{ds}
\mathrm{KL}(\rho_{t,s}\Vert q_t)
\Big|_{s=0}
\le
-\lambda_t
D_k(\rho_t,q_t)
\left[
D_k(\rho_t,q_t)
-
\epsilon_t
-
\delta_{t,\mathrm{FD}}
\right],
$$
where
$$
\delta_{t,\mathrm{FD}}
$$
quantifies the RKHS norm of the finite-difference approximation error.

The theoretical message would be:

> **Local KL descent is preserved when the finite-difference approximation error is smaller than the useful Stein discrepancy signal.**

A stronger FDFO-inspired analysis could instead characterize the expected reward-difference direction as the gradient of a smoothed terminal reward transformed by the Jacobian of the remaining flow.

This second route may be more faithful to the actual estimator.

---

## 14. Main experimental matrix

### 14.1 Sampling-step comparison

| Backbone / sampler | NFE | Method | Reward signal |
|---|---:|---|---|
| Standard baseline | 50 or 100 | SGS | exact gradient |
| Standard baseline | 50 or 100 | FD-SGS | finite difference |
| Few-step model | 10 | unguided | none |
| Few-step model | 10 | independent FD guidance | finite difference |
| Few-step model | 10 | FD-SGS | finite difference |
| Few-step model | 8 | unguided | none |
| Few-step model | 8 | independent FD guidance | finite difference |
| Few-step model | 8 | FD-SGS | finite difference |

If differentiable rewards are available, also include exact-gradient SGS at 8 and 10 NFE as an oracle comparison.

---

## 15. Key ablations

### A. Finite difference vs. exact reward gradient

Compare
$$
\nabla r
$$
against
$$
\Delta r\,\Delta z.
$$

Question:

> How much performance is lost when gradients are removed?

---

### B. Independent FD guidance vs. FD-SGS

Use exactly the same finite-difference pairs.

Independent:
$$
X_i
\leftarrow
X_i
+
\eta g_i^{\mathrm{FD}}.
$$

FD-SGS:
$$
X_i
\leftarrow
X_i
+
\eta
\frac1K
\sum_j
k(X_j,X_i)g_j^{\mathrm{FD}}
+
\text{repulsion}.
$$

This isolates the benefit of particle interaction.

---

### C. Endpoint-difference normalization

Compare
$$
g=\Delta r\,\Delta z
$$
against
$$
g
=
\Delta r
\frac{\Delta z}
{\operatorname{RMS}(\Delta z)+\epsilon}.
$$

FDFO reports that normalization is important for stability, so this should be tested explicitly in FD-SGS.

---

### D. Pair construction

Compare:

1. symmetric twin perturbation,
2. stochastic twin trajectories,
3. nearest-neighbor particle pairing.

---

### E. Guidance-step schedule

For 8 and 10 NFE compare:

- all steps,
- early steps,
- middle steps,
- middle-to-late steps,
- last steps only.

---

### F. Number of particles

$$
K\in\{2,4,8,16\}.
$$

Question:

> Does increasing \(K\) improve finite-difference gradient aggregation in addition to improving exploration?

---

### G. Repulsion

Compare
$$
\gamma=0
$$
against several positive repulsion strengths.

This tests whether repulsion remains necessary once stochastic finite-difference pairing already creates local diversity.

---

### H. Perturbation magnitude

For explicit twin perturbations, sweep
$$
\sigma_{\mathrm{FD}}.
$$

Too small:
- reward differences become noisy or numerically weak.

Too large:
- the local finite-difference interpretation breaks down.

---

## 16. Metrics

For text-to-image experiments, retain the current evaluation suite:

- target reward,
- CLIPScore,
- Aesthetic Score,
- ImageReward,
- HPSv2.

For diversity:

- LPIPS,
- TCE,
- optionally DreamSim diversity.

For efficiency:

- sequential NFE,
- reward evaluations,
- batch-element denoiser evaluations,
- wall-clock latency,
- GPU memory.

The efficiency metrics are especially important for the 8-NFE and 10-NFE claims.

---

## 17. Strongest empirical story to target

The most compelling result would be something like:

> With the same black-box reward-query budget, independent finite-difference guidance becomes noisy or collapses, whereas FD-SGS remains stable because neighboring particles share reward-improving directions and repulsion preserves coverage.

A particularly strong ablation is:

$$
\text{same finite-difference pairs}
+
\text{same reward calls}
+
\text{same NFE}
$$

for

$$
\text{independent FD}
\quad\text{vs.}\quad
\text{FD-SGS}.
$$

If FD-SGS wins under matched query and compute budgets, the gain can be attributed directly to Stein interaction.

---


## 18. Exploration mechanisms for flow models

Finite-difference reward guidance alone is not sufficient: the sampler also needs a controlled mechanism for generating nearby alternatives that reveal useful reward-improving directions. This is especially important for deterministic flow models, where a fixed initial noise and ODE solver otherwise produce a single trajectory with little local exploration.

FDFO provides several useful design ideas for this purpose. The central principle is to separate:

$$
\text{exploration}
\quad\text{from}\quad
\text{reward-directed steering}.
$$

In the proposed FD-SGS framework:

- **stochastic flow sampling** generates local alternatives;
- **finite differences** determine which local change improves reward;
- **Stein attraction** communicates useful directions across particles;
- **Stein repulsion** preserves global population coverage.

This gives each mechanism a distinct role.

### 18.1 EDM-style stochastic exploration for flow matching

A naive Euler--Maruyama perturbation is not ideal for a flow model because adding noise after an ODE step can make the state inconsistent with the time/noise conditioning seen by the velocity network. FDFO instead adapts the EDM stochastic sampler to flow matching.

In FDFO's convention, sampling proceeds from \(t=1\) (noise) to \(t=0\) (data). For a deterministic step from \(t_i\) to \(t_{i+1}\), the stochastic sampler first **overshoots** to a slightly less noisy state

$$
\widetilde t_{i+1}
=
\frac{t_{i+1}}
{1-\gamma_i t_{i+1}+\gamma_i},
$$

takes the ODE step to \(\widetilde t_{i+1}\), and then injects exactly enough fresh Gaussian noise to return to the target noise level \(t_{i+1}\).

The resulting update in the FDFO parameterization is

$$
\widetilde x_{i+1}
=
x_i
+
(\widetilde t_{i+1}-t_i)
v_\theta(x_i,t_i),
$$

followed by

$$
x_{i+1}
=
\frac{
\widetilde x_{i+1}
+
\widetilde t_{i+1}
\sqrt{\gamma_i^2+2\gamma_i}\,
\epsilon_i
}{
1+\gamma_i\widetilde t_{i+1}
},
\qquad
\epsilon_i\sim\mathcal N(0,I).
$$

Here \(\gamma_i\) controls the fraction of uncertainty that is re-randomized at the step.

**Important convention note.** The current SGS manuscript uses \(t=0\) for noise and \(t=1\) for data, opposite to FDFO. The safest implementation is therefore to define

$$
\tau = 1-t
$$

and apply the FDFO stochastic step in \(\tau\)-space rather than copying the formula directly with the SGS time variable.

This stochastic flow sampler should be treated as the **exploration operator** of FD-SGS.

---

### 18.2 Deterministic anchor + stochastic probe

FDFO reports that one member of the trajectory pair can be deterministic while the other is stochastic without degrading the method substantially.

This is attractive for inference-time SGS.

For each particle, maintain an anchor trajectory

$$
X_n^{(j,0)}
$$

that follows the original deterministic few-step sampler, and generate only one stochastic probe

$$
X_n^{(j,1)}.
$$

The finite-difference signal becomes

$$
\Delta r_{j,n}
=
r(\widehat z_{j,n}^{(1)})
-
r(\widehat z_{j,n}^{(0)}),
$$

$$
\Delta z_{j,n}
=
\widehat z_{j,n}^{(1)}
-
\widehat z_{j,n}^{(0)}.
$$

Then

$$
g_{j,n}^{\mathrm{FD}}
=
\Delta r_{j,n}
\frac{
\Delta z_{j,n}
}{
\operatorname{RMS}(\Delta z_{j,n})+\epsilon
}.
$$

This has three advantages:

1. the anchor remains exactly on the pretrained deterministic solver;
2. stochasticity is used only as an exploration probe;
3. only one additional branch per particle is required.

This should be the default pairing strategy for the first few-step implementation.

---

### 18.3 Shared initial states / common random numbers

FDFO finds improved stability when paired trajectories start from the same initial noise.

The same principle should be enforced in FD-SGS:

$$
X_0^{(j,0)}
=
X_0^{(j,1)}.
$$

The two branches should differ only through controlled stochastic exploration.

This acts as a common-random-number variance-reduction mechanism: unrelated semantic variation caused by independent initial noise is canceled, so the observed

$$
\Delta r
\quad\text{and}\quad
\Delta z
$$

are more directly attributable to the local exploration perturbation.

For Stein guidance this is especially important because a noisy finite-difference direction from one particle can influence several neighboring particles through the kernel.

---

### 18.4 Weak uniform stochasticity as the default exploration schedule

FDFO evaluates three stochasticity schedules:

1. **uniform stochasticity** at every step;
2. **random interval stochasticity** concentrated around a randomly selected noise range;
3. **prior-only stochasticity** injected near the initial noise.

Their default and strongest overall choice is a weak uniform schedule,

$$
\gamma_i = 0.0025,
$$

for all sampling steps.

The exact numerical value should not be assumed to transfer unchanged to FD-SGS or to 8/10-NFE models, but the design lesson is useful:

> Start with weak, distributed stochastic exploration rather than a large perturbation at one step.

For FD-SGS, sweep an effective stochasticity multiplier around the FDFO scale and tune separately for 8-NFE and 10-NFE sampling.

A useful initial search is

$$
\gamma
\in
\{0,\gamma_0/2,\gamma_0,2\gamma_0,4\gamma_0\},
$$

where \(\gamma_0\) is a small baseline chosen for the specific flow parameterization.

---

### 18.5 Random interval exploration

Although FDFO finds uniform stochasticity strongest overall, its **interval schedule** provides a useful mechanism for exploring changes at different semantic scales.

For each rollout, sample an interval center \(c\), then define

$$
\gamma
\propto
\operatorname{LN}(t;c,\sigma_{\mathrm{int}}),
$$

where \(\operatorname{LN}\) is a logit-normal density over time.

The FDFO construction is

$$
c\sim
\mathcal N(\mu_{\mathrm{center}},
\sigma_{\mathrm{center}}^2),
$$

$$
\gamma
\leftarrow
\operatorname{LN}(t;c,\sigma_{\mathrm{int}}),
$$

$$
\gamma
\leftarrow
\frac{\gamma}{\sum_n\gamma_n},
$$

$$
\gamma
\leftarrow
\exp(w_{\mathrm{int}}\gamma)-1.
$$

FDFO uses

$$
\mu_{\mathrm{center}}=1.3,
\qquad
\sigma_{\mathrm{center}}=1.5,
\qquad
\sigma_{\mathrm{int}}=0.25,
\qquad
w_{\mathrm{int}}=3.
$$

For FD-SGS, the interval schedule is interesting as an **exploration ablation**:

- high-noise intervals probe coarse semantic/layout changes;
- low-noise intervals probe local appearance and detail;
- different particles can use different intervals, creating heterogeneous exploration across the ensemble.

This could interact particularly well with Stein aggregation: particles probing different time scales can exchange their locally useful directions.

---

### 18.6 Do not force the guidance schedule to match the noise schedule

An important negative result from FDFO is that weighting parameter updates to match the stochastic exploration interval performs worse than applying the discovered direction more uniformly.

For FD-SGS, this suggests keeping two schedules conceptually separate:

$$
\gamma_n^{\mathrm{explore}}
$$

for generating informative perturbations, and

$$
\eta_n^{\mathrm{guide}}
$$

for applying Stein corrections.

A perturbation discovered at one noise scale can contain information that remains useful at later steps. Therefore the method should not assume

$$
\eta_n^{\mathrm{guide}}
\propto
\gamma_n^{\mathrm{explore}}.
$$

This should be tested rather than hard-coded.

---

### 18.7 Persistent finite-difference directions across a short window

FDFO applies the same terminal difference direction to all timesteps of a sampled trajectory instead of recomputing an independent reward direction at each step.

For test-time sampling, full future information is unavailable, but a causal analogue is possible.

When a finite-difference direction is obtained at step \(n\),

$$
g_{n}^{\mathrm{FD}},
$$

reuse it for a short window:

$$
\bar g_{n}
=
\beta\bar g_{n-1}
+
(1-\beta)g_n^{\mathrm{FD}},
$$

and apply

$$
\bar g_n
$$

for the next one or two Stein corrections.

For an 8-NFE or 10-NFE sampler this could substantially reduce reward evaluations while also suppressing high-variance changes in guidance direction.

This is an FD-SGS extension motivated by FDFO's time-coherent update behavior, not a procedure directly evaluated in FDFO.

---

### 18.8 Branch--evaluate--merge exploration

A natural inference-time extension is to occasionally branch a particle into multiple stochastic probes:

$$
X_n^{(j,0)},
X_n^{(j,1)},
\ldots,
X_n^{(j,B)}.
$$

Evaluate predicted-clean rewards

$$
r_{j,b}
=
r(\widehat z_{j,b}),
$$

then either:

#### Best-direction estimator

Choose

$$
b^\star
=
\arg\max_b r_{j,b},
$$

and use

$$
g_j
=
(r_{j,b^\star}-r_{j,0})
\frac{
\widehat z_{j,b^\star}
-
\widehat z_{j,0}
}{
\operatorname{RMS}
(
\widehat z_{j,b^\star}
-
\widehat z_{j,0}
)+\epsilon
}.
$$

#### Multi-probe estimator

Use all probes:

$$
g_j
=
\frac1B
\sum_{b=1}^B
(r_{j,b}-r_{j,0})
\frac{
\widehat z_{j,b}
-
\widehat z_{j,0}
}{
\operatorname{RMS}
(
\widehat z_{j,b}
-
\widehat z_{j,0}
)+\epsilon
}.
$$

The main trajectory then continues from the anchor or from a reward-selected branch.

This is not part of FDFO, but follows directly from its paired-trajectory exploration idea and is well suited to a particle method.

---

### 18.9 Adaptive exploration based on particle collapse

The current SGS repulsion term already provides a measure of ensemble geometry. This can be used to adapt the amount of stochastic exploration.

Let

$$
D_n
=
\frac{2}{K(K-1)}
\sum_{i<j}
\|X_n^{(i)}-X_n^{(j)}\|^2
$$

measure particle spread.

Then increase stochastic exploration when the ensemble becomes too concentrated:

$$
\gamma_n
=
\gamma_{\min}
+
(\gamma_{\max}-\gamma_{\min})
\,
\operatorname{clip}
\left(
\frac{D_{\mathrm{target}}-D_n}
{D_{\mathrm{target}}},
0,1
\right).
$$

An alternative is to use the median kernel similarity:

$$
C_n
=
\operatorname{median}_{i\neq j}
k(X_n^{(i)},X_n^{(j)}).
$$

Large \(C_n\) indicates particle collapse, so exploration can be increased.

This is a new FD-SGS extension rather than a technique directly proposed by FDFO.

---

### 18.10 Adaptive exploration based on reward informativeness

Not every stochastic perturbation provides a useful finite-difference signal.

Define

$$
S_n
=
\operatorname{median}_{j}
|\Delta r_{j,n}|.
$$

If

$$
S_n \approx 0,
$$

the exploration radius may be too small or the current predicted-clean samples may be too noisy for the reward to distinguish.

If \(S_n\) is extremely large while image differences are also large, perturbations may no longer be local.

This motivates adapting \(\gamma_n\) to keep the finite-difference signal in an informative range.

For example,

$$
\gamma_{n+1}
=
\begin{cases}
c_\uparrow\gamma_n,
&
S_n<S_{\min},
\$$2mm]
c_\downarrow\gamma_n,
&
S_n>S_{\max},
\$$2mm]
\gamma_n,
&
\text{otherwise}.
\end{cases}
$$

This turns stochasticity into an active finite-difference probing mechanism rather than a fixed noise hyperparameter.

---

### 18.11 Trust region around the pretrained flow

FDFO explicitly studies regularization toward the pretrained model through a velocity-space penalty.

For test-time FD-SGS, the analogous mechanism is a **flow trust region**.

Let

$$
v_{n,\mathrm{base}}^{(i)}
$$

be the pretrained flow velocity and

$$
\delta v_{n,\mathrm{Stein}}^{(i)}
$$

the FD-SGS correction.

Constrain

$$
\frac{
\|\delta v_{n,\mathrm{Stein}}^{(i)}\|
}{
\|v_{n,\mathrm{base}}^{(i)}\|+\epsilon
}
\le
c_n.
$$

Equivalently,

$$
\delta v
\leftarrow
\delta v
\min
\left(
1,
\frac{
c_n\|v_{\mathrm{base}}\|
}{
\|\delta v\|+\epsilon
}
\right).
$$

This complements the relative correction scaling already proposed for few-step models and prevents exploration plus reward steering from overwhelming the pretrained transport.

---

### 18.12 Recommended exploration design for the first 10-NFE experiment

A minimal first implementation should use:

1. **deterministic anchor trajectory** for every SGS particle;
2. **one stochastic twin** per anchor;
3. **shared initial state/noise** between anchor and twin;
4. **EDM-style flow stochasticity**, not naive additive Gaussian noise;
5. **weak uniform exploration** as the first schedule;
6. **RMS-normalized endpoint/predicted-clean difference**;
7. **Stein aggregation only after the finite-difference direction is measured**;
8. **flow trust-region clipping** on the final correction;
9. **middle-to-late sparse Stein correction steps**;
10. separate reporting of reward calls, sequential NFE, total denoiser compute, and latency.

The conceptual pipeline is

$$
\boxed{
\text{anchor}
\rightarrow
\text{stochastic local probe}
\rightarrow
\Delta r\,\Delta z
\rightarrow
\text{Stein information sharing}
\rightarrow
\text{repulsion}
\rightarrow
\text{trust-region correction}.
}
$$

This gives the flow model a genuine **explore--evaluate--share--steer** mechanism instead of relying on repulsion alone for exploration.

---

## 19. Exploration ablation table to add

| Component | Variants | Question |
|---|---|---|
| Flow stochasticity | deterministic / naive noise / EDM-style | Does noise-consistent stochastic sampling matter? |
| Pair type | two stochastic / deterministic + stochastic | Is an anchor trajectory sufficient? |
| Initial state | shared / independent | How much does common-noise pairing reduce variance? |
| Stochasticity schedule | uniform / interval / prior | At what noise scales should the flow explore? |
| Exploration strength | several \(\gamma\) values | What is the local/nonlocal transition? |
| Guidance weighting | uniform / time-weighted | Should steering follow the exploration schedule? |
| Direction reuse | one step / EMA / short window | Can reward calls be reduced? |
| Probe count | \(B=1,2,4\) | Does multi-probe finite difference improve query efficiency? |
| Adaptive exploration | fixed / diversity-adaptive / reward-adaptive | Can the sampler automatically increase exploration when particles collapse? |
| Trust region | off / norm-clipped | Does constraining deviation preserve quality at 8/10 NFE? |

The most important first comparison is

$$
\boxed{
\text{deterministic SGS}
\quad\text{vs.}\quad
\text{stochastic-exploration FD-SGS}
}
$$

under matched sequential NFE and transparent total-compute accounting.


## 20. Suggested revised contributions

A revised paper could state the contributions as:

1. **Derivative-free Stein guidance.**  
   We introduce a finite-difference variant of Stein-Guided Sampling that replaces explicit reward gradients with pairwise reward-weighted output differences, enabling test-time alignment with non-differentiable rewards.

2. **Population aggregation of zeroth-order reward information.**  
   We show that Stein interactions allow particles to exchange noisy finite-difference reward directions, while kernel repulsion preserves diversity.

3. **Few-step test-time alignment.**  
   We adapt SGS to 8-NFE and 10-NFE generators using split base-model and Stein correction steps with sparse guidance schedules.

4. **Compute-aware evaluation.**  
   We compare exact-gradient guidance, independent finite-difference guidance, and FD-SGS under matched NFE, reward-query, and wall-clock budgets.

5. **Theoretical interpretation.**  
   We connect the finite-difference direction to ascent on a smoothed reward and characterize how finite-difference approximation error modifies the local Stein KL-descent behavior.

---

## 21. Draft method paragraph

### Finite-Difference Stein-Guided Sampling

Many practically relevant reward functions are non-differentiable or prohibit backpropagation, which prevents direct evaluation of the reward score used by standard SGS. To remove this requirement, we construct a zeroth-order reward direction from pairs of nearby model predictions. For each particle \(X_t^{(j)}\), we obtain two nearby predicted clean samples \(\widehat z_t^{(j,a)}\) and \(\widehat z_t^{(j,b)}\), and evaluate their rewards. We define the finite-difference direction
$$
g_{t,\mathrm{FD}}^{(j)}
=
\left[
r(\widehat z_t^{(j,b)})
-
r(\widehat z_t^{(j,a)})
\right]
\frac{
\widehat z_t^{(j,b)}
-
\widehat z_t^{(j,a)}
}{
\operatorname{RMS}
\left(
\widehat z_t^{(j,b)}
-
\widehat z_t^{(j,a)}
\right)
+\epsilon
}.
$$
This direction points toward the locally preferred prediction without requiring differentiation through either the reward model or the generative trajectory. We replace the exact reward score in SGS with this estimator and aggregate the resulting directions across particles through the Stein kernel:
$$
\Psi_{t,\mathrm{FD}}^{(i)}
=
\frac1K
\sum_{j=1}^K
\left[
k(X_t^{(j)},X_t^{(i)})
g_{t,\mathrm{FD}}^{(j)}
+
\gamma_t
\nabla_{X_t^{(j)}}
k(X_t^{(i)},X_t^{(j)})
\right].
$$
The kernel-weighted attraction lets particles exchange locally discovered reward-improving directions, while the repulsive component prevents collapse. This produces a derivative-free test-time alignment procedure that preserves the central interacting-particle structure of SGS.

---

## 22. Draft few-step paragraph

### Few-Step Stein Corrections

Directly transferring the continuous SGS update to an 8- or 10-step sampler can produce overly large corrections because each solver interval is substantially larger than in the original high-NFE setting. We therefore separate the pretrained solver step from the Stein correction. At step \(n\), we first compute
$$
\overline X_{n+1}^{(i)}
=
\Phi_n(X_n^{(i)}),
$$
where \(\Phi_n\) is one step of the pretrained few-step sampler. At selected guidance steps \(n\in\mathcal G\), we subsequently apply
$$
X_{n+1}^{(i)}
=
\overline X_{n+1}^{(i)}
+
\rho_n
\frac{
\|
\overline X_{n+1}^{(i)}
-
X_n^{(i)}
\|
}{
\|
\Psi_{n,\mathrm{FD}}^{(i)}
\|
+\epsilon
}
\Psi_{n,\mathrm{FD}}^{(i)}.
$$
The dimensionless parameter \(\rho_n\) controls the Stein correction relative to the magnitude of the pretrained update. This normalization makes the guidance strength more transferable across solvers with different numbers of inference steps.

---

## 23. Algorithm sketch

```text
Algorithm: Few-Step Finite-Difference Stein-Guided Sampling

Input:
    pretrained sampler Φ
    black-box reward r
    kernel k
    number of particles K
    sampling times {t_n}_{n=0}^N
    guidance-step set G
    correction ratios {ρ_n}
    repulsion strengths {γ_n}

Initialize:
    X_0^(i) ~ p_0,  i = 1,...,K

for n = 0,...,N-1:

    # 1. Base pretrained step
    Xbar_(n+1)^(i) = Φ_n(X_n^(i))

    if n not in G:
        X_(n+1)^(i) = Xbar_(n+1)^(i)
        continue

    # 2. Construct nearby paired states/predictions
    for each particle j:
        construct pair Xpair^(j,a), Xpair^(j,b)
        compute clean predictions zhat^(j,a), zhat^(j,b)

        Δr_j = r(zhat^(j,b)) - r(zhat^(j,a))
        Δz_j = zhat^(j,b) - zhat^(j,a)

        gFD_j =
            Δr_j * Δz_j /
            (RMS(Δz_j) + ε)

    # 3. Stein aggregation
    for each particle i:
        Ψ_i =
            (1/K) Σ_j [
                k(Xbar_j, Xbar_i) gFD_j
                + γ_n ∇_{Xbar_j} k(Xbar_i, Xbar_j)
            ]

    # 4. Relative-magnitude correction
    for each particle i:
        base_norm = ||Xbar_(n+1)^(i) - X_n^(i)||
        stein_norm = ||Ψ_i||

        X_(n+1)^(i) =
            Xbar_(n+1)^(i)
            + ρ_n * base_norm/(stein_norm + ε) * Ψ_i

return highest-reward final particle
```

---

## 24. Open questions

The following points should be resolved experimentally before making strong claims:

1. Does the predicted-clean-sample reward provide a sufficiently reliable signal at early timesteps?
2. Is terminal-space \(\Delta z\) the correct direction to insert directly into latent-space SGS, or is an explicit mapping back to the current latent required?
3. Is symmetric perturbation better than FDFO-style stochastic trajectory pairing?
4. Can existing particles be paired without doubling compute?
5. Does the Stein kernel reduce the variance of finite-difference directions in practice?
6. How should the RBF bandwidth be adapted when the state dimensionality or few-step trajectory geometry changes?
7. Is the repulsion term still beneficial when each particle already has a stochastic twin?
8. Which guidance steps provide the best reward-quality-diversity trade-off at 8 and 10 NFE?
9. How should compute be matched fairly against Best-of-\(N\), SMC/FKS, and independent finite-difference guidance?

---

## 25. Recommended first implementation

For the first prototype, keep the design minimal:

- **NFE:** 10 first, then 8.
- **Particles:** \(K=4\) and \(K=8\).
- **Pairing:** explicit stochastic twin per particle.
- **Reward:** same PickScore setup as the current paper.
- **Guidance steps for 10 NFE:** \(\{4,6,8\}\).
- **Guidance steps for 8 NFE:** \(\{3,5,6\}\).
- **Direction normalization:** RMS normalization, following FDFO.
- **Kernel:** same RBF + median heuristic as current SGS.
- **Correction scale:** relative correction ratio
  $$
  \rho\in\{0.01,0.025,0.05,0.1\}.
  $$
- **Baselines:**
  1. unguided few-step sampler,
  2. Best-of-\(K\),
  3. independent finite-difference guidance,
  4. exact-gradient SGS if available,
  5. FD-SGS.

The highest-priority experiment is the matched-budget comparison:

$$
\boxed{
\text{Independent FD}
\quad\text{vs.}\quad
\text{FD-SGS}
}
$$

with identical particles, finite-difference pairs, reward evaluations, and NFE.

---

## References

1. **Stein-Guided Test-Time Alignment for Diffusion and Flow Models.**  
   Current manuscript.

2. McAllister, D., Aittala, M., Karras, T., Hellsten, J., Kanazawa, A., Aila, T., & Laine, S.  
   **Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models.**  
   arXiv:2603.12893, 2026.

3. Liu, Q., & Wang, D.  
   **Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm.**  
   NeurIPS 2016.

---

## One-sentence project summary

> **FD-SGS replaces backpropagated reward gradients with FDFO-style pairwise reward differences and uses Stein particle interaction to aggregate these noisy black-box directions, enabling derivative-free test-time alignment for 8-NFE and 10-NFE diffusion/flow samplers.**
