# Causal active learning

Choose the next data acquisition that is most informative about a target
query p(X_Y | do(X_T = ψ)), run it on the true system, add the rows to the
context, and repeat until a budget is used up. An acquisition is a batch of R
rows (default 3, ABCI Figure 4's N_t = 3): R rows under one intervention
do(X_i = v) on an intervenable node, all at the same value, or R observational
rows. The design rule is ABCI's bi-level optimisation (Toth et al., NeurIPS
2022, Eq. 4.15), extended with the observational candidate, with the expected
information gain written so that it needs only the model's predictions.

Abbreviations: EIG expected information gain; SCM structural causal model;
BO Bayesian optimisation; GP Gaussian process; UCB upper confidence bound;
KL Kullback–Leibler divergence; CRN common random numbers; CI confidence
interval; PFN prior-data fitted network; D number of variables; R rows per
acquisition.

## Package layout and the model protocol

The framework is model-agnostic. Only `integrations/` may import model code
(a test checks the import boundary with a fresh interpreter and an `ast`
scan), and CITNP never imports this package.

| module | contents |
|---|---|
| `protocols.py` | `CausalPredictiveModel`, `ExperimentEnvironment` (protocols only) |
| `config.py` | `ActiveLearningConfig` and its validation |
| `context.py` | `ExperimentalContext` (values, intervened mask, `append`) |
| `acquisitions.py` | `Acquisition`, `InterventionalQuery`, `CandidateSet`, `build_candidate_set`, `ConstrainedEnvironment` |
| `distributions.py` | `UnivariateGaussianMixture`, quadrature and Monte Carlo KL, quadrature entropy |
| `fantasies.py` | fantasy noise per node, fantasy contexts |
| `information_gain.py` | `estimate_expected_information_gain` (the three objectives), coherence gap |
| `value_search/` | `ValueDomain`, exact GP, BO, grid |
| `selection.py` | `select_next_acquisition`, outer re-estimation, `cost_adjusted_value`, `choice_is_clear` |
| `strategies.py`, `loop.py` | the strategies; `run_active_learning`, step records |
| `support_check.py` | `TrainingSupport` and the checks |
| `evaluation/` | records to tables, summaries (95% CI), plots |
| `integrations/citnp/` | CITNP adapter, linear SCM environment, task sources, support reader, experiment configs and runner, command |

To plug in another causal foundation model, implement `CausalPredictiveModel`:

- `predict_query(contexts, treatment, outcome, treatment_values)`: the query
  p(X_Y | do(X_T = ψ), context_b) as a `UnivariateGaussianMixture` [B, Q, K]
  on the raw scale, float64;
- `draw_fantasy_noise(num_variables, num_fantasies, num_rows, generator)`:
  the standard noise a fantasy is computed from (opaque to the core);
- `sample_fantasies(contexts, acquisitions, noise)`: fantasies [B, S, R, D];
- `supports_observational_prediction`: whether it can predict a new
  observational row p(X | D) (needed for the observational candidate's EIG).

Fantasies are drawn by plain sampling, row by row: row 1 from p(X | a, D) (for
an intervention the joint prediction under do(X_i = v), column i set to v
exactly), row 2 from p(X | a, D + row 1), and so on. This autoregressive order
is the exact predictive chain rule: the R rows of one experiment share the same
unknown mechanism and each row informs the next. Drawing them independently
from one prediction overstates what an R-row experiment reveals (the CITNP
adapter's `fantasy_row_sampling: independent` is for comparison only; the
closed-form test shows the disagreement at R = 3). The model never draws random
numbers itself, so the core decides when noise is fresh and when it is shared.

## The loop

Step 0 records the query prediction and its KL to the truth on the initial
context. Each later step, with D_t the real context:

1. predict the query p(Y | do(X_T = ψ), D_t) on the ψ values;
2. inner level: for every intervenable node, search the value v maximising
   EIG(i, v) (BO by default, a grid as an option);
3. outer level: re-estimate EIG at every node's best value and for an
   observational batch, with fresh and more numerous fantasies; pick the
   largest (cost-adjusted, see below);
4. run the chosen acquisition on the true system and append all R rows;
5. record the query, the realised information gain KL(after || before), the
   realised entropy reduction H(before) − H(after), the query KL to the
   truth, the support check and the size of the newest regime.

"EIG" below means the objective selected by `information_gain_objective`
(default `entropy_reduction`; see "Information gain objectives").

**The query and intervenable nodes.** `InterventionalQuery(treatment,
outcome, treatment_values)` names T, Y and the ψ values.
`intervenable_nodes` (None = all D nodes) alone decides the interventional
candidates, for every strategy; it models what is physically possible
(ABCI Figure 4(b): X3 cannot be intervened on). There is no separate setting
for the query nodes: to exclude T or Y, leave them out. Intervening on Y is
allowed when listed: under do(Y) the non-descendants of Y keep their
observational distribution, so such rows inform the upstream mechanisms.
`build_candidate_set` is the only place deciding which acquisitions are
allowed, and the loop always wraps the environment in `ConstrainedEnvironment`,
which raises `NonIntervenableNodeError` on any other intervention.

## Estimator versus optimiser

Two separate things: the estimator computes EIG(i, v) for one candidate; the
optimiser (BO or the grid) decides where to call it. As in ABCI, where a nested
Monte Carlo utility estimator (Eq. 4.11–4.12) is called by GP-UCB (Appendix
D.7), every BO step needs one EIG estimate.

ABCI's objective is I(Y; X_a | D), evaluated with a posterior over SCMs. A PFN
gives posterior predictive distributions directly, so the same mutual
information is written with predictions only:

    I(Y; X_a | D, ψ) = E_{x ~ p(x | a, D)} [ KL( p(y | do(ψ), D + x) || p(y | do(ψ), D) ) ],

with x all R rows. With S fantasies x_s and ψ_1..ψ_Q:

    EIG_hat(a) = (1/S) Σ_s KL_s,   KL_s = (1/Q) Σ_q KL( p(y | ψ_q, D + x_s) || p(y | ψ_q, D) ),
    standard error = std(KL_1..KL_S) / sqrt(S).

The KL between two univariate Gaussian mixtures is computed by trapezoidal
quadrature on 1024 points spanning the first mixture's components ± 8 standard
deviations (the integrand p (log p − log q) is negligible outside p's support),
with log-densities from `logsumexp` in float64 so log q stays finite where q is
tiny; a Monte Carlo version cross-checks it. All densities are on the raw scale
(the per-task standardisation changes with the context). The same model gives
the fantasies and the query, so EIG measures what the model expects to learn.

## Information gain objectives

For one query value ψ write p0(y) = p(y | do(ψ), D) for the current
prediction, p_x(y) = p(y | do(ψ), D + x) for the prediction after a fantasy x
(all R rows), and p̄ = E_x p_x for the average post-fantasy prediction, the
expectation over the model's own fantasy distribution p(x | a, D). Three
objectives (`information_gain_objective`), each averaged over the ψ values:

| name | formula | relation to I = H(p̄) − E_x H(p_x) | sign |
|---|---|---|---|
| `entropy_reduction` (default) | H(p0) − E_x H(p_x) | I + [H(p0) − H(p̄)] | may be negative |
| `kl_divergence` (before the change, the only one) | E_x KL(p_x ‖ p0) | I + KL(p̄ ‖ p0) | ≥ 0 |
| `mixture_entropy_reduction` | H(p̄) − E_x H(p_x) = E_x KL(p_x ‖ p̄) | I exactly | ≥ 0 |

I is the mutual information I(Y; x | D) between the query outcome and the
acquisition's rows under the model's predictive. **Derivation.** Start from
I = E_x KL(p_x ‖ p̄) and split the logarithm:

    E_x KL(p_x ‖ p̄) = E_x ∫ p_x log p_x − E_x ∫ p_x log p̄
                    = −E_x H(p_x) − ∫ p̄ log p̄          (E_x p_x = p̄)
                    = H(p̄) − E_x H(p_x).

Add and subtract log p̄ inside the KL to p0:

    E_x KL(p_x ‖ p0) = E_x ∫ p_x (log p_x − log p̄) + E_x ∫ p_x (log p̄ − log p0)
                     = I + ∫ p̄ (log p̄ − log p0) = I + KL(p̄ ‖ p0),

and add and subtract H(p̄) in the entropy form:

    H(p0) − E_x H(p_x) = [H(p̄) − E_x H(p_x)] + [H(p0) − H(p̄)] = I + H(p0) − H(p̄).

**Coherent model versus a PFN.** A coherent Bayesian predictive satisfies
p̄ = p0 (the prediction averaged over what it expects to see is the
prediction itself), so both correction terms vanish and the three objectives
are equal: the mutual information. A PFN is trained to approximate posterior
predictives but nothing forces p̄ = p0, so the objectives differ by the drift
KL(p̄ ‖ p0) ≥ 0 and the entropy shift H(p0) − H(p̄). The KL form rewards any
predicted change of the prediction, including a systematic shift the model
would make whatever it saw (drift); the entropy form scores only how much
narrower the prediction is expected to become, the standard formulation of
prediction-oriented Bayesian active learning, and is the default.

**Negative values.** The entropy form is negative when the fantasies make the
prediction wider on average. It is kept as a conservative acquisition value,
not clipped: an acquisition the model expects to leave it less certain scores
below zero. The GP standardises its targets, so the value search is
unaffected; the outer level's cost adjustment handles the sign (below).

**Estimation.** With S fantasies, per fantasy s and ψ: H(p0) and H(p_s)
(`entropy_quadrature`), KL(p_s ‖ p0) (`kl_estimator`), and KL(p_s ‖ p̄_S),
H(p̄_S), KL(p̄_S ‖ p0) with p̄_S the S-component average
(`average_mixture_divergences`, on one grid per candidate and ψ, p̄_S's, so
the S × K components are evaluated once rather than per fantasy). Each
objective's estimate is the mean of its S ψ-averaged terms and its standard
error their standard deviation / √S; for `mixture_entropy_reduction` the terms
share p̄_S, so the standard error is approximate, and each term is at most
log S (p̄_S contains p_s with weight 1/S), far above EIG values in practice.
All three objectives, the drift and the entropy shift are computed in every
call (inner level, outer level, dense diagnostic) and recorded; entropies and
the p̄ terms are quadratures in float64, negligible next to the forward passes,
and draw no random numbers. `information_gain.CHECK_OBJECTIVE_IDENTITIES`
(or `check_identities=True`) asserts the two identities per candidate to
1e-6 (quadrature KL only).

**Realised counterparts** (calibration plot, `steps.csv`): after the real
acquisition, `realised_entropy_reduction` = H(before) − H(after) for
`entropy_reduction` (may be negative), `realised_information_gain` =
KL(after ‖ before) for `kl_divergence`. `mixture_entropy_reduction` has no
realised quantity of its own and is plotted against the realised entropy
reduction (equal in expectation for a coherent model).

## Bi-level optimisation (ABCI Eq. 4.15, Appendix D.7)

    inner level, for every intervenable node i:   v_i* in argmax_v EIG(i, v)
    outer level:   a* = argmax over {(i, v_i*)} ∪ {observational batch} of cost_adjusted(EIG(a), cost(a))

**Value domain.** A standardised coordinate u ∈ [−1, 1] maps to a raw value:
`observational_range` (default, v = m_i + s_i c u with the node's
observational mean and standard deviation, the statistics the model's
standardisation uses, c = 3), `training_range` (the training prior's do-value
range, supplied by the integration) or `fixed_range` (one raw interval for all
nodes; the Figure 4 configs use it).

**BO per node**, as ABCI Appendix D.7: three initial values u = −1/2, 0, 1/2,
then eight GP-UCB iterations u_{k+1} = argmax_u μ_k(u) + γ σ_k(u), γ = 1
(Srinivas et al., 2010), with σ the posterior standard deviation of the latent
EIG (without observation noise), maximised on 512 values. The GP has a constant
mean, a Matérn 5/2 kernel, targets standardised before fitting, lengthscale in
[0.1, 4], hyperparameters by maximum marginal likelihood (L-BFGS, five starts,
optimised as one batched problem on one thread). The recommendation is the
evaluated value with the largest posterior mean, which averages the errors of
nearby evaluations instead of picking the luckiest estimate. All nodes run in
lockstep, one batched EIG evaluation per round.

**Fantasy noise and GP noise.** Default `fantasy_noise: independent`: every
evaluation draws fresh noise, so EIG_hat(v) = EIG(v) + error(v) with errors
independent across evaluations and variance equal to the squared standard
error: exactly GP regression's observation model. GP noise `auto` then
resolves to `standard_error`: the variance of point j is its squared standard
error (standardised scale) plus one fitted nugget in [1e-6, 1e-2], which absorbs
error in the standard errors themselves (KLs can be heavy-tailed) without
overriding them. Option `shared` (CRN): one noise draw per node per step, so the
estimate is a fixed function EIG_hat(v; ε) whose nearby errors are correlated;
comparisons between nearby values are sharper but the GP fits one sampled
curve; `auto` resolves to `fitted` (a nugget bounded by the mean squared
standard error over the variance of the estimates). Explicit
`standard_error`, `fixed` (1e-6) or `fitted` override `auto`. Each node's noise
comes from a generator that depends only on one draw from the inner-search
generator and the node index, so searching nodes in lockstep gives exactly the
results of searching each alone (with the Monte Carlo KL, which draws from a
shared generator, only in distribution). The Cholesky jitter retry is a
numerical safeguard, not a noise model.

**Outer re-estimation.** The inner-level values are maxima over noisy
estimates, biased upward by different amounts for different nodes (the
winner's curse), under either noise mode. So every node's best value and the
observational batch are re-estimated with `final_num_fantasies` (256) fresh
fantasies, and the largest wins; `choice_is_clear` is false when the runner-up
is within two combined standard errors.

**The observational candidate.** Included by default. Its EIG needs fantasy
observational rows p(X | D) from the model. Every current CITNP checkpoint has
`supports_observational_prediction = False` (the joint head always conditions
on one intervened node and was never trained on a query without
intervention), so the EIG strategy logs one warning, skips the candidate and
records `observational_candidate_available = False`; it never raises for this.
Strategies that do not score with the model (the random baselines, OBS) still
use it, since the true system produces the rows; plot captions note the
asymmetry. Optional costs (`observational_cost`, `intervention_cost`) make the
outer level maximise the cost-adjusted value (`cost_adjusted_value`):

    value / cost   if value ≥ 0,
    value × cost   if value < 0,

increasing in the value and decreasing in the cost on both sides. Dividing a
negative value by a larger cost would bring it closer to zero and rank the more
expensive acquisition higher; multiplying avoids that. With equal costs (the
default) the ranking is the plain one. `choice_is_clear` and
`exhaustive_true_gain` (whose mean true KL drops can be negative) rank the
same way.

**Cost, and when BO pays off.** For single-node interventions the value search
is one-dimensional for every D, so grid and BO costs both grow linearly in the
number of intervenable nodes: BO is cheaper by a constant factor per node (11
evaluations for a continuous optimum versus a few dozen grid values at the same
resolution). The advantage of BO that grows with dimension appears with
multi-node interventions, where the value space has one dimension per
intervened node and a grid grows exponentially. One EIG step costs (number of
intervenable nodes) × 11 × S fantasies × R sequential model calls, plus the
outer re-estimation.

## Coherence diagnostic

The predictive-update identity holds only if the model behaves like a coherent
Bayesian posterior predictive, in which case the average over fantasies of
p(y | ψ, D + x) equals p(y | ψ, D). A PFN is not guaranteed to. The outer level
reports per candidate the coherence gap KL( p(y | ψ, D) || (1/S) Σ_s p(y | ψ,
D + x_s) ), ψ-averaged (`scores.csv`); it is about 0 for the conjugate test
stub. A gap comparable to the EIG values means the ranking is not trustworthy
for that context. It is reported, never acted on. `scores.csv` also reports
the drift KL(p̄ ‖ p0) and entropy shift H(p0) − H(p̄) that separate the three
objectives (see "Information gain objectives").

## Training support

`TrainingSupport` holds the trained D values, the largest D the architecture
can take, the context size range, the minimum number of observational rows,
the maximum number of interventional regimes, the rows-per-regime range, the
do-value range and whether observational queries were trained. A regime is all
rows intervened on one node (repeated acquisitions on a node join its regime).
D above the architecture's maximum (the `extra_node` encoding's index
embeddings) raises before the loop; D not trained on, a context size outside
the range, too few observational rows, too many regimes or do-values outside
the range give a warning and the step's `outside_training_support` flag, at
every step; the newest regime's size is recorded (small regimes are rare in
training; not a flag).

For `oracle_d5_iw_a15_sc.yaml`: context 64–360 rows; at least 33 observational
rows; 0–5 interventional regimes on distinct nodes; at least 1 row per regime
(1.5% of regimes have exactly 1 row, 2.9% at most 2, 15% at most 10); do-values
on the raw scale, the same for every node, N(0, s²) with s ~ U(0.5, 3) or with
probability 0.1 Uniform(−w, w), w ~ U(3, 6); the check uses [−6, 6].

**Known difference, not flagged:** within a regime the training prior always
has distinct do-values (one draw per row), whereas an R-row acquisition repeats
one do-value R times. The model has never seen repeated do-values within a
regime.

## Strategies

| name | what it does | ABCI name |
|---|---|---|
| `expected_information_gain` | the bi-level EIG selection | U_CR |
| `observational` | observational rows only | OBS |
| `random_fixed_value` | uniform over intervenable nodes and the observational candidate; value `fixed_intervention_value` (0) | RAND-FIXED |
| `random` | as above; value uniform over `random_value_range` (or the node's value domain) | RAND |
| `treatment` | always intervene on T, value uniform over its domain; raises if T is not intervenable | – |
| `exhaustive_true_gain` | evaluation only: inner search per node, then every candidate run `true_gain_draws` times (default 32) on the true system with fresh draws; the largest mean drop in query KL to the truth wins, and the loop then runs it on a new real draw (an upper reference for ranking experiments; see below) | – |

**Why `exhaustive_true_gain` uses fresh draws.** Until the fix, it ran each
candidate once with a copy of the environment generator, so the rows it ranked
were exactly the rows the loop then acquired. The choice per step was the same,
but the chosen draw's luck entered the context: on panel (b) about two thirds
of its per-step drop in query KL was luck (0.033 on the seen rows versus 0.012
on fresh draws of the same experiment), the context drifted into states better
than the data justify, and from there it preferred observational rows (46% of
choices in steps 4-10), which move the prediction least, although on average
they were worse than the best intervention. Ranking by the mean over fresh
draws from its own generator (`true_gain` stream) and leaving the environment
generator to the real step removes both effects; the reference is then an
upper bound for ranking experiments by their expected true value.

ABCI's U_CD and U_CML (causal discovery and model learning utilities) need a
posterior over graphs or SCMs, which a PFN does not have; they are not
implemented.

## Evaluation outputs

The metric is ABCI's Query KLD (Appendix F, Eq. F.3): E_ψ KL( p_true(Y |
do(X_T = ψ)) || p_model(Y | do(X_T = ψ), D_t) ), by the same quadrature, on ψ
values fixed per task and shared by all strategies, restarts and settings (an
explicit `treatment_values` list, or `num_query_samples` draws from
`treatment_range`, once per task). The EIG targets the same ψ values.

`run` writes, with prefix `--out PREFIX`: `PREFIX_steps.csv` (one row per
setting × task × restart × strategy × step, steps 0..budget),
`PREFIX_scores.csv` (EIG strategy, per candidate: value, re-estimated EIG,
standard error, coherence gap, surrogate mean and standard deviation, the
objective, every objective's value and standard error, drift, entropy
shift), `PREFIX_bayesian_optimisation_trace.csv` (per BO evaluation: round, u,
value, EIG, standard error, proposing UCB, lengthscale, output scale, nugget,
noise mode, the objective and every objective's value), `PREFIX_surrogate.csv` (GP posteriors at `--surrogate_steps`),
`PREFIX_summary.csv` (per setting × strategy × step: `num_runs`,
`mean_query_kl`, `standard_error_query_kl`, `ci95_lower`, `ci95_upper` with the
Student t quantile on num_runs − 1 degrees of freedom,
`fraction_outside_training_support`, and
`fraction_observational_candidate_available`), `PREFIX_dense_value_diagnostic.csv`
(with `--dense_value_diagnostic`: EIG on 41 values per node with
`final_num_fantasies`, for both noise modes, with the distance between the BO
recommendation and the dense maximiser, and every objective's value),
`PREFIX_meta.json` and figures (query KL with 95% CI, calibration of the
predicted objective against its realised counterpart, chosen-acquisition
frequency, GP posterior per node with the dense diagnostic and the
observational EIG). `PREFIX_steps.csv` holds `information_gain_objective`,
`predicted_information_gain` (the selected objective),
`predicted_entropy_reduction`, `predicted_kl_divergence`,
`predicted_mixture_entropy_reduction`, `realised_information_gain` (KL) and
`realised_entropy_reduction`.

## ABCI Figure 4, inside the training distribution

The two configs in `integrations/citnp/configs/` replicate ABCI Figure 4 with
the CITNP model, inside its training distribution (`oracle_d5_iw_a15_sc.yaml`):
the fixed five-node DAG X1→X2, X1→X3, X1→X4, X2→X3, X2→X4, X3→X5, X4→X5 (ABCI
Figure 4; confirmed by Ruizi), query p(X5 | do(X3 = ψ)) with ψ ~ U[2, 5],
10 SCMs × 3 restarts, R = 3, 50 acquisitions, the baselines OBS, RAND-FIXED
(value 0) and RAND, and our EIG strategy in place of U_CR (same objective,
estimated through the PFN); panel (a) all nodes intervenable
(`abci_figure_4_all_intervenable.yaml`), panel (b) X3 not intervenable
(`abci_figure_4_treatment_not_intervenable.yaml`). Not replicable: U_CD and
U_CML. `exhaustive_true_gain` is not in ABCI's figure and roughly doubles the
cost; it is left out of the configs. `--strategies` cannot be combined with
`--experiment`, so add it through a copy of a config with
`strategies: [exhaustive_true_gain]`.
Both configs set `information_gain_objective: entropy_reduction` explicitly:
the first Figure 4 runs, made before the entropy objectives existed, used
`kl_divergence`; pass `--information_gain_objective kl_divergence` to
reproduce them.

| | ABCI | ours | reason |
|---|---|---|---|
| mechanisms | GP | the checkpoint's mechanism prior (linear Gaussian, inverse Wishart) | the model is trained on that prior; absolute KL values are not comparable with the paper |
| initial observational rows | 5 | 64 | the prior's minimum context size (64 rows, of which at least 33 observational; the initial context is purely observational) |
| final context | 155 rows | 214 rows | within the prior's 360-row maximum |
| RAND values and value domain | U(−7, 7) | U(−6, 6), `fixed_range` [−6, 6] | the prior's widest do-value range |
| query ψ | U[2, 5] | U[2, 5] | already inside the prior's do-value range |

Expected support flags: none. Small first regimes (3 rows) are recorded, not
flagged; the repeated do-value within an acquisition is the known difference
described under Training support. With current CITNP checkpoints the EIG
strategy cannot score the observational candidate, while OBS and the random
baselines can choose it; `plot-panels` notes this in the caption.

Seeds: task t's mechanism parameters and ψ values depend on (seed, t);
restart r's initial rows and environment stream on (seed, t, r); acquisition
randomness on (seed, t, r, strategy). Every strategy and both panels, run with
the same `--seed`, see the same SCMs, ψ values and initial rows per (t, r):
the comparisons are paired.

## Commands

```bash
pip install -e .   # once, so causal_active_learning is importable
python -m causal_active_learning.integrations.citnp run \
    --ckpt RUN/checkpoint.pt --model_config RUN/resolved_model_config.yaml \
    --data_config CITNP/experiments/configs/oracle_d5_iw_a15_sc.yaml \
    --experiment causal_active_learning/integrations/citnp/configs/abci_figure_4_all_intervenable.yaml \
    --out fig4_a [--seed 0] [--task_indices 0,1] [--restart_indices 0,1,2]
python -m causal_active_learning.integrations.citnp merge \
    --inputs fig4_a_part0 fig4_a_part1 --out fig4_a
python -m causal_active_learning.integrations.citnp plot-panels \
    --summaries fig4_a_summary.csv fig4_b_summary.csv \
    --experiments <yaml a> <yaml b> \
    --titles "All nodes intervenable" "X3 not intervenable" --out fig4.png
# ad hoc runs without an experiment file (k-shot style psi grid per task)
python -m causal_active_learning.integrations.citnp run ... --environment example \
    --fixed_graph_edges "0->1,1->2,2->4,0->3,3->4,0->2" --treatment 2 --outcome 4 \
    --seeds 0,1,2,3,4 --value_search bayesian_optimisation,grid --out al_example
python -m causal_active_learning.integrations.citnp run ... --environment prior \
    --n_tasks 50 --out al_prior
python -m causal_active_learning.integrations.citnp plot PREFIX [--all_calibration_pairs]
```

Every `ActiveLearningConfig` field given on the command line (for example
`--information_gain_objective kl_divergence`) overrides the experiment's
`active_learning` section.

## One design step (sequential wet-lab mode)

```python
import torch
from causal_active_learning import (
    ActiveLearningConfig, ActiveLearningGenerators, ExperimentalContext,
    InterventionalQuery, choice_is_clear, select_next_acquisition,
)
from causal_active_learning.integrations.citnp.model import CITNPPredictiveModel
from CITNP.evaluation.model_api import load_model

model = load_model(ckpt, model_config, "cuda", num_nodes=5)
adapter = CITNPPredictiveModel(model, "cuda")
context = ExperimentalContext(values, intervened)  # [N, 5] raw, [N, 5] bool
query = InterventionalQuery(2, 4, torch.linspace(2, 5, 16, dtype=torch.float64))
config = ActiveLearningConfig(acquisition_budget=1, intervenable_nodes=(0, 1, 3, 4))
chosen, scores = select_next_acquisition(
    adapter, context, query, config, ActiveLearningGenerators.from_seed(0)
)
print(chosen.describe(), choice_is_clear(scores))
# run it in the lab, then: context = context.append(new_rows, chosen.target_node)
```

## Tests

```bash
pytest testing/test_causal_active_learning_*.py          # ~3 min on a CPU
pytest testing/test_causal_active_learning_*.py -m slow  # + environment moments
```

The backward-compatibility test compares the `kl_divergence` runs bit for bit
with `testing/golden/causal_active_learning_kl_divergence.json`, recorded on the
commit before the entropy objectives (one torch thread; how to record it again
is in `testing/make_causal_active_learning_kl_reference.py`).

## References

- C. Toth, L. Lorch, C. Knoll, A. Krause, F. Pernkopf, R. Peharz, J. von
  Kügelgen. Active Bayesian Causal Inference. NeurIPS 2022.
- N. Srinivas, A. Krause, S. Kakade, M. Seeger. Gaussian Process Optimization
  in the Bandit Setting: No Regret and Experimental Design. ICML 2010.

## Out of scope

Multi-node interventions and multi-dimensional value search (the GP takes
vector inputs already); batch design across different nodes; variance
reduction for fantasies (raise `num_fantasies` first if the trace shows EIG
noise limiting BO); other BO acquisition rules; U_CD and U_CML; amortised
acquisition; non-simulated environments; observational predictions from CITNP
(plan Part B, a separate branch).
