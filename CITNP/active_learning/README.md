# Active learning on the joint model

Choose the next intervention (node and value) that tells a trained joint model
the most about one target query p(X_Y | do(X_T = ψ)), run it on the true
structural causal model (SCM), add the result to the context, and repeat until
the experiment budget is used up. The design rule is the bi-level optimisation
of ABCI (Active Bayesian Causal Inference, Toth et al., NeurIPS 2022, Eq. 4.15),
with the expected information gain (EIG) written so that it needs only the
model's predictions.

Abbreviations: EIG expected information gain; SCM structural causal model;
BO Bayesian optimisation; GP Gaussian process; UCB upper confidence bound;
KL Kullback–Leibler divergence; PFN prior-data fitted network.

| module | contents |
|---|---|
| `config.py` | `ActiveLearningConfig` (all settings, validated), `CandidateExperiment`, `candidate_target_nodes` |
| `predictive_model.py` | `PredictiveModel` protocol, `ModelApiPredictiveModel` (via `CITNP.evaluation.model_api` only), `QueryMixture` |
| `fantasy.py` | `ExperimentContext`, `append_rows_to_context`, `sample_fantasy_rows` |
| `information_gain.py` | `estimate_expected_information_gain`, quadrature and Monte Carlo KL of Gaussian mixtures |
| `gaussian_process.py` | exact GP (Matérn 5/2, known heteroscedastic noise, L-BFGS hyperparameter fit) |
| `design_optimisation.py` | `ValueDomain`, `BayesianOptimisationValueOptimiser`, `GridValueOptimiser`, `reevaluate_and_select`, `choice_is_clear` |
| `environment.py` | `ExperimentEnvironment` protocol, `LinearSCMEnvironment` (wraps `evaluation.oracle.LinearSCM`) |
| `training_support.py` | `TrainingSupport`: does a context stay inside the training prior's context shapes? |
| `loop.py` | `select_next_experiment` (one design step), `run_active_learning` (the loop), `ActiveLearningGenerators` |
| `experiment.py`, `plots.py` | tasks x strategies x seeds -> tables and figures (the `active-learning` command) |

## The loop

With D_t the current real context, one step is:

1. predict the query p(Y | do(X_T = ψ), D_t) for every ψ on the query grid;
2. inner level: for every candidate node X_i, find its most informative value
   v_i\* by BO, where each evaluation of EIG(i, v) draws S fantasy rows
   x^(s) ~ p(X_-i | do(X_i = v), D_t) from the joint head, forms the fantasy
   contexts D_t + {x^(s)} and predicts the query on each;
3. outer level: re-evaluate EIG(i, v_i\*) for every node with more, fresh
   fantasies and pick the largest;
4. run the chosen experiment on the true SCM, append the real rows
   (D_{t+1} = D_t + rows; the fantasy contexts are thrown away);
5. predict the query on D_{t+1}; record the realised information gain
   KL(after || before) and the KL to the truth, both averaged over ψ.

Candidate nodes are all nodes except T and Y (`exclude_query_nodes`, default);
with `exclude_query_nodes=False` T is a candidate too. Y never is: setting Y
removes its response to T, so the experiment says nothing about the query.

## Bi-level optimisation (ABCI Eq. 4.15, Appendix D.7)

    i* in argmax_i EIG(i, v_i*),   where for every i:  v_i* in argmax_v EIG(i, v).

**Inner level** (`BayesianOptimisationValueOptimiser`), per node, as ABCI
Appendix D.7: values are searched in the node's standardised coordinate u,
v = m_i + s_i u (m_i, s_i the node's mean and standard deviation in the model's
own standardisation of the context), on [−c, c] (`value_domain:
observational_range`, c = 3) or on the central 99% of the training prior's
do-value distribution (`prior_range`). Three deterministic initial points
(−c/2, 0, c/2), then eight GP-UCB iterations
u_{k+1} = argmax_u μ_k(u) + γ σ_k(u), γ = 1 (Srinivas et al., 2010), the UCB
maximised on a dense grid of 512 values. The GP has a constant mean, a Matérn
5/2 kernel, targets standardised before fitting, and known per-point noise
(the squared standard error of each EIG estimate, plus a jitter); its
hyperparameters maximise the log marginal likelihood (L-BFGS, 5 random starts,
lengthscale in [0.05, 2] domain widths). The recommendation is the evaluated
value with the largest posterior mean, not the largest noisy estimate.
`GridValueOptimiser` (values −2..2 in u) is the baseline inner level.

All nodes run in lockstep: each round, every node proposes one value and all
proposals are evaluated in one batched forward pass. Each node's GP restarts
come from a generator that depends only on one draw from the BO generator and
the node index, so lockstep and separate runs give identical results.

**Outer level** (`reevaluate_and_select`): picking the node with the largest
inner-level value would favour a node whose BO saw lucky noise, since the
maximum of noisy estimates is biased upward (the winner's curse). So every
node's recommended experiment is re-evaluated with `final_num_fantasies`
(256) fantasies whose noise is fresh (the outer-level generator), and the
largest re-evaluated EIG wins. The choice is flagged as not clear-cut when the
runner-up is within two combined standard errors.

## The EIG estimator (ABCI Eq. 3.5, 4.11–4.12)

ABCI maximises I(Y; X_a | D), the mutual information between the query and the
experiment outcome, and evaluates it with a posterior over SCMs. The model
gives posterior predictive distributions directly, not an SCM posterior, so the
mutual information is written in its predictive-update form, which needs only
predictions:

    I(Y; X_a | D, ψ) = E_{x ~ p(x | do(a), D)} [ KL( p(y | do(ψ), D + x) || p(y | do(ψ), D) ) ],

and, since ψ is independent of the outcome, averaged over the ψ grid:

    EIG(i, v) = (1/S) Σ_s (1/Q) Σ_q KL( p(y | ψ_q, D^(s)) || p(y | ψ_q, D) ).

Each KL between univariate Gaussian mixtures is computed by trapezoidal
quadrature on 1024 points spanning the first mixture's components ± 8 standard
deviations (`kl_estimator: quadrature`), or by Monte Carlo (`monte_carlo`).
Every estimate comes with its standard error over fantasies, which the GP uses
as observation noise. All densities are on the raw data scale in float64: the
per-task standardisation changes when a row is added, so standardised densities
of two contexts are not comparable. Fantasies and the query come from the same
joint model (the query is the marginal of Y in its joint prediction), so the
EIG measures what the model expects to learn, not a disagreement between two
models.

Multi-row experiments draw their fantasy rows autoregressively by default (draw
row 1, append it, predict again, draw row 2, ...), the exact predictive chain
rule. `fantasy_mode: independent` draws the rows i.i.d. from one prediction;
it is cheap but ignores that the rows of one experiment share the unknown
mechanism, so it misstates the information of a multi-row experiment.

## Common random numbers

BO models EIG(i, v) as a smooth function of v. If every evaluation drew fresh
fantasy noise, the GP would see that noise on top of the function and need
more evaluations. Instead, all evaluations of one node within one step reuse
the same standard noise (`model_api.JointSampleNoise`: a uniform per fantasy
that picks the mixture component by inverse cumulative distribution, plus the
normal draws of the covariance factor and diagonal). The fantasy rows then move
smoothly with v (the component changes only where a uniform crosses a weight
boundary). One noise draw per node per step, from the inner-level generator;
different nodes get independent draws.

The loop's random streams (`ActiveLearningGenerators`) are separate: inner
fantasy noise, outer fantasy noise, BO, the baselines' random choices, the
dense diagnostic, and the true SCM. Changing acquisition settings therefore
never changes the real data drawn for a given seed and sequence of experiments.
The true SCM uses a numpy generator, because the existing sampler
(`LinearSCM.sample`) takes one.

## Cost, and when BO pays off

Per step at D = 5 with three candidate nodes and S = 64: 11 lockstep rounds of
3 x 64 fantasy contexts, plus the outer re-evaluation of 3 x 256. BO needs 11
EIG evaluations per node for a continuous optimum, where a grid of the same
resolution needs a few dozen. For single-node interventions the value search is
one-dimensional for any number of variables D, so both grid and BO cost grow
linearly in the number of candidate nodes: BO is cheaper by a constant factor
per node, not by a factor that grows with D. The dimensional advantage of BO
appears with multi-node interventions, where the value space has one dimension
per intervened node and a grid grows exponentially (the GP already takes vector
inputs, [num_points, value_dimension]).

## Coherence caveat and diagnostic

The predictive-update identity holds exactly only if the model behaves like a
coherent Bayesian posterior predictive, i.e. its fantasies satisfy the
martingale property E_s[ p(y | ψ, D^(s)) ] = p(y | ψ, D). A PFN is not
guaranteed to (the joint head enforces no consistency between predictions). The
outer re-evaluation therefore reports, per node, the coherence gap

    KL( p(y | ψ, D) || (1/S) Σ_s p(y | ψ, D^(s)) ),  averaged over ψ,

(`scores.csv`, `coherence_gap`). It is about zero for a coherent predictive; a
gap comparable to the EIG values means the EIG ranking is not trustworthy for
that context. It is reported, never acted on.

## Training support (out-of-distribution check)

Before the loop, the worst case of the final context is compared with the run's
data config, and after every step the actual context is (`training_support.py`):
total rows within `n_ctx_range`; at least `n_obs_min + 1` observational rows;
at most min(`k_max`, D) interventional regimes, a regime being all rows
intervened on one node (in the prior, each regime targets a distinct single
node and every row of a regime has its own do-value, so repeated experiments on
one node form one regime). Leaving it gives a warning and the step's
`outside_training_support` flag, not an error.

For `oracle_d5_iw_a15_sc.yaml`: 64–360 context rows; an observational regime
in every context with at least 33 rows; 0–5 interventional regimes (P = 0.25,
0.27, 0.19, 0.13, 0.09, 0.06 for 0..5); every regime has at least one row, and
1.5% of interventional regimes have exactly one row (2.9% at most two, 15% at
most ten). Do-values are raw-scale N(0, s²), s ~ U(0.5, 3), or with probability
0.1 Uniform(−w, w), w ~ U(3, 6). So `rows_per_experiment = 1` is inside the
support, but a regime that stays at one or two rows is rare in training.

## The command

```bash
# fixed D = 5 example, X1->X2->X3->X5, X1->X4->X5, X1->X3; query X5 | do(X3)
python -m CITNP.evaluation active-learning --ckpt RUN/checkpoint.pt \
    --model_config RUN/resolved_model_config.yaml \
    --data_config CITNP/experiments/configs/oracle_d5_iw_a15_sc.yaml \
    --environment example --fixed_graph_edges 0->1,1->2,2->4,0->3,3->4,0->2 \
    --treatment 2 --outcome 4 --value_optimisers bayesian_optimisation,grid \
    --experiment_budget 10 --seeds 0,1,2,3,4 [--dense_value_diagnostic] --out al_example
# 50 tasks of the training prior (one random (T, Y) per task unless given)
python -m CITNP.evaluation active-learning ... --environment prior --n_tasks 50 --out al_prior
python -m CITNP.evaluation plot-active-learning al_example   # redraw the figures
```

Strategies (`--strategies`, default all): `expected_information_gain` (the
method), `random` (uniform node and value, ABCI's RAND), `observational`
(observational rows only, ABCI's OBS), `treatment` (always do(T), a strong
baseline since do(T) data speaks to the query directly) and
`exhaustive_true_gain` (evaluation only: runs each node's BO-chosen experiment
on the true SCM and keeps the one with the lowest KL to the truth, an upper
reference for the outer level). The strategies with an inner level run once per
`--value_optimisers` entry. Every strategy starts from the same initial
context and environment random state for a given (task, seed).

Outputs (prefix `--out`): `_steps.csv` (one row per task x strategy x seed x
step: chosen node and value, clear choice, predicted and realised information
gain, KL to truth, outside-support flag), `_scores.csv` (EIG strategies, one
row per candidate node: optimised value, re-evaluated EIG, standard error,
coherence gap, surrogate estimate), `_bayesian_optimisation_trace.csv` (every
BO evaluation: round, u, value, EIG, standard error, the UCB value that
proposed it, the GP lengthscale and output scale), `_surrogate.csv` (final GP
posterior per node on a grid), `_dense_value_diagnostic.csv`, `_meta.json`, and
figures `_kl_to_truth.png` (as ABCI Fig. 4/9 "Query KLD"), `_calibration.png`,
`_chosen_node.png`, `_surrogate_task*_seed*_step*.png`.
`--dense_value_diagnostic` (fixed example only, expensive) also evaluates EIG
on 41 values per node with `final_num_fantasies` at `--dense_value_diagnostic_steps`
and overlays it on the GP figure, to check that BO finds the optimum of
exhaustive search.

## One design step from Python (the sequential wet-lab mode)

```python
import torch
from CITNP.active_learning import (
    ActiveLearningConfig, ActiveLearningGenerators, ExperimentContext,
    ModelApiPredictiveModel, choice_is_clear, select_next_experiment,
)
from CITNP.evaluation.model_api import load_model

model = load_model(ckpt, model_config, "cuda", num_nodes=5)
predictive = ModelApiPredictiveModel(model, 5, "cuda")
context = ExperimentContext(values, intervened)  # [N, 5] raw, [N, 5] bool
config = ActiveLearningConfig(experiment_budget=1)
chosen, scores = select_next_experiment(
    predictive, context, treatment=2, outcome=4,
    query_treatment_values=torch.linspace(-3, 3, 13, dtype=torch.float64),
    config=config, generators=ActiveLearningGenerators.from_seed(0),
)
print(chosen.describe(), choice_is_clear(scores))
# run it in the lab, then: context = context.append(new_rows, chosen.target_node)
```

## Tests

```bash
pytest testing/test_active_learning.py            # fast (~35 s on a CPU)
pytest testing/test_active_learning.py -m slow    # + environment moments
```

## References

- C. Toth, L. Lorch, C. Knoll, A. Krause, F. Pernkopf, R. Peharz, J. von
  Kügelgen. Active Bayesian Causal Inference. NeurIPS 2022.
- N. Srinivas, A. Krause, S. Kakade, M. Seeger. Gaussian Process Optimization
  in the Bandit Setting: No Regret and Experimental Design. ICML 2010.

## Not implemented (out of scope)

Multi-node interventions and multi-dimensional acquisition optimisation; batch
design (several experiments chosen jointly); other BO acquisition rules;
amortised acquisition; non-simulated environments (the `ExperimentEnvironment`
protocol leaves room for a wet-lab batch read from a file); the observational
candidate for `model_api` models (the joint head has no observational
prediction, so `include_observational_candidate` raises NotImplementedError
there).
