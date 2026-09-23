# Conformal Prediction

A collection of notebooks applying **split conformal prediction** to different computer vision tasks, based on [Angelopoulos & Bates (2021)](https://arxiv.org/pdf/2107.07511).

## What is Conformal Prediction?

Standard models output a single prediction. That hides uncertainty: a classifier might assign 45% to class A and 44% to class B yet still confidently return A. **Conformal prediction (CP)** wraps _any_ trained model to produce a **prediction set** $\hat{C}_\alpha(x)$ instead of a single answer, guaranteed to contain the true label with probability at least $1 - \alpha$:

$$
P\bigl(Y \in \hat{C}_\alpha(X)\bigr) \;\geq\; 1 - \alpha
$$

This is a **finite-sample, distribution-free** guarantee — it holds for any model, any architecture, any data distribution, as long as calibration and test data are **exchangeable** (roughly: i.i.d.). The only cost is a small held-out **calibration set** not used during training.

### The three-step recipe

| Step             | What happens                                                                                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **1. Train**     | Fit your model on the training set as usual                                                                                         |
| **2. Calibrate** | Run the model on a held-out calibration set; compute a _nonconformity score_ $s(x, y)$ for each point; extract a quantile $\hat{q}$ |
| **3. Predict**   | At test time, include output $y$ in the prediction set iff its score is $\leq \hat{q}$                                              |

### Nonconformity scores

A nonconformity score $s(x, y)$ measures how _surprising_ it would be for input $x$ to have label $y$ under the model — higher means more surprising. The specific score depends on the task; each notebook defines and explains its own.

### The calibration quantile

Next comes the critical step: define $\hat{q}$ to be the $\frac{\lceil (n+1)(1-\alpha) \rceil}{n}$ empirical quantile of $s_1, \ldots, s_n$, where $\lceil \cdot \rceil$ is the ceiling function ($\hat{q}$ is essentially the $1 - \alpha$ quantile, but with a small correction).

Intuitively, $\hat{q}$ acts as a threshold of surprise: it represents the maximum nonconformity score you are willing to tolerate while still ensuring the true label is covered with probability at least $1 - \alpha$.

Practically, this means you sort your $n$ calibration scores in ascending order and pick the score at rank $k = \lceil (n+1)(1-\alpha) \rceil$. For example, with $n = 1{,}000$ calibration points and $\alpha = 0.10$ ($90\%$ target coverage), $k = \lceil 1{,}001 \times 0.90 \rceil = 901$. That 901st score becomes your cutoff $\hat{q}$: at test time, any candidate label whose nonconformity score is $\leq \hat{q}$ is included in the prediction set.

### Coverage guarantee & what it actually means

The guarantee is **marginal**:

$$
\mathbb{P}\bigl(Y_{\text{test}} \in \hat{C}_\alpha(X_{\text{test}})\bigr) \;\geq\; 1 - \alpha
$$

under the key assumption of **exchangeability** (i.e. calibration data and future test samples are drawn from the same distribution).

It is important to understand what this guarantee does and does not promise:

- ❌ **It does not mean** every individual calibration set guarantees $\geq 1 - \alpha$ coverage on its own. A specific calibration split may result in a threshold $\hat{q}$ whose true population coverage is slightly above or below $1-\alpha$.
- ❌ **It does not mean** that any finite test batch (e.g. 5,000 test images) will exhibit _exactly_ $\geq 1 - \alpha$ empirical coverage. Natural binomial sampling variance and calibration variance mean individual runs might show e.g. 89.1% or 90.6% when the target is 90%.
- ✅ **It means** the entire procedure—over repeated draws of calibration sets and test points—guarantees that in the long run, at least $1 - \alpha$ of test points will be covered.

### Evaluating Conformal Prediction

Following Chapter 3 of [Angelopoulos & Bates (2021)](https://arxiv.org/pdf/2107.07511), we evaluate practical properties beyond basic marginal coverage:

1. **Adaptivity (Section 3.1):** Marginal coverage only guarantees $1 - \alpha$ _on average_. We check **Feature-Stratified Coverage (FSC)** to ensure hard classes aren't secretly undercovered, and **Size-Stratified Coverage (SSC)** to verify reliability when the model outputs small vs. large prediction sets.
2. **Correctness Checks (Section 3.3):** By precomputing scores once on all held-out data, we can repeatedly re-split that pool into hundreds of random (calibration + test) combinations in seconds. This verifies that empirical coverage matches theoretical moments across repeated trials, confirming there are no implementation bugs.

## Installation & Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for package management. You can install dependencies for either **CPU** or **GPU**:

### Option A: GPU

```bash
uv sync --extra gpu
```

### Option B: CPU

```bash
uv sync --extra cpu
```

## Repo structure

```
conformal-prediction/
├── image_classification/
│   └── svhn.ipynb          — split CP with LAC score on Street View House Numbers (SVHN)
└── object_detection/
    ├── helpers.py          — Pascal VOC label parsing, greedy IoU matching & geometry utilities
    └── voc_2007.ipynb      — Two-Step CP (ClassThr label sets & Box-Mult adaptive box intervals) on Pascal VOC 2007 with YOLO26
```

## Further reading

- Anastasios N. Angelopoulos & Stephen Bates, [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511) (arXiv:2107.07511)
- Alexander Timans, Christoph-Nikolas Straehle, Kaspar Sakmann, & Eric Nalisnick, [Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](https://arxiv.org/abs/2403.07263) (arXiv:2403.07263)
