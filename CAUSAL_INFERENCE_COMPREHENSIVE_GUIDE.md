# Causal Inference: Comprehensive Guide

**Version:** 1.0  
**Last Updated:** April 2026  
**Scope:** Fundamentals to Advanced Methods  
**Authors:** Research Compilation

---

## Table of Contents

1. [Causal Inference Fundamentals](#causal-inference-fundamentals)
2. [Causal Discovery](#causal-discovery)
3. [Causal Effect Estimation](#causal-effect-estimation)
4. [Advanced Methods](#advanced-methods)
5. [Applications & Tools](#applications--tools)
6. [References](#references)

---

## Causal Inference Fundamentals

### 1.1 Introduction to Causality

Causal inference addresses the fundamental question: **"What would happen if we intervened?"**

Unlike traditional statistical analysis that focuses on correlation and prediction, causal inference explicitly models interventions and their outcomes.

**Key Concepts:**
- **Correlation vs. Causation**: Correlation measures association between variables; causation measures the effect of intervention
- **Observational Studies**: Data collected without intervention
- **Randomized Controlled Trials (RCTs)**: Gold standard with randomized assignment

### 1.2 Causal Graphs and DAGs

**Directed Acyclic Graphs (DAGs)** provide a visual representation of causal relationships.

#### DAG Components:

```
Nodes: Variables (X, Y, Z, W)
Edges: Causal relationships (arrows indicate causation)
No Cycles: "Acyclic" means no variable causes itself indirectly
```

#### Example DAG:

```
    U (Unobserved Confounder)
   / \
  v   v
  X → Y ← Z
      ↑
      W
```

**Interpretation:**
- X causally affects Y
- Z causally affects Y
- U affects both X and Y (confounding)
- W affects Y directly
- Direction of arrows represents temporal causality

### 1.3 Confounding Variables and Backdoor Paths

**Confounding**: A variable that affects both treatment and outcome, creating spurious association.

**Backdoor Path**: A path from X to Y that does NOT go through the arrow X→Y.

#### Example Backdoor Path:

```
In the DAG: U ← X ← W → Y ← U

The path: X ← W → Y is a backdoor path
This creates spurious association between X and Y
```

**Colliders**: A variable with arrows coming in from both sides.

```
   X → Z ← Y
   
Z is a collider. Conditioning on Z creates association 
between X and Y even if they're not causally related.
```

### 1.4 Back-door Criterion

**Back-door Criterion** determines when we can estimate causal effects from observational data.

**Definition**: A set of variables S satisfies the back-door criterion relative to (X, Y) if:

1. No variable in S is a descendant of X
2. S blocks all backdoor paths from X to Y (paths not containing the edge X→Y)

**Mathematical Formulation:**

```
ATE = E[Y|do(X=x)] - E[Y|do(X=x')]

Can be estimated from observational data if back-door criterion is satisfied:

E[Y|do(X=x)] = ∫ E[Y|X=x, S=s] P(S=s) ds
```

**Example:**

```
DAG: U → X, U → Y, X → Y, W → X, W → Y

Variables satisfying back-door criterion: {U, W}
This blocks: U → Y and W → Y
```

### 1.5 Front-door Criterion

**Front-door Criterion** provides identification when back-door criterion fails.

**Definition**: A set of variables M satisfies the front-door criterion if:

1. M blocks all directed paths from X to Y
2. There are no unblocked backdoor paths from X to M
3. All backdoor paths from M to Y are blocked by X

**Formula:**

```
E[Y|do(X=x)] = ∫_m ∫_x' P(M=m|X=x) E[Y|X=x', M=m] 
               [P(X=x') - P(X=x'|do(X=x))] dx' dm
```

### 1.6 d-separation and Graphical Identification

**d-separation (d for "directional")**: A graphical criterion for conditional independence.

**Definition**: Two variables X and Y are d-separated given set Z if all paths between X and Y are blocked by Z.

**Path Blocking Rules:**

1. **Chain**: X → M → Y
   - Blocked by conditioning on M: X ⊥ Y | M
   
2. **Fork**: X ← M → Y
   - Blocked by conditioning on M: X ⊥ Y | M
   
3. **Collider**: X → M ← Y
   - Blocked by NOT conditioning on M or its descendants
   - Conditioning on M creates association: X ⊥ Y, but X ⊥ Y | M may be false

**Example d-separation Analysis:**

```
DAG: X → M → Y, X ← U → Y

Path 1: X → M → Y (chain)
Path 2: X ← U → Y (fork through U)

Given M: Path 1 is blocked
Given M, U: Both paths blocked → X ⊥ Y | {M, U}
```

**Graphical Identification**: Using d-separation to identify causal effects.

```
Causal effect E[Y|do(X=x)] is identified if:
1. All backdoor paths from X to Y can be blocked
2. Or front-door criterion is satisfied
3. Or other graphical criteria apply
```

### 1.7 Pearl's Causal Calculus

**Pearl's do-Calculus**: A formal system for manipulating causal expressions.

**Notation:**

```
do(X=x): Intervention (setting X to x)
E[Y|do(X=x)]: Causal effect (post-intervention expectation)
```

**Three Rules of Do-Calculus:**

**Rule 1 (Ignoring Observations):**
```
If (Y ⊥ Z | X, W) in G_{X̄}:
P(Y|do(X=x), Z=z, W=w) = P(Y|do(X=x), W=w)

Where G_{X̄} is the graph with outgoing edges from X removed
```

**Rule 2 (Action/Observation Exchange):**
```
If (Y ⊥ Z | X, W) in G_{X̄Z̄}:
P(Y|do(X=x), do(Z=z), W=w) = P(Y|do(X=x), Z=z, W=w)

Where G_{X̄Z̄} is the graph with edges from X and Z removed
```

**Rule 3 (Ignoring Actions):**
```
If (Y ⊥ Z | X, W) in G_{X̄}:
P(Y|do(X=x), do(Z=z), W=w) = P(Y|do(X=x), W=w)

Where G_{X̄} has edges from Z removed
```

**Application Example:**

```
Identify: E[Y|do(X=x)]

Given DAG and satisfying back-door criterion with S:

Step 1: Use rules to show:
        E[Y|do(X=x)] = E[E[Y|X=x,S]|do(X=x)]
        
Step 2: Rule 1 allows:
        E[Y|do(X=x)] = ∫ E[Y|X=x, S=s] P(S=s|do(X=x)) ds
        
Step 3: Since S blocks backdoor:
        E[Y|do(X=x)] = ∫ E[Y|X=x, S=s] P(S=s) ds
```

### 1.8 Mathematical Framework: Structural Causal Models

**Structural Causal Model (SCM)** specifies causal relationships formally.

**Definition:**

```
For each variable V_i:
V_i := f_i(PA_i, U_i)

Where:
- PA_i: Parents of V_i (direct causes)
- U_i: Exogenous noise/unobserved factors
- f_i: Deterministic function
```

**Example SCM:**

```
U_x: Exogenous for X
U_y: Exogenous for Y
U_z: Exogenous for Z (unobserved confounder)

X := f_x(U_x)
Y := f_y(X, Z, U_y)
Z := f_z(U_z)

Implicit DAG:
U_x → X, U_z → Z, X → Y, Z → Y

Causal effect of X on Y:
E[Y|do(X=x)] = E[f_y(x, Z, U_y)]
```

**Post-Intervention Distribution:**

```
After intervention do(X=x):
- X is fixed to x
- All other mechanisms unchanged

P(Y|do(X=x)) = ∫∫ P(U_y, U_z) 
               δ(X - x) f_y(x, f_z(U_z), U_y) dU_y dU_z
```

---

## Causal Discovery

### 2.1 Overview of Causal Discovery

Causal discovery algorithms learn the causal structure (DAG) from data without prior knowledge.

**Assumptions Required:**
1. **Causal Markov Assumption**: Variables are independent given their parents
2. **Faithfulness Assumption**: Observed independencies match the DAG structure
3. **Acyclicity**: No cycles (variables don't cause themselves)

**Three Main Approaches:**
1. Constraint-based (tests for independence)
2. Score-based (maximize fit quality)
3. Functional causal models (assume specific functional forms)

### 2.2 Constraint-Based Algorithms

#### PC Algorithm (Peter-Clark)

**Idea**: Start with complete graph, remove edges based on conditional independence tests.

**Algorithm Steps:**

```
1. Start with fully connected undirected graph
2. For each pair of variables (X, Y):
   - For increasing conditioning set sizes:
     - If X ⊥ Y | S (conditional independence test):
       * Remove edge X-Y
       * Record S as the separating set
3. Orient edges using orientation rules (v-structures, etc.)
```

**Pseudocode:**

```python
def PC_algorithm(data, alpha=0.05):
    # Step 1: Skeleton learning
    graph = CompleteGraph(variables)
    
    for k in range(len(variables)-1):
        for (X, Y) in adjacent_pairs(graph):
            # Find conditioning set of size k
            S_candidates = other_adjacent_to_X_and_Y(graph)
            
            for S in combinations(S_candidates, k):
                p_value = independence_test(X, Y, S, data)
                
                if p_value > alpha:  # X ⊥ Y | S
                    remove_edge(graph, X, Y)
                    record_separating_set(X, Y, S)
                    break
    
    # Step 2: Orient edges (v-structures, colliders, etc.)
    orient_edges(graph)
    
    return graph
```

**Computational Complexity**: O(n^3) for n variables

**Advantages:**
- Theoretically sound under assumptions
- Identifies causal structure uniquely

**Disadvantages:**
- Sensitive to independence test choices
- May fail with strong dependencies
- Computational cost with many variables

#### FCI Algorithm (Fast Causal Inference)

**Enhancement of PC** that handles latent confounders.

**Key Differences:**

```
PC Algorithm:
- Assumes no unmeasured confounding
- Produces fully oriented DAG
- Causal relationships identified

FCI Algorithm:
- Allows latent confounders
- Produces Partial Ancestral Graph (PAG)
- May represent uncertainty about directions
```

**PAG Notation:**

```
X --> Y    : X causes Y (arrow direction certain)
X <-> Y    : Possible confounder between X and Y
X o-> Y    : Direction unknown
X o-o Y    : No causal relationship or confounder
```

**Algorithm Modifications:**

```
1. Skeleton learning (same as PC)
2. Determine v-structures (colliders)
3. Propagate orientation rules
4. Mark uncertain edges with 'o' notations
```

### 2.3 Score-Based Algorithms

#### Greedy Equivalence Search (GES)

**Idea**: Greedily search over DAGs to maximize score function.

**Search Strategy:**

```
1. Start with empty graph
2. Forward phase: Add edges that most increase score
3. Backward phase: Remove edges that don't decrease score much
4. Continue until convergence
```

**Score Functions:**

**BIC (Bayesian Information Criterion):**
```
BIC = log(n) * k - 2 * log(L)

Where:
- n: number of samples
- k: number of parameters
- L: likelihood

Lower BIC = Better fit with fewer parameters
```

**BGe Score (Bayesian Gaussian Equivalent):**
```
BGe Score = ∑_i BGe(V_i | PA_i)

BGe(V_i | PA_i) = log(Γ(α/2)) - log(Γ(α/2 + n/2))
                  + 0.5 * log(|T|) - 0.5 * log(|T + X'_i X_i|)

Where:
- Γ: Gamma function
- T: Hyperparameter matrix
- X_i: Data matrix
```

**Pseudocode (GES):**

```python
def GES(data, score_fn='BIC'):
    graph = EmptyGraph()
    scores = [score_fn(graph, data)]
    
    # Forward phase
    improved = True
    while improved:
        improved = False
        best_edge = None
        best_score = scores[-1]
        
        for (X, Y) not in graph.edges:
            graph.add_edge(X, Y)
            score = score_fn(graph, data)
            
            if score < best_score:  # Lower is better
                best_score = score
                best_edge = (X, Y)
            
            graph.remove_edge(X, Y)
        
        if best_edge:
            graph.add_edge(*best_edge)
            scores.append(best_score)
            improved = True
    
    # Backward phase (similar with edge removal)
    
    return graph
```

**Computational Complexity**: O(n^2) per iteration, multiple iterations needed

**Advantages:**
- Works with larger graphs
- More robust to violations of assumptions

**Disadvantages:**
- May converge to local optima
- Depends heavily on score function
- Slower than constraint-based methods

### 2.4 Functional Causal Models

**Idea**: Model causal relationships with specific functional forms.

**Basic Form:**

```
Y := f(X, U)

Where:
- X: Direct causes
- U: Noise (exogenous variable)
- f: Deterministic function
```

**Linear Functional Model:**

```
Y := β₀ + β₁X + U,  where E[U]=0, X ⊥ U

This ensures:
- Linear relationship
- No unmeasured confounding (X ⊥ U)
```

**Non-Linear Functional Model:**

```
Y := f(X) + U,  where f is non-linear

Example: Y := sin(X) + exp(X) + U
```

**Key Assumption: Causal Faithfulness**

```
The causal mechanism f must be such that:
Causal relationships in structure → Observed dependencies

Violation example: β₁ + β₂ = 0 would create spurious independence
```

### 2.5 LiNGAM (Linear Non-Gaussian Acyclic Model)

**Core Idea**: Use non-Gaussian noise to identify causal structure uniquely.

**Model:**

```
X := B X + E

Where:
- X: Vector of variables
- B: Causal coefficient matrix (lower triangular after permutation)
- E: Non-Gaussian noise vector, E_i ⊥ E_j for i ≠ j
```

**Key Insight:**

```
Standard Linear Model (Gaussian):
Y := β₁X₁ + β₂X₂ + ε, ε ~ N(0, σ²)

Cannot distinguish between:
- X₁ → Y, X₂ → Y
- X₂ → Y, X₁ → Y (reverse)

LiNGAM with Non-Gaussian Noise:
If ε is non-Gaussian (e.g., exponential, Laplace):
- Can uniquely identify causal direction
```

**Algorithm: LiNGAM Discovery**

```python
def LiNGAM(X, order_detection='auto'):
    # Step 1: ICA (Independent Component Analysis)
    # Estimate unmixed signals
    W = ICA(X)  # Unmixing matrix
    A = inv(W)  # Mixing matrix
    
    # Step 2: Causal order detection
    if order_detection == 'auto':
        # Detect ordering by residual variance
        ordering = detect_causal_order(W)
    else:
        ordering = order_detection
    
    # Step 3: Permute to get lower triangular form
    B = permute_to_triangular(A, ordering)
    
    # Step 4: Extract causal relationships
    causal_matrix = B
    
    return causal_matrix, ordering
```

**Advantages:**
- Unique identification under non-Gaussian assumption
- Works with acyclic structures
- Computationally efficient

**Limitations:**
- Requires acyclic structure
- Sensitive to non-Gaussianity assumption
- May fail with near-Gaussian noise

### 2.6 Temporal Causal Discovery

**Extension**: Discover causal relationships across time.

**Temporal DAG:**

```
Time t-2: X_{t-2} → Y_{t-2}
Time t-1: X_{t-1} → Y_{t-1}, X_{t-1} → Y_t
Time t:   X_t → Y_t

Lagged relationships: X_{t-τ} → Y_t for lag τ > 0
```

**Granger Causality** (Classical approach):

```
X Granger-causes Y if:
P(Y_t | Y_{t-1}, Y_{t-2}, ...) ≠ P(Y_t | Y_{t-1}, ..., X_{t-1}, X_{t-2}, ...)

In words: Past values of X help predict Y beyond Y's own past
```

**Mathematical Definition:**

```
Var(ε) < Var(ε*)

Where:
- ε: Residuals from AR model of Y
- ε*: Residuals from AR model of Y augmented with past X

Test using F-statistic or likelihood ratio
```

**Temporal PC Algorithm:**

```
Modified PC for time series:
1. Only consider edges X_t → Y_{t+τ} for τ ≥ 1
2. Cannot have edges Y_t → X_{t-τ} (respect time)
3. Contemporaneous edges X_t ↔ Y_t allowed (confounding)
```

**Application: Econometrics Example**

```
Interest Rate_{t-1} → Stock Price_t
Stock Price_{t-1} → Interest Rate_t (bidirectional)

Can be discovered from time series data
```

---

## Causal Effect Estimation

### 3.1 Types of Causal Effects

#### Average Treatment Effect (ATE)

```
ATE = E[Y(1) - Y(0)]

Where:
- Y(1): Potential outcome if treated
- Y(0): Potential outcome if control
```

**Estimation:**

```
Under randomization: ATE = E[Y|T=1] - E[Y|T=0]
```

#### Conditional Average Treatment Effect (CATE) / Heterogeneous Treatment Effect (HTE)

```
CATE(X) = E[Y(1) - Y(0) | X]

Effect depends on covariates X
```

**Example:**

```
Drug effectiveness varies by age:
CATE(age=30) = E[Y(1) - Y(0) | age=30] = 0.5
CATE(age=70) = E[Y(1) - Y(0) | age=70] = 0.1
```

### 3.2 Randomized Controlled Trials (RCT)

**Gold Standard for Causal Inference**

**Design:**

```
1. Random assignment: T ⊥ Y(0), Y(1)
2. Treatment group: receives intervention
3. Control group: receives placebo/no treatment
4. Measure outcomes: Y
```

**Causal Identification:**

```
Since T is randomized:
E[Y|do(T=1)] = E[Y|T=1]
E[Y|do(T=0)] = E[Y|T=0]

ATE = E[Y|T=1] - E[Y|T=0]
```

**Advantages:**
- Simple identification
- No confounding by design
- Causal estimates unbiased

**Disadvantages:**
- Often expensive and slow
- Ethical constraints
- Limited external validity (generalization)

**Statistical Test:**

```
H₀: ATE = 0
H₁: ATE ≠ 0

Test statistic: t = ATE / SE(ATE)

SE(ATE) = √(Var(Y|T=1)/n₁ + Var(Y|T=0)/n₀)
```

### 3.3 Propensity Score Matching

**Idea**: Create pseudo-randomized groups using propensity scores.

**Propensity Score:**

```
e(X) = P(T=1|X)

Probability of treatment given covariates
```

**Intuition:**

```
If two units have same propensity score e(X) = p:
- P(T=1|e=p) = p (by definition)
- T approximately independent of X given e
- Can compare treated and untreated units "like randomization"
```

**Algorithm:**

```python
def propensity_score_matching(X, T, Y, caliper=0.1):
    # Step 1: Estimate propensity scores
    e = estimate_propensity_score(X, T)  # Often logistic regression
    
    # Step 2: Match treated to control
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    
    matches = {}
    for i in treated_idx:
        # Find nearest control unit by propensity score
        distances = np.abs(e[i] - e[control_idx])
        
        if min(distances) < caliper:  # Caliper: max allowed distance
            j = control_idx[np.argmin(distances)]
            matches[i] = j
    
    # Step 3: Estimate ATE on matched sample
    matched_treated = [Y[i] for i in matches.keys()]
    matched_control = [Y[j] for j in matches.values()]
    
    ATE = np.mean(matched_treated) - np.mean(matched_control)
    
    return ATE, matches
```

**Mathematical Justification:**

```
Under assumptions:
1. Unconfoundedness: T ⊥ Y(0), Y(1) | X
2. Overlap: 0 < e(X) < 1 for all X

Result (Rosenbaum & Rubin 1983):
T ⊥ Y(0), Y(1) | e(X)

Matching on propensity score is sufficient
```

**Advantages:**
- Reduces dimensions (score is 1-D vs many covariates)
- Intuitive to implement
- Works well with adequate overlap

**Disadvantages:**
- Sensitive to propensity score estimation
- May have poor matches (common support)
- Bias if unobserved confounders exist

### 3.4 Inverse Probability Weighting (IPW)

**Idea**: Weight observations by inverse of propensity score to create pseudo-population.

**Mathematical Intuition:**

```
Units with low propensity to be treated but are treated:
- Unusual, informative about treatment effect
- Upweight them (weight = 1/e(X) is large)

Units with high propensity to be treated and are treated:
- Common, less informative
- Downweight them (weight = 1/e(X) is small)
```

**Weighting Scheme:**

```
For treatment group (T=1):
Weight_i = 1 / e(X_i)

For control group (T=0):
Weight_i = 1 / (1 - e(X_i))

Estimates are unbiased causal effects
```

**Estimator:**

```
ATE = E[T*Y / e(X)] - E[(1-T)*Y / (1-e(X))]

Or more explicitly:

ATE = [∑(T_i * Y_i / e_i)] / [∑(T_i / e_i)] 
      - [∑((1-T_i)*Y_i / (1-e_i))] / [∑((1-T_i) / (1-e_i))]
```

**Algorithm:**

```python
def inverse_probability_weighting(X, T, Y):
    # Step 1: Estimate propensity scores
    e = estimate_propensity_score(X, T)
    
    # Step 2: Calculate weights
    weights = np.where(T == 1, 1/e, 1/(1-e))
    
    # Step 3: Weighted estimator
    treated_idx = T == 1
    
    weighted_treated_mean = np.sum(Y[treated_idx] * weights[treated_idx]) / \
                           np.sum(weights[treated_idx])
    
    weighted_control_mean = np.sum(Y[~treated_idx] * weights[~treated_idx]) / \
                          np.sum(weights[~treated_idx])
    
    ATE = weighted_treated_mean - weighted_control_mean
    
    return ATE
```

**Theoretical Properties:**

```
Under assumptions:
1. Unconfoundedness: T ⊥ Y(0), Y(1) | X
2. Overlap: 0 < e(X) < 1

IPW estimator is:
- Unbiased: E[ATE_IPW] = ATE
- √n-consistent: √n(ATE_IPW - ATE) → N(0, σ²)
- Asymptotically normal for hypothesis testing
```

**Variance Estimation:**

```
Var(ATE_IPW) ≈ 1/n * E[(T*Y/e(X) - (1-T)*Y/(1-e(X)) - ATE)²]

Can be estimated using bootstrap or closed-form formula
```

**Robustness to Propensity Score Misspecification:**

```
If propensity score model is misspecified:
- IPW can be highly biased
- Very sensitive to overlap violations
- Large weights cause high variance
```

### 3.5 Doubly Robust Estimators

**Idea**: Combine regression and propensity score methods for protection against misspecification.

**Property**: If EITHER the propensity score OR regression model is correct, estimate is unbiased.

**Doubly Robust Estimator Formula:**

```
DR = 1/n * ∑[T_i*Y_i/e_i - (T_i - e_i)*m₁(X_i)/e_i] 
   - 1/n * ∑[(1-T_i)*Y_i/(1-e_i) + (T_i - e_i)*m₀(X_i)/(1-e_i)]

Where:
- e_i = propensity score estimate
- m₁(X_i) = outcome model E[Y|T=1, X_i]
- m₀(X_i) = outcome model E[Y|T=0, X_i]
```

**Intuition:**

```
Component 1: IPW estimate
Component 2: Adjustment for regression residuals
  - Reduces bias from propensity score errors
  - Uses outcome model as "doubly robust" backup
```

**Algorithm:**

```python
def doubly_robust_estimator(X, T, Y):
    # Step 1: Estimate propensity scores
    e = estimate_propensity_score(X, T)
    
    # Step 2: Estimate outcome models
    m1 = fit_regression_model(X[T==1], Y[T==1])  # E[Y|T=1, X]
    m0 = fit_regression_model(X[T==0], Y[T==0])  # E[Y|T=0, X]
    
    # Step 3: Calculate predictions
    y1_pred = m1.predict(X)
    y0_pred = m0.predict(X)
    
    # Step 4: Doubly robust estimation
    dr_treated = (T*Y/e) - ((T-e)/e)*y1_pred
    dr_control = ((1-T)*Y/(1-e)) + ((T-e)/(1-e))*y0_pred
    
    ATE_DR = np.mean(dr_treated) - np.mean(dr_control)
    
    return ATE_DR
```

**Theoretical Guarantees:**

```
Assume:
- Unconfoundedness
- Overlap
- Either propensity score or outcome model is correct

Then:
- E[ATE_DR] = ATE (unbiased)
- √n-consistent
- Asymptotically normal

**Semiparametric Efficiency:**
The doubly robust estimator achieves the semiparametric 
efficiency bound - it has minimum asymptotic variance 
among all unbiased estimators
```

**Variance and Confidence Intervals:**

```python
def dr_variance_bootstrap(X, T, Y, n_bootstrap=1000):
    ate_bootstrap = []
    n = len(X)
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        ate_b = doubly_robust_estimator(X[idx], T[idx], Y[idx])
        ate_bootstrap.append(ate_b)
    
    se = np.std(ate_bootstrap)
    ci_lower = np.percentile(ate_bootstrap, 2.5)
    ci_upper = np.percentile(ate_bootstrap, 97.5)
    
    return se, (ci_lower, ci_upper)
```

### 3.6 Heterogeneous Treatment Effects (HTE)

**Definition**: Treatment effect varies by individual characteristics.

#### CATE (Conditional Average Treatment Effect)

```
τ(x) = E[Y(1) - Y(0) | X = x]

The causal effect conditional on X
```

**Example: Medical Treatment**

```
Drug effectiveness:
- CATE(age < 40) = 0.8 (high effect)
- CATE(40 ≤ age < 60) = 0.5 (medium effect)
- CATE(age ≥ 60) = 0.1 (low effect)

Personalized medicine: treat based on CATE
```

#### Estimation Methods

**Method 1: Stratification by Covariates**

```
1. Partition X into strata (e.g., age groups)
2. Estimate ATE within each stratum
3. Stratum-specific estimates = CATE estimates
```

**Limitations:**
- Curse of dimensionality
- Small sample sizes in strata
- Loses continuous structure

**Method 2: Causal Trees**

```
Generalization of decision trees for causal effects.

Idea: Recursively partition X space to maximize 
heterogeneity in treatment effects

Split criterion: Maximize difference in 
(mean(Y|T=1) - mean(Y|T=0)) between left/right children
```

**Algorithm Sketch:**

```python
def causal_tree(X, T, Y, depth=0, max_depth=5):
    if depth >= max_depth or len(X) < min_samples:
        # Leaf node: estimate CATE
        cate = np.mean(Y[T==1]) - np.mean(Y[T==0])
        return LeafNode(cate)
    
    best_split = None
    best_gain = 0
    
    # Try all possible splits
    for feature in range(X.shape[1]):
        for threshold in X[:, feature]:
            # Split data
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            
            if len(X[left_mask]) == 0 or len(X[right_mask]) == 0:
                continue
            
            # Calculate heterogeneity gain
            left_effect = np.mean(Y[left_mask & (T==1)]) - \
                         np.mean(Y[left_mask & (T==0)])
            right_effect = np.mean(Y[right_mask & (T==1)]) - \
                          np.mean(Y[right_mask & (T==0)])
            
            gain = abs(left_effect - right_effect)
            
            if gain > best_gain:
                best_gain = gain
                best_split = (feature, threshold)
    
    if best_split is None:
        return LeafNode(np.mean(Y[T==1]) - np.mean(Y[T==0]))
    
    # Recursively build left and right subtrees
    feature, threshold = best_split
    left_tree = causal_tree(X[X[:,feature] <= threshold], 
                           T[X[:,feature] <= threshold],
                           Y[X[:,feature] <= threshold],
                           depth+1, max_depth)
    
    right_tree = causal_tree(X[X[:,feature] > threshold],
                            T[X[:,feature] > threshold],
                            Y[X[:,feature] > threshold],
                            depth+1, max_depth)
    
    return InternalNode(feature, threshold, left_tree, right_tree)
```

**Method 3: Causal Forests**

```
Ensemble of causal trees with random splitting.

Key improvements over single causal tree:
1. Reduces variance through averaging
2. More stable estimates
3. Asymptotic normality for inference
```

**Algorithm:**

```
1. Generate B bootstrap samples
2. For each sample:
   - Randomly select subset of features for splits
   - Grow causal tree to fixed depth
3. Average predictions across trees: 
   τ̂(x) = 1/B ∑_b τ̂_b(x)
```

**Theoretical Properties:**

```
Under regularity conditions:
√n(τ̂(X) - τ(X)) → N(0, σ²(X))

Where σ²(X) can be estimated for confidence intervals
```

**Code Example:**

```python
from causalml.inference.tree_based import CausalTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Single causal tree
ct = CausalTreeRegressor()
ct.fit(X, T, Y)
cate_pred = ct.predict(X_test)

# Causal forest
cf = CausalForestRegressor(n_trees=100, 
                           max_depth=10,
                           min_samples_leaf=5)
cf.fit(X, T, Y)
cate_pred = cf.predict(X_test)
```

---

## Advanced Methods

### 4.1 Instrumental Variables

**Problem**: Unobserved confounder prevents identification.

```
DAG: U → T → Y, U → Y

Unobserved U confounds T and Y
Back-door criterion cannot be satisfied
```

**Solution**: Find an Instrument Z.

**Instrumental Variable Requirements:**

```
1. Relevance: Z affects T
   Z is not independent of T given X
   
2. Exogeneity: Z does not affect Y except through T
   Z is independent of U (unobserved confounder)
   No direct Z → Y path
```

**Causal DAG with Instrument:**

```
       U
      / \
     v   v
Z → T → Y

Z is exogenous (no arrow into Z)
Z → Y only through T
```

**Two-Stage Least Squares (2SLS)**

**Stage 1**: Estimate T from Z

```
T = γ₀ + γ₁Z + γ₂X + ν
  T̂ = γ̂₀ + γ̂₁Z + γ̂₂X
```

**Stage 2**: Estimate Y using predicted T

```
Y = β₀ + β₁T̂ + β₂X + ε

β̂₁ = Causal effect of T on Y (IV estimator)
```

**Formal 2SLS Formula:**

```
β̂_2SLS = (X'P_z X)⁻¹ X'P_z Y

Where:
- P_z = Z(Z'Z)⁻¹Z' (projection onto Z)
- Isolates exogenous variation in T
```

**IV Estimator for Single Instrument:**

```
τ = Cov(Z, Y) / Cov(Z, T)

Intuition: 
Effect on Y per unit effect on T
Both mediated by exogenous instrument Z
```

**Code Example:**

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def two_stage_ls(Z, T, Y, X=None):
    # Combine Z with X for instruments
    if X is not None:
        Z_full = np.column_stack([Z, X])
    else:
        Z_full = Z.reshape(-1, 1)
    
    # Stage 1: Predict T from Z
    lr1 = LinearRegression()
    lr1.fit(Z_full, T)
    T_pred = lr1.predict(Z_full)
    
    # Stage 2: Predict Y from predicted T
    lr2 = LinearRegression()
    if X is not None:
        T_X = np.column_stack([T_pred, X])
    else:
        T_X = T_pred.reshape(-1, 1)
    
    lr2.fit(T_X, Y)
    
    # Extract causal effect of T on Y
    tau_iv = lr2.coef_[0]  # First coefficient is T's effect
    
    return tau_iv
```

**Testing IV Assumptions**

**Weak Instrument Test** (First-stage F-statistic):

```
In stage 1 regression: T = γ₀ + γ₁Z + ν

H₀: γ₁ = 0 (instrument is weak)
H₁: γ₁ ≠ 0 (instrument is strong)

First-stage F-statistic: 
F = (SS_Z / SS_residual) * (n - k - 1) / k

Rule of thumb: F > 10 indicates strong instrument
```

**Endogeneity Test** (Durbin-Wu-Hausman):

```
Tests whether OLS and IV estimates differ significantly.

If they're similar: May not need IV (no endogeneity)
If they differ: IV needed for consistency
```

### 4.2 Regression Discontinuity

**Setting**: Treatment assigned based on threshold of running variable.

```
DAG: Running_Variable → Treatment (sharp cutoff)
     Running_Variable → Outcome
     Treatment → Outcome
```

**Example: Education Policy**

```
School eligibility: determined by birth date
- Cutoff: January 1, 1990
- Students born ≤ Jan 1, 1990: must attend school
- Students born > Jan 1, 1990: can defer

Running variable: Birth date
Treatment threshold: January 1, 1990
Outcome: Educational attainment
```

**Sharp vs Fuzzy Regression Discontinuity**

**Sharp RD:**
```
P(T=1 | Running_Var = c⁻) = 1
P(T=1 | Running_Var = c⁺) = 0

At the cutoff c, treatment probability jumps from 1 to 0
```

**Fuzzy RD:**
```
0 < P(T=1 | Running_Var = c⁻) < 1
0 < P(T=1 | Running_Var = c⁺) < 1

Treatment probability increases at cutoff but not deterministic
```

**Sharp RD Estimation**

**Local Polynomial Regression:**

```
Fit polynomial on each side of cutoff
Estimate discontinuity at cutoff

Y = α + β₀(Running_Var - c) + β₁(Running_Var - c)₊ + ε

Where (x)₊ = max(x, 0)

Treatment effect at cutoff: β₁
```

**Algorithm:**

```python
def sharp_regression_discontinuity(X, Y, cutoff, bandwidth='auto'):
    """
    X: Running variable
    Y: Outcome
    cutoff: Threshold value
    """
    
    # Select observations within bandwidth
    if bandwidth == 'auto':
        bandwidth = optimal_bandwidth(X, Y, cutoff)
    
    in_bandwidth = np.abs(X - cutoff) <= bandwidth
    X_in = X[in_bandwidth]
    Y_in = Y[in_bandwidth]
    
    # Create treatment indicator and centered running variable
    T = (X_in >= cutoff).astype(int)
    X_centered = X_in - cutoff
    
    # Fit polynomial regression
    X_poly = np.column_stack([X_centered, X_centered * T])  # Linear with kink
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_poly, Y_in)
    
    # Treatment effect is the kink coefficient
    tau_rd = lr.coef_[1]
    
    return tau_rd
```

**Fuzzy RD as IV**

When treatment is fuzzy, use instrumental variables:
- Instrument: Just crossed cutoff (indicator)
- Treatment: Actual treatment (may not follow cutoff)

```
First stage: T = α + γ(X ≥ cutoff) + β(X - c) + ε
Second stage: Y = constant + τ*T̂ + β(X - c) + ε

τ_Fuzzy_RD = Reduced form effect / First stage effect
           = Effect on Y / Effect on T
```

**Validity Assumptions:**

```
1. Continuity at cutoff:
   lim_{X→c⁻} E[Y(0)|X] = lim_{X→c⁺} E[Y(0)|X]
   
   (Potential outcomes continuous at threshold)

2. No manipulation:
   Agents cannot precisely manipulate running variable
   (No bunching at cutoff)

3. No other interventions at cutoff
```

### 4.3 Synthetic Control Method

**Setting**: Panel data with multiple units and time periods.

**Problem**: Estimate treatment effect for one unit (treated unit) using others (donors).

**Example: Impact of Policy Change**

```
Germany reunified in 1990.
Question: What would West Germany look like without reunification?

Use trends from similar countries (synthetic control) 
as counterfactual
```

**Method:**

```
1. Identify treated unit and pool of donor units
2. Create weighted combination of donors
   - Weights should match pre-treatment trends
3. Compare treated unit to synthetic control after treatment
```

**Formal Framework:**

**Pre-treatment fit (find weights):**

```
Minimize: ∑_{t=1}^{T₀} (Y₁ₜ - ∑_j w_j Y_{jt})²

Where:
- Y₁ₜ: Treated unit outcome at time t
- Y_{jt}: Donor unit j outcome at time t
- w_j: Weights (∑w_j = 1, w_j ≥ 0)
- T₀: Pre-treatment period

Optimal weights make synthetic control match treated unit 
in pre-treatment
```

**Post-treatment effect:**

```
τ_t = Y₁ₜ - Y*_{1t}

Where Y*_{1t} = ∑_j w_j Y_{jt} (synthetic control)

Treatment effect = gap between treated and synthetic control
```

**Algorithm:**

```python
def synthetic_control(Y_treated, Y_donors_pre, Y_donors_post):
    """
    Y_treated: (n_pre + n_post,) array
    Y_donors_pre: (n_donors, n_pre) array
    Y_donors_post: (n_donors, n_post) array
    
    Returns: Estimated treatment effect after treatment
    """
    
    from scipy.optimize import minimize
    
    n_pre = Y_treated[:len(Y_donors_pre[0])].shape[0]
    Y_treated_pre = Y_treated[:n_pre]
    Y_treated_post = Y_treated[n_pre:]
    
    # Objective: minimize pre-treatment fit
    def objective(w):
        synthetic_pre = np.dot(w, Y_donors_pre)
        return np.sum((Y_treated_pre - synthetic_pre)**2)
    
    # Constraints: weights sum to 1, non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(Y_donors_pre.shape[0])]
    
    # Optimize
    result = minimize(objective, x0=np.ones(Y_donors_pre.shape[0])/Y_donors_pre.shape[0],
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    
    # Create synthetic control for post-treatment period
    synthetic_post = np.dot(optimal_weights, Y_donors_post)
    
    # Treatment effect
    treatment_effect = Y_treated_post - synthetic_post
    
    return treatment_effect, optimal_weights
```

**Validity Assumptions:**

```
1. No interference: Treatment on treated unit 
   doesn't affect donors

2. No anticipation: Treated unit doesn't change behavior
   in anticipation of treatment

3. Constant treatment effect: Effect same across time
   (or can be modified for time-varying effects)

4. Good pre-treatment fit: Synthetic control matches 
   treated unit before treatment
```

### 4.4 Difference-in-Differences

**Setting**: Panel data with treatment at specific time.

**Key Idea**: Compare treatment and control groups before and after treatment.

```
       Before T=0  After T=1
Control    Y₀         Y₀'
Treated    Y₁         Y₁'

DD effect = (Y₁' - Y₁) - (Y₀' - Y₀)
```

**Mathematical Framework:**

```
Y_{it} = α + β*Group_i + γ*Time_t + δ(Group_i × Time_t) + ε_{it}

Where:
- i: Unit (control=0, treatment=1)
- t: Time (before=0, after=1)
- Group_i × Time_t: Interaction (=1 only for treated after)
- δ: Difference-in-differences estimator
```

**Estimation:**

```
DD = E[Y₁ₜ=₁] - E[Y₁ₜ=₀] - (E[Y₀ₜ=₁] - E[Y₀ₜ=₀])
    = (Y₁₁ - Y₁₀) - (Y₀₁ - Y₀₀)
```

**Example: Minimum Wage Study**

```
Question: Does minimum wage increase reduce employment?

Treatment: New Jersey raises minimum wage (treatment)
Control: Pennsylvania keeps old wage (control)

DD = (NJ_after - NJ_before) - (PA_after - PA_before)

If DD < 0: Negative employment effect
If DD > 0: Positive employment effect
If DD ≈ 0: No employment effect
```

**Code:**

```python
def difference_in_differences(data, group_col, time_col, outcome_col):
    """
    data: DataFrame with group, time, and outcome
    group_col: Column indicating treatment group
    time_col: Column indicating time period
    outcome_col: Column of outcome variable
    """
    
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    
    # Create dummy variables
    X = pd.get_dummies(data[[group_col, time_col]], drop_first=True)
    X['interaction'] = data[group_col] * data[time_col]
    
    # Add constant
    X = pd.concat([pd.Series(1, index=X.index), X], axis=1)
    X.columns = ['const', 'group', 'time', 'interaction']
    
    y = data[outcome_col]
    
    # Estimate
    lr = LinearRegression()
    lr.fit(X, y)
    
    dd_effect = lr.coef_[3]  # Interaction coefficient
    
    return dd_effect
```

**Validity Assumptions:**

```
1. Parallel trends:
   Treated and control would follow same trend without treatment
   
   Check: Compare pre-treatment trends
   
   Visual test:
   - Plot both groups before and after
   - Pre-treatment trends should be parallel

2. No time-varying confounders
   affecting treatment and outcome

3. No feedback from outcome to treatment
   (no anticipation effects)

4. Stable unit treatment value assumption (SUTVA)
   Treatment of one unit doesn't affect others
```

**Multiple Time Periods (Event Study)**

```
Generalize to many time periods:

Y_{it} = α_i + λ_t + ∑_{τ≠-1} β_τ D_{it}^τ + ε_{it}

Where:
- α_i: Unit fixed effects
- λ_t: Time fixed effects
- D_{it}^τ: Treatment indicator for time τ relative to treatment
- β_τ: Effect τ periods after treatment

Plot β_τ over time to see treatment effect trajectory
```

**Code Example:**

```python
def event_study(data, unit_col, time_col, outcome_col, 
                treatment_time_col, relative_time_range=(-5, 5)):
    
    import pandas as pd
    import numpy as np
    
    # Create relative time variable
    data['relative_time'] = data[time_col] - data[treatment_time_col]
    
    # Filter to time range
    min_rel, max_rel = relative_time_range
    data_filtered = data[(data['relative_time'] >= min_rel) & 
                        (data['relative_time'] <= max_rel)]
    
    # Create dummy variables for each relative time
    dummies = pd.get_dummies(data_filtered['relative_time'], 
                             prefix='time', drop_first=True)
    
    # Add unit and time fixed effects
    unit_dummies = pd.get_dummies(data_filtered[unit_col], 
                                  prefix='unit', drop_first=False)
    time_dummies = pd.get_dummies(data_filtered[time_col], 
                                  prefix='year', drop_first=False)
    
    X = pd.concat([dummies, unit_dummies, time_dummies], axis=1)
    y = data_filtered[outcome_col]
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Extract coefficients for relative times
    n_times = len(dummies.columns)
    effects = lr.coef_[:n_times]
    
    return effects
```

### 4.5 Causal Forests

**Extension of Causal Trees to ensemble method.**

**Key Ideas:**

```
1. Build many trees with randomization
2. Average predictions for stability
3. Use asymptotically normal estimates for inference
```

**Algorithm (Athey & Wager, 2019):**

```
1. For each tree b = 1, ..., B:
   a. Sample with replacement (bootstrap)
   b. Randomly select features for each split
   c. Grow tree by minimizing heterogeneity criterion
   d. Predict CATE at test point using this tree

2. Ensemble prediction:
   τ̂(x) = 1/B ∑_b τ̂_b(x)

3. Variance estimate for confidence intervals:
   σ̂²(x) = 1/B ∑_b (τ̂_b(x) - τ̂(x))²
```

**Mathematical Foundation:**

```
Asymptotic normality:
√n (τ̂(x) - τ(x)) →^d N(0, σ²(x))

Where σ²(x) can be consistently estimated

Allows:
- Hypothesis tests H₀: τ(x) = 0
- Confidence intervals: τ̂(x) ± 1.96 * σ̂(x)/√n
- Prediction intervals
```

**Feature Importance from Causal Forests:**

```
Measure how much each feature matters for heterogeneity.

Importance = Decrease in heterogeneity when feature is used for splits
          across all trees

High importance: Feature strongly modifies treatment effect
Low importance: Effect doesn't depend on this feature
```

**Code Example:**

```python
from causalml.inference.tree_based import CausalForestRegressor

# Initialize causal forest
cf = CausalForestRegressor(
    n_trees=100,
    max_depth=25,
    min_samples_leaf=5,
    n_jobs=-1
)

# Fit on training data
cf.fit(X_train, T_train, Y_train)

# Predict CATE on test data
cate_predictions = cf.predict(X_test)

# Get variance for confidence intervals
cate_variance = cf.predict_variance(X_test)
cate_se = np.sqrt(cate_variance)

# Construct 95% CI
ci_lower = cate_predictions - 1.96 * cate_se
ci_upper = cate_predictions + 1.96 * cate_se

# Feature importance
feature_importance = cf.feature_importance()
```

---

## Applications & Tools

### 5.1 Causal Inference Libraries

#### DoWhy Library

**Purpose**: Transparent, modular causal inference in Python.

**Installation:**
```bash
pip install dowhy
```

**Basic Workflow:**

```python
from dowhy import CausalModel

# Step 1: Create causal model
model = CausalModel(
    data=df,
    treatment="T",
    outcome="Y",
    common_causes=["X1", "X2"],
    instruments=["Z"],
    graph="graph.txt"  # Optional: DAG specification
)

# Step 2: Identify causal effect
identified_estimand = model.identify_effect()

# Step 3: Estimate causal effect
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)

# Step 4: Refute (test robustness)
refute_results = model.refute_estimate(
    identified_estimand, estimate,
    method_name="random_common_cause"
)

print(estimate)
```

**Estimation Methods:**

```
Backdoor:
- backdoor.propensity_score_matching
- backdoor.propensity_score_stratification
- backdoor.linear_regression
- backdoor.propensity_score_weighting

Instrumental Variables:
- iv.instrumental_variable
- iv.wald

Front-door:
- frontdoor.linear_regression
```

**Refutation (Robustness Check):**

```python
# Add random confounder
refute_results = model.refute_estimate(
    identified_estimand, estimate,
    method_name="add_random_common_cause",
    confound_score=0.1
)

# Random placebo treatment
refute_results = model.refute_estimate(
    identified_estimand, estimate,
    method_name="placebo_treatment_refuter"
)

# Subset validation
refute_results = model.refute_estimate(
    identified_estimand, estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.8
)
```

#### CausalML Library

**Purpose**: Machine learning for heterogeneous treatment effect estimation.

**Installation:**
```bash
pip install causalml
```

**Workflows:**

```python
from causalml.inference.meta import BaseXRegressor, BaseRRegressor
from causalml.inference.tree_based import CausalForestRegressor

# Method 1: X-Learner (recommended)
x_learner = BaseXRegressor(
    base_model=RandomForestRegressor(),
    cv=5
)
cate_x = x_learner.fit_predict(X, T, Y)

# Method 2: S-Learner
from causalml.inference.meta import BaseSLearner
s_learner = BaseSLearner(base_model=RandomForestRegressor())
cate_s = s_learner.fit_predict(X, T, Y)

# Method 3: T-Learner
from causalml.inference.meta import BaseTLearner
t_learner = BaseTLearner(base_model=GradientBoostingRegressor())
cate_t = t_learner.fit_predict(X, T, Y)

# Method 4: Causal Forest
cf = CausalForestRegressor(n_trees=100)
cf.fit(X, T, Y)
cate_cf = cf.predict(X)

# Method 5: DR-Learner (Doubly Robust)
from causalml.inference.meta import DRLearner
dr_learner = DRLearner(
    outcome_learner=RandomForestRegressor(),
    propensity_learner=LogisticRegression()
)
cate_dr = dr_learner.fit_predict(X, T, Y)
```

**Tree-based Methods Comparison:**

```
S-Learner:
- Single model trained on combined data
- Fast, simple
- May mix treatment and control bias

T-Learner:
- Separate models for treatment/control
- Better interpretation
- May have data split inefficiency

X-Learner:
- Asymmetric fitting
- Good for imbalanced treatment
- Theoretically optimal under conditions

R-Learner (Robinson):
- Uses residualization
- Robust to model misspecification
- Better for high dimensions

DR-Learner (Doubly Robust):
- Combines propensity and outcome models
- Robust to either model misspecification
- Best theoretical properties
```

### 5.2 Real-World Case Studies

#### Case 1: E-Commerce Recommendation System

**Problem**: Estimate effect of recommended products on purchase.

**Challenges:**
- Recommendations selected based on user history (confounding)
- Different users see different recommendations
- Treatment (recommendation) not random

**Approach:**

```
1. Data: User history, recommendations, purchases
2. Propensity score: P(recommend = 1 | user_history)
3. Method: Doubly robust estimation
4. Outcome: Average treatment effect on converted users
```

**Python Implementation:**

```python
import pandas as pd
import numpy as np
from causalml.inference.meta import DRLearner
from sklearn.ensemble import RandomForestRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('ecommerce_data.csv')

# Features: user browsing history, device, etc.
features = ['view_count', 'avg_price', 'device_mobile', 
            'time_on_site', 'previous_purchases', 'account_age']
X = data[features]

# Treatment: was product recommended
T = data['recommended'].astype(int)

# Outcome: did user purchase
Y = data['purchased'].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Estimate heterogeneous treatment effects
dr_learner = DRLearner(
    outcome_learner=RandomForestRegressor(n_estimators=100, max_depth=10),
    propensity_learner=LogisticRegression(max_iter=1000)
)

# Fit and predict
cate = dr_learner.fit_predict(X_scaled, T, Y)

# Analyze results
results_df = data.copy()
results_df['estimated_effect'] = cate

# Segment users by effect
results_df['effect_high'] = cate > np.percentile(cate, 75)
results_df['effect_low'] = cate < np.percentile(cate, 25)

print(f"Average treatment effect: {cate.mean():.4f}")
print(f"High effect users: {results_df['effect_high'].sum()}")
print(f"Low effect users: {results_df['effect_low'].sum()}")

# Business decision: Allocate recommendations to high-effect users
high_effect_users = results_df[results_df['effect_high']]['user_id'].tolist()
```

#### Case 2: Healthcare Treatment Analysis

**Problem**: Estimate treatment effect of medication on patient outcomes.

**Challenges:**
- Observational data (not randomized)
- Patients self-selected treatment
- Multiple confounders (age, comorbidities, etc.)

**Approach:**

```
1. Propensity score matching
2. Doubly robust estimation
3. Heterogeneous effects by age/severity
```

**Python Implementation:**

```python
import pandas as pd
import numpy as np
from causalml.inference.meta import BaseXRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load healthcare data
df = pd.read_csv('patient_outcomes.csv')

# Features: age, comorbidities, vitals
features = ['age', 'weight', 'systolic_bp', 'diastolic_bp', 
            'heart_rate', 'diabetes', 'hypertension', 'asthma']
X = df[features]

# Treatment: received new medication
T = df['medication'].astype(int)

# Outcome: improvement score (0-100)
Y = df['improvement_score']

# Estimate CATE
x_learner = BaseXRegressor(
    base_model=RandomForestRegressor(n_estimators=100, max_depth=15)
)
cate = x_learner.fit_predict(X, T, Y)

# Segment patients
df['estimated_effect'] = cate
df['responder'] = cate > np.percentile(cate, 75)

# Analyze by age group
age_groups = pd.cut(df['age'], bins=[0, 30, 50, 70, 100])
by_age = df.groupby(age_groups).agg({
    'estimated_effect': ['mean', 'std'],
    'responder': 'sum'
})

print("Treatment effect by age:")
print(by_age)

# Clinical decision: Higher-dose treatment for responders
responder_ids = df[df['responder']]['patient_id'].tolist()
print(f"\nIdentified {len(responder_ids)} potential responders")
```

#### Case 3: Marketing Campaign ROI

**Problem**: Measure return on investment of marketing intervention.

**Design**: A/B test with propensity matching for observational data.

```python
import pandas as pd
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.generalized_estimating_equations import Poisson
from statsmodels.genmod.cov_struct import Exchangeable

# Campaign data
df = pd.read_csv('campaign_data.csv')

# Features: customer attributes
features = ['customer_age', 'lifetime_value', 'previous_purchases',
            'email_open_rate', 'account_tenure']
X = df[features]

# Treatment: received marketing email
T = df['email_sent'].astype(int)

# Outcome: purchase amount
Y = df['purchase_amount']

# Method: IPW with overlap checking
from causalml.inference.meta import DRLearner

# Estimate propensity scores
from sklearn.linear_model import LogisticRegression
ps_model = LogisticRegression()
ps_model.fit(X, T)
propensity_scores = ps_model.predict_proba(X)[:, 1]

# Check overlap
print(f"Propensity score range: [{propensity_scores.min():.3f}, {propensity_scores.max():.3f}]")
if propensity_scores.min() < 0.1 or propensity_scores.max() > 0.9:
    print("Warning: Poor overlap detected")

# Estimate effect
dr_learner = DRLearner(
    outcome_learner=RandomForestRegressor(n_estimators=100),
    propensity_learner=LogisticRegression()
)
cate = dr_learner.fit_predict(X, T, Y)

# ROI calculation
avg_effect = np.mean(cate[T == 1])  # Effect on treated
campaign_cost = 10  # Cost per email
roi = (avg_effect - campaign_cost) / campaign_cost

print(f"\nAverage effect of email: ${avg_effect:.2f}")
print(f"Campaign cost per email: ${campaign_cost:.2f}")
print(f"ROI: {roi:.1%}")
```

### 5.3 Benchmark Datasets

#### IHDP Dataset (Infant Health Development Program)

**Size**: 747 units, 6 continuous + 19 binary covariates
**Treatment**: Participation in educational program
**Outcome**: IQ test score
**Use**: Benchmarking CATE estimation methods

```python
# Download IHDP data
from sklearn.datasets import fetch_openml

# IHDP preprocessing
def load_ihdp_data(version=1, train=True):
    """
    Load IHDP dataset
    version: 1-10 (10 runs)
    train: True for training, False for test
    """
    url = f"https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{version}.csv"
    data = pd.read_csv(url)
    
    if train:
        return data
    # Test set would be separate file
```

#### LaLonde Dataset

**Size**: 2675 units (614 treated, 2061 control)
**Treatment**: Job training program
**Outcomes**: Earnings 1978
**Covariates**: Demographics, education, employment history
**Classic dataset for causal inference research**

```python
import pandas as pd

# LaLonde NSW data
url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/Ecdat/Ltireland.csv"
data = pd.read_csv(url)

# NSW treatment assignment
treatment = (data['nsw'] == 1).astype(int)
outcome = data['re78']  # 1978 earnings

# Typical analysis: estimate treatment effect of job training
```

#### ACIC Benchmarks

**Size**: Variable (simulated for control)
**Treatment**: Binary
**Outcomes**: Continuous or binary
**Covariates**: 58 pre-treatment variables
**10 datasets × 100 runs = 1000 realistic scenarios**

```python
# Load from ACIC benchmark repository
# Contains pre-specified ground truth causal effects
# Useful for validating CATE estimation accuracy
```

### 5.4 Evaluation Metrics for CATE Estimation

**Mean Absolute Error:**
```python
def mae(true_cate, pred_cate):
    return np.mean(np.abs(true_cate - pred_cate))
```

**Mean Squared Error:**
```python
def mse(true_cate, pred_cate):
    return np.mean((true_cate - pred_cate)**2)
```

**Accuracy of Effect Direction:**
```python
def effect_direction_accuracy(true_cate, pred_cate):
    true_sign = np.sign(true_cate)
    pred_sign = np.sign(pred_cate)
    return np.mean(true_sign == pred_sign)
```

**Bias-Variance Tradeoff:**
```python
def evaluate_cate(true_cate, pred_cate):
    bias = np.mean(pred_cate - true_cate)
    variance = np.var(pred_cate - true_cate)
    mse = np.mean((pred_cate - true_cate)**2)
    return {
        'bias': bias,
        'variance': variance,
        'mse': mse
    }
```

---

## References

### Foundational Papers

1. **Pearl, J. (2009)**. *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
   - Foundational work on causal calculus and do-notation
   - Covers DAGs, d-separation, backdoor/front-door criteria

2. **Rubin, D. B. (1974)**. "Estimating causal effects of treatments in randomized and nonrandomized studies." *Journal of Educational Psychology*, 66(5), 688-701.
   - Introduces Potential Outcomes framework
   - SUTVA assumption

3. **Rosenbaum, P. R., & Rubin, D. B. (1983)**. "The central role of the propensity score in observational studies for causal effects." *Biometrika*, 70(1), 41-55.
   - Propensity score theory
   - Matching and stratification

4. **Angrist, J. D., & Pischke, J. S. (2008)**. *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.
   - Practical causal inference methods
   - IV, RDD, D-in-D

5. **Athey, S., & Wager, S. (2019)**. "Estimating treatment effects with causal forests." *Journal of the American Statistical Association*, 113(523), 1228-1242.
   - Causal forests for heterogeneous effects
   - Asymptotic normality

6. **Kennedy, E. H. (2020)**. "Optimal doubly robust nonparametric inference." *arXiv preprint arXiv:2004.14497*.
   - Doubly robust estimation
   - Semiparametric efficiency

7. **Imbens, G. W., & Angrist, J. D. (1994)**. "Identification and estimation of local average treatment effects." *Econometrica*, 62(2), 467-475.
   - Instrumental variables
   - LATE framework

8. **Abadie, A., Diamond, A., & Hainmueller, J. (2010)**. "Synthetic control methods for comparative case studies." *Journal of the American Statistical Association*, 105(490), 493-505.
   - Synthetic control method
   - Policy evaluation

9. **Shimizu, S., Hoyer, P. O., Hyvärinen, A., & Kerminen, A. (2006)**. "A linear non-Gaussian acyclic model for causal discovery." *Journal of Machine Learning Research*, 7, 2003-2030.
   - LiNGAM algorithm
   - Causal discovery

10. **Spirtes, P., Glymour, C., & Scheines, R. (2000)**. *Causation, Prediction, and Search* (2nd ed.). MIT Press.
    - PC algorithm
    - Constraint-based causal discovery

11. **Peters, J., Janzing, D., & Schölkopf, B. (2017)**. "Elements of causal inference: foundations and learning algorithms." *MIT Press*.
    - Modern causal inference foundations
    - Functional causal models

12. **Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019)**. "Meta-learners for estimating heterogeneous treatment effects using machine learning." *arXiv preprint arXiv:1706.03762*.
    - X-Learner, S-Learner, T-Learner
    - Machine learning for causal inference

### Software & Libraries

- **DoWhy**: https://github.com/py-causal/dowhy
- **CausalML**: https://github.com/uber/causalml
- **EconML**: https://github.com/Microsoft/EconML
- **CausalTree**: https://github.com/susanathey/causalTree (R)

### Online Resources

- Causal Inference Primer: https://www.youtube.com/c/JuliusFromKorea
- Pearl's Lectures: https://www.youtube.com/results?search_query=judea+pearl+causality
- Brady Neal's Course: https://www.bradyneal.com/causal-inference-course

---

**Document Version**: 1.0  
**Last Updated**: April 2026  
**Total Sections**: 5 major sections + 12 references
