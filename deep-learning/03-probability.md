---
template: default
use_math: true
---

# Probability and Information Theory

## Why Probability?

- Machine learning must always deal with uncertain quantities and sometimes stochastic (nondeterministic) quantities
- Three Possible Sources of Uncertainty
    - Inherent stochasticity in the system being modeled
    - Incomplete observability
    - Incomplete modeling

- *Degree of Belief*
: Degree to which an event is likely to happen

- *Frequentist Probability*
: Probability related directly to the rates at which events occur

- *Bayesian Probability*
: Probability related directly to the qualitative levels of uncertainty

## Random Variables

- *Random Variable*
: A variable that can take on different values randomly
    - Typically denoted with a lowercase letter in plain typeface and the values it can take on with lowercase script letter
    - Random variables can be discrete or continuous
        - Discrete variables are variable that have a finite or countably infinite number of states
        - Continuous variables are variables associated with a real value

## Probability Distribution

- *Probability Distribution*
: A description of how likely a random variable or set of random variables is to take on eah of its possible states

### Discrete Variables and Probability Mass Functions

- *Probability Mass Function (PMF)*
: A probability distribution over discrete variables
    - Typically denoted with a capital *P*

- *Joint Probability Distribution*
: A probability mass function that can act on many variables at the same time
    - Can be written $P(x,y)$ or $P(x = x, y = y)$

- To be a PMF on a random variable x, a function *P* must satisfy the following properties:
    - The domain of *P* must be the set of all possible states of x
    - $\forall x \in x, 0 \le P(x) \le 1$
    - $\sum_{x \in x} P(x) = 1$
        - This property is referred to as being normalized

- *Uniform Distribution*
: When each state is equally likely
    - PMFs of uniform distributions are P(x = x_i) = \frac{1}{k}$

### Continuous Variables and Probability Density Functions

- *Probability Density Function*
:

## Marginal Probability

- *Marginal Probability Distribution*
:

- *Sum Rule*
:

## Conditional Probability

- *Conditional Probability*
:

## The Chain Rule of Conditional Probabilities

-  *Chain Rule*
:

- *Product Rule*
:

## Independence and Conditional Independence

- *Independent*
:

- *Conditionally Independent*
:

## Expectation, Variance, and Covariance

- *Expectation*
:

- *Expected Value*
:

- *Variance*
:

- *Standard Deviation*
:

- *Covariance*
:

- *Correlation*
:

- *Covariance Matrix*
:

## Common Probability Distributions

### Bernoulli Distribution

- *Bernoulli Distribution*
:

### Multinoulli Distribution

- *Multinoulli (Categorical) Distribution*
:

### Gaussian Distribution

- *Normal Distribution*
:

- *Gaussian Distribution*
:

- *Precision*
:

- *Standard Normal Distribution*
:

- *Central Limit Theorem*
:

- *Multivariate Normal Distribution*
:

- *Precision Matrix*
:

### Expopnential and Laplace Distributions

- *Exponential Distribution*
:

- *Laplace Distribution*
:

### The Dirac Distribution and Empirical Distribution

- *Dirac Delta Function*
:

- *Generalized Function*
:

- *Empirical Distribution*
:

- *Empirical Frequency*
:

### Mixtures of Distributions

- *Mixture Distribution*
:

- *Latent Variable*
:

- *Gaussian Mixture Model*
:

- *Prior Probability*
:

- *Posterior Probability*
:

- *Universal Approximator*
:

## Useful Properties of Common Functions

- *Logistic Sigmoid*
:

- *Saturates*
:

- *Softplus Function*
:

- *Logit*
:

- *Positive Part Function*
:

- *Negative Part Function*
:

## Bayes' Rule

- *Bayes' Rule*
:

## Technical Details of Continuous Variables

- *Measure Theory*
:

- *Measure Zero*
:

- *Almost Everywhere*
:

- *Jacobian Matrix*
:

## Information Theory

- *Self-information*
:

- *Nats*
:

- *Bits (Shannons)*
:

- *Shannon Entropy*
:

- *Differential Entropy*
:

- *Kullback-Leibler (KL) Divergence*
:

## Structured Probabilistic Models

- *Structured Probabilistic Model*
:

- *Graphical Model*
:

- *Directed Model*
:

- *Undirected Model*
:

- *Proportional*
:

- *Description*
:

