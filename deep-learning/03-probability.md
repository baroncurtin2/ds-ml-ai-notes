---
template: default
use_math: true
---

# Probability and Information Theory

## Why Probability?

- Machine learning must always deal with uncertain quantities and sometimes stochastic, nondeterministic, quantities
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
    - PMFs of uniform distributions are $P(x = x_i) = \frac{1}{k}$

### Continuous Variables and Probability Density Functions

- *Probability Density Function (PDF)*
: The probability distribution of continuous random variables
    - To be a probability density function, a function *p* must satisfy the following properties:
        - The domain of *p* must be the set of all possible states of x
        - $\forall x \in x, p(x) \ge 0$
        - $\int p(x)dx = 1$
    - The PDF can be integrated to find the actual probability mass of a set of points
        - The probability that $x$ lies in some set $\mathbb{S}$ is given by the integral of $p(x)$ over that set
    - Gives the probability of landing inside an infinitesimal region with volume $\delta x$ is given by $p(x)\delta x$

## Marginal Probability

- *Marginal Probability Distribution*
: The probability distribution over the subset of variables

- *Sum Rule*
: Used to find the marginal probability distribution
    - $\forall x \in x, P(x = x) = \sum_y P(x = x, y = y)$

- For continuous variables, we need to use integration instead of summation $p(x) = \int p(x,y)dy$

## Conditional Probability

- *Conditional Probability*
: The probability of some event given that some other event happened
    - Can be computed with the formula $P(y = y\|x = x) = \frac{P(y = y, x = x)}{P(x = x)}$
    - Only defined when $P(x = x) > 0$

- *Intervention query*
: Computing the consequences of an action

- *Causal modeling*
: Intervention queries are used in this domain

## The Chain Rule of Conditional Probabilities

-  *Chain Rule (Product Rule)*
: Any joint probability distribution over many random variables may be decomposed into conditional distributions over only one variable
    - $P(x^1,...,x^n) = P(x^1)\Pi_{i=2}^{n} P(x^i \| x^1,...,x^{i-1})$

## Independence and Conditional Independence

- *Independent*
: When two random variables' probability distributions can be expressed as a product of two factors
    - $\forall x \in x, y \in y, p(x = x, y = y) = p(x = x)p(y = y)$

- *Conditionally Independent*
: When two random variables, when given a random variable z, if their conditional probability distributions factorizes in this way for every value of z

## Expectation, Variance, and Covariance

- *Expectation (Expected Value)*
: The average, or mean value of some function $f(x)$ with respect to a probability distribution $P(x)$ that $f$ takes on when $x$ is drawn from $P$
    - For discrete variables, this can be computed with a summation
        -$\mathbb{E}_{x~P}[f(x)] = \sum_xP(x)f(x)$
    - For continuous variables, it is computed with an integral
        -$\mathbb{E}_{x~p}[f(x)] = \int p(x)f(x)dx$

- *Variance*
: Gives a measure of how much the values of a function of a random variable *x* may vary as we sample different values of x from its probability distribution
    - $Var(f(x)) = \mathbb{E}[(f(x)) - \mathbb{E}[f(x)]^2]$
    - When the variance is low, the values of $f(x)$ cluster near their expected value

- *Standard Deviation*
: Square root of the variance

- *Covariance*
: Gives some sense of how much two values are linearly related to each other as well as the scale of these values
    - $Cov(f(x), g(y)) = \mathbb{E}[(f(x) - \mathbb{E}[f(x)]) g(y) - \mathbb{E}[g(y)]]$
    - High absolute values of the covariance mean that the values change very much and are both far from their respective means at the same time
    - If covariance is positive, then both variables tend to take on relatively high values simultaneously
    - If covariance is negative, then one variable tends to take on a relatively high value at the times that the other takes on a relatively low value and vice versa

- *Correlation*
: Normalizes the contribution of each variable in order to measure only how much the variables are related, rather than also being affected by the scale of the separate variables

- *Covariance Matrix*
: A random vector $\textbf{x} \in \mathbb{R}^n$ is an $n \times n$ matrix, such that
    - $Cov(x)_{i,j} = Cov(x_i, x_j)$
    - The diagonal elements of the covariance matrix give the variance
        - $Cov(x_i, x_i) = Var(x_i)$

## Common Probability Distributions

### Bernoulli Distribution

- *Bernoulli Distribution*
: A distribution over a single binary random variable
    - It has the following properties
        - $P(x = 1) = \phi$
        - $P(x = 0) = 1 - \phi$
        - $P(x = x) = \phi^x(1 - \phi)^{1-x}$
        - $\mathbb{E}_x[x] = \phi$
        - $Var_x(x) = \phi (1 - \phi)$

### Multinoulli Distribution

- *Multinoulli (Categorical) Distribution*
: A distribution over a single discrete variable with *k* different states, where *k* is finite
    - Is parameterized by a vector $p \in [0,1]^{k-1}$ where $p_i$ gives the probability of the i-th state
    - The final k-th state's probability is given by $1 - 1^Tp$, where $1^Tp \le 1$

### Gaussian Distribution

- *Normal Distribution (Gaussian Distribution)*
: The most commonly used distribution over real numbers
    - $\mathcal{N}(x;\mu,\sigma^2) = \sqrt{\frac{1}{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x - \mu)^2)$
    - The two parameters $\mu \in \mathbb{R}$ and $\sigma \in (0, \inf)$ control the normal distribution
    - The parameter $\mu$ gives the coordinate of the central peak and is the mean of the distribution: $\mathbb{E}[x] = \mu$
    - The standard deviation of the distribution is given by $\sigma$, and the variance by $\sigma^2$
    - When we evaluate the PDF, we need to square and invert $\sigma$
    - A more efficient way of parametrizing the distribution is to use a parameter $\beta \in (0, \inf)$ to control the precision, or inverse variance of the distribution
        - $\mathcal{N}(x;\mu,\beta^{-1}) = \sqrt{\frac{\beta}{2\pi}}\exp(-\frac{1}{2}\beta(x - \mu)^2)$

- *Precision*
: The inverse variance of the distribution

- *Central Limit Theorem*
: Shows that the sum of many independent random variables is approximately normally distributed

- *Multivariate Normal Distribution*
: When the normal distribution is generalized to $\mathbb{R}^n$
    - It may be parametrized with a positive definite symmetric matrix $\Sigma$
    - $\mathcal{N}(x;\mu,\Sigma) = \sqrt{\frac{1}{(2\pi)^n\det(\Sigma)}}\exp(-\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu))$
        - $\mu$ still gives the mean of the distribution
        - $\Sigma$ gives the covariance matrix of the distribution
    - When we need to evaluate the PDF several times for many different values of the parameters, we can use a *precision matrix* $\beta$
        - $\mathcal{N}(x;\mu,\beta^{-1}) = \sqrt{\frac{\det(\beta)}{(2\pi)^n}}\exp(-\frac{1}{2}(x - \mu)^T\beta^{-1}(x - \mu))$

- *Precision Matrix*
: Used instead of covariance matrix when we need to evaluate the PDF several times for many different values of the parameters

- *Isotropic Gauassian Distribution*
: Covariance matrix is a scalar times the identity matrix

### Expopnential and Laplace Distributions

- *Exponential Distribution*
: A probability distribution with a sharp point at $x = 0$
    - $p(x; \lambda) = \lambda1_{x \ge 0}\exp(-\lambda x)$

- *Laplace Distribution*
: A distribution closely related to the exponential distribution that allows us to place a sharp peak of probability mass at an arbitrary point $\mu$
    - $\text{Laplace}(x;\mu,\gamma) = \frac{1}{2\gamma} \exp(-\frac{\|x - \mu\|}{\gamma})$

### The Dirac Distribution and Empirical Distribution

- *Dirac Delta Function*
: When we wish to specify that all the mass in a probability distribution clusters around a single point
    - $p(x) = \delta(x - \mu)$
    - Defined such that it is zero valued everywhere except 0, yet integrates to 1
    - Commonly used as a component of an empirical distribution
        - $\hat{p}(x) = \frac{1}{m}\sum_{i=1}^{m}\delta(x - x^i)$

- *Generalized Function*
: A function that is defined in terms of its properties when integrated

- *Empirical Distribution*
: Put probability mass $\frac{1}{m}$ on each of the *m* points $x^1,...,x^m$ forming a given data set or collection of samples

- *Empirical Frequency*
: For discrete variables, an empirical distribution can be conceptualized as a multinoulli distribution with a probability associated with each possible input value that is simple equal to the empirical frequency of that value in the training set

### Mixtures of Distributions

- *Mixture Distribution*
: Probability distributions defined by combining other simpler probability distributions
    - $P(x) = \sum_iP(c = i)P(x\|c = i)$

- *Latent Variable*
: A random variable that we cannot observe directly

- *Gaussian Mixture Model*
: A powerful and common type of mixture model
    - Components $p(x\|c = i)$ are Gaussians
    - Each component has a separately parametrized mean $\mu^i$ and covariance $\Sigma^i$

- *Prior Probability*
: Expresses the model's beliefs about c before it has observed x
    - $\alpha_i = P(c = i)$

- *Posterior Probability*
: Computed after observation of x
    - $P(c\|x)$

- *Universal Approximator*
: Any smooth density can be approximated with any specific nonzero amount of error by a Gaussian mixture model with enough components

## Useful Properties of Common Functions

- *Logistic Sigmoid*
: Commonly used to produce the $\phi$ parameter of a Bernoulli distribution because its range is (0,1), which lies within the valid range of values for $\phi$
    - $\sigma(x) = \frac{1}{1 + \exp(-x)}$

- *Saturates*
: A function that becomes very flat and insensitive to small changes in its input

- *Softplus Function*
: Useful for producing the $\beta$ or $\sigma$ parameter of a normal distribution because its range is (0, $\inf$)
    - $\zeta(x) = \log(1 + \exp(x))$

- Properties to memorize:
    - $\sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(0)}$
    - $\frac{d}{dx}\sigma(x) = \sigma(x)(1 - \sigma(x))$
    - $1 - \sigma(x) = \sigma(-x)$
    - $\log\sigma(x) = -\zeta(-x)$
    - $\frac{d}{dx}\zeta(x) = \sigma(x)$
    - $\forall x \in (0, 1), \sigma^{-1}(x) = \log(\frac{x}{1 - x})$
    - $\forall x \gt 0, \zeta^{-1}(x) = \log(\exp(x) - 1)$
    - $\zeta(x) = \int_{-\inf}^{x} \sigma(y)dy$
    - $\zeta(x) - \zeta(-x) = x$

- *Logit*
: The function $\sigma^{-1}$

## Bayes' Rule

- *Bayes' Rule*
: When we know $P(y\|x)$ and need to know $P(x\|y), but also know $P(x)$ so we can compute the desired quantity

## Technical Details of Continuous Variables

- *Measure Theory*
: Useful for describing theorems that apply to most points in $\mathbb{R}^n$ but do not apply to some corner cases

- *Measure Zero*
: A set of points that is negligibly small

- *Almost Everywhere*
: A property of measure theory that everywhere holds throughout all space except for on a set of meaure zero

- *Jacobian Matrix*
:

## Information Theory

- A branch of applied mathematics that revolves around quantifying how much information is present in a signal
- We would like to quantify information in a way that formalizes this intuition:
    - Likely events should have low information content, and in the extreme case, events that are guaranteed to happen should have no information content whatsoever
    - Less likely events should have higher information content
    - Independent events should have additive information

- *Self-information*
: Defined as $I(x) = -\log P(x)$

- *Nats*
: When using the natural logarithm, one nat is the amount of information gained by observing an event of probability $\frac{1}{e}$

- *Bits (Shannons)*
: When using the base-2 logarithm, one bit is the amount of information gained by observing an event of probability $\frac{1}{e}$

- *Shannon Entropy*
: The amount of uncertainty in an entire probability distribution
    - $$H(x) = \mathbb{E}_{x~P} [I(x)] = - \mathbb{E}_{x~P} [\log P(x)]$$
    - The expected amount of information in an event drawn from that distribution

- *Differential Entropy*
: The Shannon entropy is known as this when x is continuous

- *Kullback-Leibler (KL) Divergence*
: When we have two separate probability distributions $P(x)$ and $Q(x)$ over the same random variable x, the measure of how different they are
    - $$D_{KL}(P\|\|Q) = \mathbb{E}_{x~P}[\log \frac{P(x)}{Q(x)}] = \mathbb{E}_{x~P} [\log P(x) - \log Q(x)]$$

## Structured Probabilistic Models

- *Structured Probabilistic Model (Graphical Model)*
: When we represent the factorization of a probability distribution with a graph

- *Directed Model*
: Use models with directed edges and they represent factorizations into conditional probability distributions
    - $p(x) = \Pi_i p(x_i \| P_{\mathcal{aG}}(xi))$

- *Undirected Model*
: Uses graphs with undirected edges and they represent factorizations into a set of functions and are not usually probability distributions of any kind

- *Proportional*
: The probability of a configuration of random variables is proportional to the product of all these factors

- *Description*
: Being directed or undirected is not a property of a probability distribution; it is a property of a particular description of a probability distribution, but any probability distribution may be described in both ways
