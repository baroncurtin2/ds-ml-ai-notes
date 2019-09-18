---
template: default
use_math: true
---

# Machine Learning Basics

- Machine learning algorithms have settings called *hyperparameters*, which must be determined outside the learning algorithm itself
- Machine learning is essentially a form of applied statistics with increased emphasis on the use of computers to statistically estimate complicated functions and a decreased emphasis on proving confidence intervals around these functions

## Learning Algorithms

- A machine learning algorithm is an algorithm that is able to learn from data

### The Task, *T*

- *Example*
: A collection of features that have been quantitatively measured from some object or event that we want the machine learning system to process
    - Machine learning tasks are usually described in terms of how the machine learning should process an *example*
    - Typically represented as a vector $x \in \mathbb{R}^n$ where each entry $x_i$ of the vector is another feature

- Types of Tasks
    - Classification: the computer program is asked to specify which of the *k* categories some input belonds to
        - To solve this task, the learning algorithm is usually asked to produce a function $f: \mathbb{R}^n \rightarrow {1,...,k}$
    - Classification with Missing Inputs
        - When some of the inputs may be missing, rather than providing a single classification function, the learning algorithm must learn a *set* of functions
    - Regression: the computer program is asked to predict a numerical value given some input
    - Transcription: the machine learning system is asked to observe a relatively unstructured representation of some kind of data and transcribe the information into discrete textual form
    - Machine Translation: the computer program must convert a sequence of symbols into another language
    - Structured Output: involve any task where the output is a vector (or other data structure containing multiple values) with important relationships between the different elements
    - Anomaly Detection: the computer program sifts through a set of events of objects and flags some of them as being unusual or atypical
    - Synthesis and Sampling: the machine learning algorithm is asked to generate new examples that are similar to those in the training data
    - Imputation of Missing Values: the machine learning algorithm is given a new example $x \in \mathbb{R}^n$, but with some entries $x_i of \textbf{x}$ missing and the algorithm must provide a prediction of the values of the missing entries
    - Denoising: the machine learning algorithm is given as input a *corrupted example* $\~{\textbf{x}} \in \mathbb{R}^n$ obtained by an unknown corruption process from a *clean example* $\textbf{x} \in \mathbb{R}^n$ and the algorithm must predict the clean example $\textbf{x}$ from its corrupted version $\~{\textbf{x}}$
        - More generally, predict the conditional probability distribution $p(\textbf{x} \| \~{\textbf{x}})$
    - Density Estimation / Probability Mass Function Estimation: the machine learning algorithm is asked to learn a function $p_{model} : \mathbb{R}^n \rightarrow \mathbb{R}$, where $p_{model}(x)$ can be interpreted as a probability density function (if x is continuous) or a probability mass function (if x is discrete) on the space that the examples were drawn from

### The Performance Measure, *P*

- To evaluate the abilities of a machine learning algorithm, we must design a quantitative measure of its performance

- *Accuracy*
: Is the proportion of examples for which the model produces the correct output

- *Error Rate*
: The proportion of examples for which the model produces an incorrect output

- *Test Set*
: A set of data that is separate from the data used for training the machine learning system using for evaluating the performance measures


### The Experience, *E*

- Machine learning algorithms can be broadly categorized as **unsupervised** or **supervised** by what kind of experience they are allowed to have during the learning process

- *Unsupervised*
:

- *Supervised*
:

- *Data Set*
: A collection of many examples

- *Data Points*
: Alternative term for examples

- *Unsupervised Learning Algorithms*
: Experience a dataset containing many features, then learn useful properties of the structure of this dataset
    - In the context of deep learning, we usually want to learn the entire probability distribution that generated a dataset, whether explicitly, as in density estimation, or implicitly, for tasks like synthesis or denoising
    - Involves observing several examples of a random vector **x** and attempting to implicitly or explicitly learn the probability distribution $p(x)$, or some interesting properties of that distribution

- *Supervised Learning Algorithms*
: Experience a dataset containing features, but each example is also associated with a **label** or **target**
    - Involves observing several examples of a random vector **x** and an associated value or vector **y**, then learning to predict **y** from **x**, usually by estimating $p(y \| x)$

- *Reinforcement Learning*
: Interact with an environment so there is a feedback loop between the learning system and its experiences

- *Design Matrix*
: A matrix containing a different example in each row
    - Each column of the matrix corresponds to a different feature


### Example: Linear Regression

- *Linear Regression*
: Build a system that can take a vector $\textbf{x} \in \mathbb{R}^n$ as input and predict the value of a scalar $y \in \mathb{R}$ as its output
    - The output of linear regression is a linear function of the input
    - $\hat{y} = \textbf{w}^\intercal \textbf{x}$
        - $\textbf{w} \in \mathbb{R}^n$ is a vector of parameters

- *Parameters*
: Values that control the behavior of the system

- *Weights*
: Determine how each feature affects the prediction

- *Mean Squared Error (MSE)*
 - $\text{MSE}_{test} = \frac{1}{m} \sum_i(\hat{\textbf{y}}^{(test)} - \textbf{y}^{(test)})_i^2$
 - This measure decreases to 0 when $\hat{\textbf{y}}^{(test)} - \textbf{y}^{(test)}$
 - The error increases whenever the Euclidean distance between the predictions and the targets increases
 - To minimize $\text{MSE}_{train}$, we can simply solve for where its gradient is 0:
    - $\nabla_w \text{MSE}_{train} = 0$
    - $\Rightarrow \nabla_w \frac{1}{m} \Vert \hat{\textbf{y}}^{(train)} - \textbf{y}^{(train)} \Vert_2^2 = 0$
    - $\Rightarrow \frac{1}{m} \nabla_w \Vert \textbf{X}^{(train)} \textbf{w} - \textbf{y}^{(train)} \Vert_2^2 = 0$
    - $\Rightarrow \nabla_w (\textbf{X}^{(train)}\textbf{w} - \textbf{y}^{(train)})^\intercal (\textbf{X}^{(train)}\textbf{w} - \textbf{y}^{(train)}) = 0$
    - $\Rightarrow \nabla_w (\textbf{w}^{\intercal} \textbf{X}^{(train) \intercal} \textbf{X}^{(train)}\textbf{w} - 2 \textbf{w}^\intercal \textbf{X}^{(train)\intercal} \textbf{y}^{(train)} + \textbf{y}^{(train)\intercal} \textbf{y}^{(train)}) = 0$
    - $\Rightarrow 2 \textbf{X}^{(train)\intercal} \textbf{X}^{(train)}\textbf{w} - 2 \textbf{X}^{(train)\intercal}\textbf{y}^{(train)} = 0$
    - $\Rightarrow \textbf{w} = (\textbf{X}^{(train)\intercal}\textbf{X}^{(train)})^{-1}\textbf{X}^{(train)\intercal}\textbf{y}^{(train)}$

- To make a machine learning algorithm, we need to design an algorithm that will improve the weights **w** in a way that reduces $\text{MSE}_{test}$ when the algorithm is allowed to gain experience by observing a training set

- *Bias*
: The intercept term in linear regression

## Capacity, Overfitting, Underfitting

- *Generalization*
: The ability to perform well on previously unobserved inputs
    - The central challenge in machine learning

- *Training Error*
: The computed error measure on the training set

- *Generalization Error (Test Error)*
: The expected value of the error on a new input

- *Test Set*
: Examples collected separately from the training set

- *Statistical Learning Theory*
: Provides answers on how we can affect performance on the test set when we can observe only the training set

- *Data-Generating Process*
: How the training and test data are generated by a probability distribution over datasets

- *I.I.D Assumptions*
: Assumptions that the examples in each dataset are independent from each other, and that the training set and test set are identically distributed

- *Identically Distributed*
: Drawn from the same probability distribution as each other

- *Data-Generating Distribution*
: The shared underlying distribution that is used to generate every train example and every test example
    - Denoted $p_{data}$

- *Underfitting*
: Occurs when the model is not able to obtain a sufficiently low error value on the training set

- *Overfitting*
: The gap between the training error and test error is too large

- *Capacity*
: The model's ability to fit a wide variety of functions
    - Machine learning algorithms will generally perorm best when their capacity is appropriate for the true complexity of the task they need to perform and the amount of training data they are provided with

- *Hypothesis Space*
: The set of functions that the learning algorithm is allowed to select as being the solution
    - One way to control the capacity of a learning algorithm

- *Representational Capacity*
: The model specifies which family of functions the learning algorithm can choose from when varying the parameters in order to reduce a training objective

- *Occam's Razor*
: States that among competing hypotheses that explain known observations equally well, we should choose the "simplest" one

- *Vapnik-Chervonenkis Dimension (VC Dimension)*
: Measures the capacity of a binary classifier and is the largest possible value of *m* for which there exists a training set of *m* different $\textbf{x}$ points that the classifier can label arbitrarily

- *Nonparametric Models*
: Opposite of parametric models
    - Parametric models learn a function described by a parameter vector whose size is finite and fixed before any data is observed
    - Nonparametric models have no such limitation

- *Nearest Neighbor Regression*
: Stores the $\textbf{X}$ and $\textbf{y}$ from the training set and when asked to classify a test point, the model looks up the nearest entry in the training set and returns the associated regression target
    - An example of a nonparametric model
    - $\hat{y} = y_i$, where $i = \text{argmin}\Vert \textbf{X}_{i,:} - \textbf{x} \Vert_2^2$
    - When a nearest neighbor algorithm is allowed to break ties by averaging the $y_i$ values for all $\textbf{X}_{i,:}$ that are tied for nearest, then this algorithm is able to achieve the minimum possible training error on any regression dataset

- *Bayes Error*
: The error incurred by an oracle making predictions from the true distribution $p(\textbf{x}, y)$

### The No Free Lunch Theorem

- *No Free Lunch Theorem*
: States that averaged over all possible data-generating distributions, every classification algorithm has the same error rate when classifying previously unobserved points

- The goal of machine leaning is to understand what kinds of distributions are relevant to the "real world" that an AI agent experiences, and what kinds of machine learning algorithms performs well on data drawn from the kinds of data-generating distributions we care about

### Regularization

- The no free lunch theorem implies that we must design out machine learning algorithms to perform well on a specific task
- The behavior of our algorithm is strongly affected not just by how large we make the set of functions allowed in its hypothesis space, by by the specific identity of those functions

- *Weight Decay*
: We minimize a sum $J(\textbf{w})$ comprising both the mean squared error on the training and a criterion that expresses a preference for the weights to have smaller squared $L^2$ norm
    - $J(\textbf{w}) = \text{MSE}_{train} + \lambda \textbf{w}^\intercal \textbf{w}$
    - $\lambda$ is a value chosen ahead of time that controls the strength of our preference for smaller weights

- *Regularizer*
: We can regularize a model that learns a function $f(x;0)$ by adding a penalty to the cost function
    - In the case of weight decay, the regularizer is $\Omega (\textbf{w}) = \textbf{w}^\intercal \textbf{w}$

- *Regularization*
: Any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error

## Hyperparameters and Validation Sets

- Most machine learning algorithms have hyperparameters, settings that we can use to control the algorithm's behavior
- We can always fit the training set better with a higher-degree polynomial and a weight decay setting of $\lambda = 0$ than we could with a lower-degree polynomial and a positive weight decay setting

- *Validation Set*
: A set of examples that the training algorithm does not observe
    - Always constructed from the training data
    - This is the subset of the training data used to guide the hyperparameters

### Cross-Validation

- When the dataset is too small, alternative procedures enable one to use all the examples in the estimation of the mean test error, at the price of increased computational cost
- These procedures are based on the idea of repeating the training and testing computation on different randomly chosen subsets or splits of the original dataset
- The most common is the k-fold cross-validation procedure, in which a partition of the dataset is formed by splitting it into *k* nonoveralpping subsets
    - The test error may then be estimated by taking the average test error across *k* trials

## Estimators, Bias, and Variance

### Point Estimation

- Point estimation is the attempt to provide the single "best" prediction of some quantity of interest

- *Point Estimator (Statistic)*
: Any function of the data
    - A good estimator is a function whose output is close to the true underlying $\theta$ that generated the training data

- *Function Estimation*
: Trying to predict a variable $\textbf{y}$ given an input vector $\textbf{x}$

### Bias

- The bias of an estimator is defined as $\text{bias}(\hat{\theta}_m) = \mathbb{E}(\hat{\theta}_m - \theta)$

- *Unbiased*
: When $\text{bias}(\hat{\theta}_m) = 0$

- *Asymptotically Unbiased*
: When $\lim_{m \to \infty}\text{bias}(\hat{\theta}_m) = 0$

- While unbiased estimators are clearly desirable, they are not always the "best" estimators

### Variance and Standard Error

- *Variance*
: There variance of an estimator is simple the variance $Var(\hat{\theta})$

- *Standard Error*
: The square root of the variance

- The variance, or the standard error, of an estimator provides a measure of how we would expect the estimate we compute from data to vary as we independently resample the dataset from the underlying data-generating process

### Trading off Bias and Variance to Minimize Mean Squared Error

- Bias measures the expected deviation from the true value of the function or parameter
- Variance provides a measure of the deviation from the expected estimator value that any particular sampling of the data is likely to cause
- The most common way to negotiate this trade-off is to use cross-validation

- *Mean Squared Error (MSE)*
: $\text{MSE} = \mathbb{E}[(\hat{\theta}_m - \theta)^2]
    - $\text{MSE} = \text{Bias}(\hat{\theta}_m)^2 + \text{Var}(\hat{\theta}_m)$
    - Measures the over expected deviation-in a squared error sense-between the estimator and the true value of the parameter $\theta$

### Consistency

- We usually wish that as the number of data points $m$ in our dataset increases, our point estimates converge to the true value of the corresponding parameters
    - We would like $\text{p}\lim_{m \to \infty}\hat{\theta}_m = 0$

- *Consistency*
: The condition described by the equation above
    - Strong consistency when $\hat{\theta} \to \theta$

- *Almost Sure Convergence*
: Occurs when $p(\lim_{m \to \infty}\textbf{x}^{(m)} = x) = 1$

## Maximum Likelihood Estimation

- Taking the logirithm of the likelihood does not change its argmax but transforms a product into a sum:
    - $$\theta_{ML} = \text{arg}_{\theta} \max \sum_{i=1}^m \log p_{model}(\textbf{x}^{(i)};\theta)$$

- The KL divergence is given by
    - $$D_{KL}(\hat{p}_{data}\Vert p_{model}) = \mathbb{E}_{x \approx \hat{p}_{data}}[\log \hat{p}_{data}(x) - \log p_{model}(x)]$$

- Minimizing this KL divergence corresponds exactly to minimizing the cross-entropy between distributions
    - Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the training set and the probability distribution defined by model

### Conditional Log-Likelihood and Mean Squared Error

- The conditional maximum likelihood estimator is $\theta_{ML} = \text{arg}_{\theta}\max P(\textbf{Y} \| \textbf{X}; \theta)$
- If the examples are assumed to be i.i.d, then this can be decomposed into $$\theta_{ML} = \text{arg}_{\theta}\max \sum_{i=1}^m \log P(\textbf{y}^{(i)} \| \textbf{x}^{(i)}; \theta)$$

### Properties of Maximum Likelihood

- The main appeal of the maximum likelihood estimator is that it can be shown to be the best estimator asymptotically as the number of examples $m \to \infty$, in terms of its rate of convergence as $m$ increases

- *Statistical Efficiency*
: One consistent estimator may obtain lower generalization error for a fixed number of samples $m$, or equivalently, may require fewer examples to obtain a fixed level of generalization error

- *Parametric Case*
: The goal is to estimate the value of a parameter, not the value of a function

- Maximum likelihood is often considered the preferred estimator to use for machine learning because of consistency and efficiency
- When the number of examples is small enough to yield overfitting behavior, regularization strategies such as weight decay may be used to obtain a biased version of maximum likelihood that has less variance when training data is limited

## Bayesian Statistics

- *Frequentist Statistics*
:

- *Bayesian Statistics*
:

- *Prior Probability Distribution*
:

### Maximum a Posteriori (MAP) Estimation

- *Maximum a Posterior (MAP) Estimate*
: 

## Supervised Learning Algorithms

### Probabilistic Supervised Learning

- *Logistic Regression*
:

### Support Vector Machines

- *Kernel Trick*
:

- *Kernel*
:

- *Gaussian Kernel*
:

- *Radial Basis Function*
:

- *Template Matching*
:

- *Kernel Machines (Kernel Methods)*
:

- *Support Vectors*
:

### Other Simple Supervised Learning Algorithms

- *Decision Tree*
:

## Unsupervised Learning Algorithms

### Principal Components Analysis


### *k*-means Clustering


## Stochastic Gradient Descent

- *Stochastic Gradient Descent*
:

- *Minibatch*
:

## Building a Machine Learning Algorithm

## Challenges Motivating Deep Learning

### The Curse of Dimensionality

- *Curse of Dimensionality*
:

### Local Constancy and Smoothness Regularization

- *Smoothness Prior (Local Constancy Prior)*
:

- *Local Kernels*
:

### Manifold Learning

- *Manifold*
:

- *Manifold Learning*
:

- *Manifold Hypothesis*
:

