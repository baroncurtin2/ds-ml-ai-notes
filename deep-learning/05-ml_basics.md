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
    - Machine Translation
    - Structured Output
    - Anomaly Detection
    - Synthesis and Sampling
    - Imputation of Missing Values
    - Denoising
    - Density Estimation / Probability Mass Function Estimation

### The Performance Measure, *P*

- *Accuracy*
:

- *Error Rate*
:

- *Test Set*
:


### The Experience, *E*

- *Unsupervised*
:

- *Supervised*
:

- *Data Set*
:

- *Data Points*
:

- *Unsupervised Learning Algorithms*
:

- *Supervised Learning Algorithms*
:

- *Label (Target)*
:

- *Supervised Learning*
:

- *Reinforcement Learning*
:

- *Design Matrix*
:


### Example: Linear Regression

- *Linear Regression*
:

- *Parameters*
:

- *Weights*
:

- *Mean Squared Error*
:

- *Normal Equations*
:

- *Bias*
:

## Capacity, Overfitting, Underfitting

- *Generalization*
:

- *Training Error*
:

- *Generalization Error*
:

- *Test Error*
:

- *Test Set*
:

- *Statistical Learning Theory*
:

- *Data-Generating Process*
:

- *I.I.D Assumptions*
:

- *Independent*
:

- *Identically Distributed*
:

- *Data-Generating Distribution*
:

- *Underfitting*
:

- *Overfitting*
:

- *Capacity*
:

- *Hypothesis Space*
:

- *Effective Capacity*
:

- *Occam's Razor*
:

- *Vapnik-Chervonenkis Dimension*
:

- *Underfitting Regime*
:

- *Overfitting Regime*
:

- *Optimal Capacity*
:

- *Nonparametric Models*
:

- *Nearest Neighbor Regression*
:

- *Bayes Error*
:

### The No Free Lunch Theorem

- *No Free Lunch Theorem*
:

### Regularization

- *Weight Decay*
:

- *Regularizer*
:

- *Regularization*
:

## Hyperparameters and Validation Sets

- *Capacity*
:

- *Validation Set*
:

### Cross-Validation

## Estimators, Bias, and Variance

### Point Estimation

- *Point Estimator (Statistic)*
:

- *Function Estimation*
:

### Bias

- *Unbiased*
:

- *Asymptotically Unbiased*
:

- *Sample Mean*
:

- *Sample Variance*
:

- *Unbiased Sample Variance*
:

### Variance and Standard Error

- *Variance*
:

- *Standard Error*
:

### Trading off Bias and Variance to Minimize Mean Squared Error

- *Mean Squared Error (MSE)*
:

### Consistency

- *Consistency*
:

- *Almost Sure Convergence*
:

## Maximum Likelihood Estimation

### Conditional Log-Likelihood and Mean Squared Error

### Properties of Maximum Likelihood

- *Statistical Efficiency*
:

- *Parametric Case*
:

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

