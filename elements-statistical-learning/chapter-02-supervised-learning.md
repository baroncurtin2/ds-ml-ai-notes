---
template: default
use_math: true
---


# Introduction

- **Inputs**: set of variables that are measrured or preset. Also called *predictors*, *features* in pattern recognition, and *independent variables*

- **Outputs**: set of variables that are influenced by the *inputs*. Also called *responses* or *dependent variables*

- **Supervised Learning**: an exercise where the goal is to use *inputs* to predict the value of the *outputs*

# Variable Types and Terminology

## Types of Variables

- **Quantitative Variables**: variables that are numerical and represent a measurable quantity

- **Qualtitative Variables**: variables that are names of labels
  - Also referred to as *categorical* or *discrete* variables, or *factors*
  - Typically are represented by numerical codes and are referred to as *targets*
    - Binary cases are represented by 0 and 1
    - Dummy variables are used when there are mor than 2 categories
- **Ordered Categorical**: variables where there is an ordering between the values but not metric notion is appropriate. The difference between two variables may not be the same as the difference between the next two set of variables i.e. small-medium and medium-large

## Prediction Tasks

- **Regression**: predict quantitative outputs

- **Classification**: predict qualitative outputs

## Textbook Conventions

- An input variable will typically be denoted by the symbol ***X***

- If ***X*** is a vector, its components can be access by subscripts $X_j$

- Quantitative outputs will be denoted by ***Y***
  - If $Y$ takes on all real number values $\mathbb{R}$, then $\hat{y}$ will also take on all real number values

- Qualtitative outputs will be denoted by ***G***, for group
  - For categorical outputs, $\hat{G}$ should also take on all real number values $\mathbb{R}$ if $G$ takes on all real number values
  - For binary classification, binary coded targets are coded as $Y$ and treated as quantitative output
  - Predictions of $\hat{Y}$ will lie in $[0, 1]$ and class labels can be assigned according to whether $\hat{y} \geq .5$

- Observed values are written in lowercase
  - The *i*th observed value of *X* is written as $x_i$. $x_i$ is a scalar or vector
- Matrices are reprented by bold uppercase letters
  - a set of *N* input p-vectors $x_i$, $i = 1,...,N$ would be represented by the $N * p$ matrix ***X***
  - In general, vectors will not be bold, except when they have $N$ components
  - This convention distinguishes a $p$-vector of inputs $x_i$ for the $i$th observaion from the $N$-vector **$x$**$_j$ consisting of all the observations on variable $X_j$
  - All vectors are assumed to be column vectors and the $i$th row of **X** is $x_i^T$, the vector transpose of $x_i$
  - [Good explanation of the above](https://stats.stackexchange.com/questions/224374/help-understanding-p-vector-language)
- The learning task is: given the value of an input vector $X$, make a good prediction of the output $Y$ denoted by $\hat{Y}$, "y-hat"

## Prediction
- Prediction rules need to be created based of off *training data*


# Two Simple Approaches to Prediction: Least Squares and Nearest Neighbors
- The linear model makes huge assumptions about shape and structure but yield stable yet possibly inaccurate predictions
- The *k*-nearest neighbors makes mild structural assumptions but its prediction are often accurate yet can be unstable


## Linear Models and Least Squares
- Given a vector of inputs $X^T = (X_1, X_2, ..., X_p)$, the model that predicts the output $Y$ is 

$$\hat{Y} = \hat{\beta}_0 + \sum_{j=1}^{p} X_j\hat{\beta}_j$$

- The term $\hat{\beta}_0$ is the *intercept*, or *bias* in machine learning

- People often include the constant variable 1 in $X$ and include $\hat{\beta}_0$ in the vector of coefficients $\hat{\beta}$
- The linear model in vector form is
$$\hat{Y} = X^T\hat{\beta}$$
  - $X^T$ denotes a column vector, transposed to be a row vector, or a matrix tranpose
  - $\hat{Y}$ is a vector of scalars
  - If $\hat{Y}$ is a *K*-vector then $\beta$ would be a $p x K$ matrix of coefficients

## Nearest-Neighbor Methods

## From Least Squares to Nearest Neighbors


# Statistical Decision Theory


# Local Methods in High Dimensions


# Statistical Models, Supervised Learning and Function Approximation



# Structured Regression Models



# Classes of Restricted Estimators



# Model Selection and the Bias-Variance Tradeoff