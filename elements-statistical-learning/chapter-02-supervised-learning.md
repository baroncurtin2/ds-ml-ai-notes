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

- Linear Model
  - Given a vector of inputs $X^T = (X_1, X_2, ..., X_p)$, the model that predicts the output $Y$ is 

  $$\hat{Y} = \hat{\beta}_0 + \sum_{j=1}^{p} X_j\hat{\beta}_j$$

  - The term $\hat{\beta}_0$ is the *intercept*, or *bias* in machine learning
  - People often include the constant variable 1 in $X$ and include $\hat{\beta}_0$ in the vector of coefficients $\hat{\beta}$
  - The linear model in vector form is
  $$\hat{Y} = X^T\hat{\beta}$$
    - $X^T$ denotes a column vector, transposed to be a row vector, or a matrix tranpose
    - $\hat{Y}$ is a vector of scalars
    - If $\hat{Y}$ is a *K*-vector then $\beta$ would be a $p x K$ matrix of coefficients
  - Viewed as a function over the *p*-dimensional input space:
    - $f(X) = X^T\beta$ is linear
    - The gradient (derivative), $f'(X) = \beta$, is a vector in input space that points in the steepest uphill direction

- Least Squares
  - The most popular method of fittting a linear model to a set of training data is least square where coefficients $\beta$ are picked to minimize the *residual sum of squares*

  $$RSS(\beta) = \sum_{i=1}^{N}(y_i - x_i^T\beta)^2$$

  - $RSS(\beta)$ is a quadratic function of the parameters
  - A minimum will always exist but may not be unique
  - RSS in matrix notation

  $$RSS(\beta) = (y - X\beta)^T(y - X\beta)$$

    - **X** is an $N * p$ matrix where each row is an input vector
    - **y** is an *N*-vector of the outputs in the training set
    - Finding the derivative with respect to $\beta$:

      $$X^T(y - X\beta) = 0$$

    - If $X^TX$ is nonsingular, $\hat{\beta}$ is:

    $$\hat{\beta} = (X^TX)^-1X^Ty$$

    - The fitted value at the *i*th input $x_i$ is $\hat{y}_i = \hat{y}(x_i) = x_i^T\hat{\beta}$
    - At an arbitrary input $x_0$, the prediction is $\hat{y}(x_0) = x_0^T\hat{\beta}$

## Nearest-Neighbor Methods

- Nearest-neighbor methods use observations in the training set $\tau$ closest in input sapce to $x$ to form $\hat{Y}$
- The equation for *k*-nearest neighbor fit ($\hat{Y}$) is defined as:

$$\hat{Y}(x) = \frac{1}{k} \sum_{x_i \epsilon N_k(x)}y_i$$

  - $N_k(x)$ is the neighborhood of $x$ defined by the $k$ closest points $x_i$ in the training sample
  - Closeness is defined by the Euclidean distance formula

  $$d(q, p) = \sqrt{(q_1 - p_1)^2 + (q_2 - p_2)^2 + ... + (q_n - p_n)^2}$$

  or 

  $$\sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$

  - The *effective* number of parameters of *k*-nearest neighbors is $\frac{N}{k}$ and is generally bigger than *p*, and decreases with increasing *k*


## From Least Squares to Nearest Neighbors

- The linear decision boundary from least squares is very smooth and stable to fit but relies on the assumption that a linear decision boundary is appropriate
- *k*-nearest neighbor procedures do not rely on any stringent assumptions about underlying data and can adapt to any situation
  - This means however that any subregion of the decision boundary depends on a handful of input points and their positions, which causes high variance and low bias
- *k*-nearest neighbors has been enhanced by:
  - Kernel methods use weights that decrease smoothly to zero with distance from the target point, rather than the effect 0/1 weights used by *k*-nearest neighbors
  - In high-dimensional spaces the distance kernels are modified to emphasize some variable more than others
  - Local regression fits linear models by locally weighted least squares rather than fitting constants locally
  - Linear models fit to a basis expansion of the original inputs allow arbitrarily complex models
  - Projection pursuit and neural network models consist of sums of non-linearly transformed linear models


# Statistical Decision Theory

- Let $X /epsilon \mathbb{R}^p$ denote a real valued random input vector
- Let $Y /epsilon \mathbb{R}$ denote a real valued random output variable with joint distribution $Pr(X, Y)$
- We seek a function $f(X)$ for predicting $Y$ given values of the input $X$
- This theory requires a *loss function* $L(Y, f(X))$ for penalizing errors in prediction
  - The most common and convenient is the *squared error loss*

  $$L(Y, f(X)) = (Y - f(X))^2$$

- The criterion for choosing *f* becomes or the expected (squared) prediction error:

$$EPE(f) = E(Y - f(X))^2$$


$$ = \int{[y - f(x)]^2Pr(dx, dy)}$$

- By conditioning on $X$, we can write EPE as:

$$EPE(f) = E_XE_{Y|X}([Y - f(X)]^2|X)$$

- Minimizing EPE pointwise:

$$f(x) = argmin_cE_{Y|X}([Y - c]^2|X = x)$$

- The solution is:

$$f(x) = E(Y|X = x)$$

  - This is the conditional expectation or the *regression* function
  - The best prediction of $Y$ ay any point $X = x$ is the conditional mean when best is measured by average squared error

- Nearest neighbor methods attempt to directly implment this using the training data
  - At each point $x$, we might ask for the average of all $y_i$s with input $x_i = x$

  $$\hat{f}(x) = Ave(y_i|x_i \epsilon N_k(x))$$

    - "Ave" denotes average
    - $N_k(x)$ is the neighborhood containing $k$ points in $\tao$ closest to $x$
    - Two approximations are made here:
      - Expectation is approximated by averaging over sample data
      - Conditioning at a point is relaxed to conditioning on some region "close" to the target point
  - Under mild regularity conditions on the join probability distribution $Pr(X, Y)$:
    - As $N, k \to \inf$ such that $k/N \to 0$, 
    $\hat{f}(x) \to E(Y|X = x)$

- Both *k*-nearest neighbors and least sqaures approximate conditional expectations by averages but:
  - Least squares assumes $f(x)$ is well approximated  by a globally linear function
  - *k*-nearest neighbors assumes (f) is well approximated by a locally constant function

- Additive models assume that

$$ f(X) = \sum_{j=1}^{p}f_j(X_j)

# Local Methods in High Dimensions


# Statistical Models, Supervised Learning and Function Approximation



# Structured Regression Models



# Classes of Restricted Estimators



# Model Selection and the Bias-Variance Tradeoff