---
template: default
use_math: true
---

# Numerical Computation

## Overflow and Underflow

- The fundamental difficulty in performing continuous math on a digital computer is that we need to represent infinitely many real numbers with a finite number of bit patterns
- This means that for all all real numbers, we incur some approximation error when we represent the number in the computer

- *Underflow*
: Occurs when numbers near zero are rounded to zero

- *Overflow*
: Occurs when numbers with large magnitude are approximated as $\infty$ or $-\infty$

- *Softmax Function*
: Often used to predict the probabilities associated with a multinoulli distribution
    - Must be stabilized against underflow and overflow
    - $$\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_{j=1}^n\exp(x_j)}$$

## Poor Conditioning

- *Condition Number*
: The ratio of the magnitude of the largest and smallest eigenvalue
    - $$\text{max}_{i,j}\left|\frac{\lambda_i}{\lambda_j}\right|$$

## Gradient-Based Optimization

- *Objective Function (Criterion)*
: The function we want to minimize or maximize
    - When minimizing, it may also be called the *cost function*, *loss function*, *error function*
    - Often denote the value that minimizes or maximizes a function with a superscript \*, for example: $x^* = \text{argmin} f(x)$

- *Derivative*
: Gives the slope of $f(x)$ at the point x
    - Specifies how to scale a small change in the input to obtain the corresponding change in the output: $f(x + \epsilon) \approx f(x) + \epsilon f'(x)$

- *Gradient Descent*
: The process of reducing $f(x)$ by moving $x$ in small steps with the opposite sign of the derivative

- *Critical Points (Stationary Points)*
: Points where $f'(x) = 0$
    - Provides no information about which direction to move

- *Local Minimum*
: A point where $f(x)$ is lower than at all neighboring points so it is no longer possible to decrease $f(x)$ by making infinitesimal steps

- *Local Maximum*
: A point where $f(x)$ is higher than at all neighboring points so it is no longer possible to decrease $f(x)$ by making infinitesimal steps

- *Saddle Points*
: Critical points that are neither maxima nor minima

- *Global Minimum*
: The point that obtains the absolute lowest value of $f(x)$
    - There can be only one global minimum or multiple global minima of the function

- *Partial Derivatives*
: Measures how $f$ changes as only the variable $x_i$ increases at point $x$
    - $\frac{\delta}{\delta x_i}f(x)$

- *Gradient*
: Generalizes the notion of derivative to the case where the derivative is with respect to a vector
    - The gradient of $f$ is the vector containing all the partial derivatives, denoted $\nabla_x f(x)$
    - Element $i$ of the gradient is the partial derivative of $f$ with respect to $x_i$

- *Directional Derivative*
: The slope of the function $f$ in direction $u$
    - The derivative of the function $f(x + \alpha u)$ with respect to $\alpha$, evaluation at $\alpha = 0$
    - Using the chain rule, we can see that $\frac{\delta}{\delta\alpha}f(x + \alpha u)$ evaluates to $u^\intercal \nabla_x f(x)$ when $\alpha = 0$
    - To minimize $f$, we would like to find the direction in which $f$ decreases the fastest

- *Method of Steepest Descent (Gradient Descent)*
: The method of decreasing $f$ by moving in the direction of the negative gradient

- *Learning Rate*
: A positive scalar determining the size of the step
    - Popular approach to choosing this value is to set of a small constant

- *Line Search*
: Another approach to determining the learning rate is to evaluate $f(x - \epsilon \nabla_x f(x))$ for several values of $\epsilon$ and choose the one that results in the smallest objective function value

- *Hill Climbing*
: Ascending an objective function of discrete parameters

### Beyond the Gradient: Jacobian and Hessian Matrices

- *Jacobian Matrix*
: The matrix containing all the partial derivative vectors of a function whose input and output are both vectors
    - The Jacobian matrix is defined as $\textbf{J} \in \mathbb{R}^{n \times m} \text{of} \textbf{f}$ is defined such that $J_{i,j} = \frac{\delta}{\delta x_{j}}f(x)_i$


- *Second Derivative*
: The derivative of a derivative
    - Denoted as $\frac{\delta^2}{\delta x_u \delta x_j}$
    - Tells us whether a gradient step will cause as much of an improvement as we would expect based on the gradient alone

- *Curvature*
: What the second derivative tells us

- *Hessian Matrix*
: When the function has multiple input dimensions, the second derivatives can be collected together into this matrix
    - Defined such that $\textbf{H}(f)(x)_{i,j} = \frac{\delta^2}{\delta x_i \delta x_j} f(x)$
    - Equivalent to the Jacobian of the gradient

- *Second Derivative Test*
: When $f'(x) = 0 \text{ and } f''(x) > 0$, then $x$ is a local minimum, when $f'(x) = 0 \text{ and } f''(x) < 0$, then $x$ is a local maximum, when $f''(x) = 0$, the test is inconclusive

- *Newton's Method*
: Based on using a second-order Taylor series expansion to approximate $f(x)$ near some point $x^{(0)}$
    - $f(x) \approx f(x^{(0)}) + (x - x^{(0)})^\intercal \nabla_x f(x^{(0)}) + \frac{1}{2}(x - x^{(0)})^\intercal \textbf{H}(f)(x^{(0)})(x - x^{(0)})$
    - If we then solve for the critical point of this function, we obtain $x^* = x^{(0)} - \textbf{H}(f)(x^{(0)})^{-1} \nabla_x f(x^{(0)})$

- *First-Order Optimization Algorithm*
: Optimization algorithms that use only the gradient, such as gradient descent

- *Second-Order Optimization Algorithm*
: Optimization algorithms that also use the Hessian matrix, such as Newton's method

- *Lipschitz Continuous*
: A function $f$ whose rate of change is bounded by a Lipschitz constant $\mathcal{L}$
    - $\forall x, \forall y, \|f(x) - f(y)\| \le \mathcal{L} \|\| x - y \|\|_2$

- *Convex Optimization*
: Able to provide many more guarantees by making stronger restrictions
    - Application only to convex functions-functions for which the Hessian is positive semidefinite everwhere

## Constrained Optimization

- Sometimes we may wish not only to maximize or minimize a function $f(x)$ over all possible values of $x$, but also find the maximal or minimal value of $f(x)$ for values of $x$ in some set $\mathbb{S}$

- * Constrained Optimization*
: Find the maximal or minimal value of $f(x)$ for values of $x$ in some set $\mathbb{S}$

- *Feasibile*
: Points $x$ that lie within the set $\mathbb{S}$ in constrained optimization terminology

- *Karush-Kuhn-Tucker*
: Provides a very general solution to constrained optimization

- *Generalized Lagrangian (Generalized Lagrange Function)*
    - To define the Lagrangian, we first need to describe $\mathbb{S}$ in terms of equations and inequalities
    - $\mathbb{S} = {x \| \forall i, g^{(i)}(x) = 0 \text{ and } \forall j,h^{(j)}(x) \le 0}$
    - The generalized Lagrangian is: $L(x, \lambda, \alpha) = f(x) + \sum_i \lambda_i g^{(i)}(x) + \sum_j \alpha_j h^{(j)}(x)$

- *Equality Constraints*
: The equations involving $g^{(i)}$

- *Inequality Constraints*
: The equations involving $h^{(j)}$

- *Active*
: When $h^{(i)}(x^*) = 0$

## Example: Linear Least Squares