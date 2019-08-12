---
template: default
use_math: true
---

# Linear Algebra

- *Algebra* is a common approach to formalizing intuitive concepts by constructing a set of objects (symbols) and a set of rules to manipulate these objects
- *Linear Algebra* is the study of vectors and certain rules to manipulate vectors
- Types of Vector Objects:
    - *Geometric Vectors*: directed segments which can be drawn
        - Can be added to yield another geometric vector
        - Can be multiplied to yield another geometric vector
    - *Polynomials*:
        - Can be added together to yield another polynomial
        - Can be multiplied by a scalar to yield another polynomial
    - *Audio signals*: represented as a series of numbers
        - Can be added together to yield another audio signal
        - Can be multiplied by a scalar to yield another audio signal
    - *Elements of $\mathbb{R}^n$

## Systems of Linear Equations

- General form of a *system of linear equations*: $a_{m1}x_1 + ... + a_{mn}x_n = b_m$
    - $a_{ij} \in \mathbb{R}$ and $b_i \in \mathbb{R}$
    - $x_1,...,x_n$ are the *unknwons* in the system
    - Every *n*-tuple $(x_1,...,x_n) \in \mathbb{R}^n$ that satisfies the equation is a *solution* of the linear equation system
- For any real-valued system of linear equations, there will either be no solutions, exactly one, or infinitely many solutions
- For a compact notation, the coefficients $a_{ij}$ are collected into vectors and the vectors into matrices:
$$x_1 \left[\begin{array}{ccc}a_{11} \\ \vdots \\a_{m1}\end{array}\right] + x_2 \left[\begin{array}{ccc}a_{12} \\ \vdots \\a_{m2}\end{array}\right] + ... + x_n \left[\begin{array}{ccc}a_{1n} \\ \vdots \\a_{mn}\end{array}\right] = \left[\begin{array}{c} b_1 \\ \vdots \\ b_m \end{array}\right]$$


## Matrices

 - Can be used to compactly represent systems of linear equations
 - Can also represent linear functions (linear mappings)
 - By convention (1, n)-matrices are called rows
 - (m, 1)-matrices are called columns
 - $\mathbb{R}^{m \times n}$ is the set of all real-valued (*m, n*)-matrices

### Matrix Addition and Multiplication

- The sum of two matrices $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{m \times n}$ is defined as the element-wise sum

$$ A + B := \left[\begin{array}{ccc} a_{11} + b_{11} & ... & a_{1n} + b_{1n} \\ \vdots & & \vdots \\ a_{m1} + b_{m1} & ... & a_{mn} + b_{mn}\end{array}\right] \in \mathbb{R}^{m \times n}$$

- The product of two matrices $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n * k}$ the elements $c_{ij}$ of the product $C = AB \in \mathbb{R}^{m * k}$ are defined as
    - To compute element $c_{ij}$, we multiply the elements of the *i*th row of **A** with the *j*th column of **B** and then sum them up
    - This is called the *dot product*
    - Matrix multiplication can only occur if the column dimensions of the left matrix match the row dimensions of the right matrix

$$ c_{ij} = \sum_{l = 1}^{n}a_{il}b_{lj}, i = 1,...,m, j = 1,...,k$$

- *Identity Matrix*: the $n * n$-matrix containing 1 on the diagonal and 0 everywhere else

$$ I_n := \left[\begin{array}{ccc} 1 & 0 & ... & 0 & ... & 0 \\ 0 & 1 & ... & 0 & ... & 0 \\ \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\ 0 & 0 & ... & 1 & ... & 0 \\ \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\ 0 & 0 & ... & 0 & ... & 1 \end{array}\right]$$

#### Properties of Matrices

- *Associativity*
    - $\forall A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}, C \in \mathbb{R}^{p \times q}: (AB)C = A(BC)$
    - For all A in Real Number space (m \times n), B in Real Number space (n \times p), C in Real Number space (p * q): $(AB)C = A(BC)$
- *Distributivity*
    - $\forall A, B \in \mathbb{R}^{m \times n}, C, D \in \mathbb{R}^{n \times p}: (A + B)C = AC + BC$
    - For all A, B in Real Number space (m \times n), C, D in Real Number space (n \times p): (A + B)C = AC + BC
- *Identity Multiplication*
    - $\forall A \in \mathbb{R}^{m \times n}: I_mA = AI_n = A$
        - $I_m \ne I_n$ for (m \ne n)
    - For all A in Real Number space (m \times n): (I_mA = AI_n = A)


### Inverse and Transpose

- *Inverse*
    - The *inverse* is denoted by $A^{-1}$
    - $AB = I_n = BA$
        - ***B*** is the inverse of ***A***
    - Not every matrix $A$ possesses an inverse $A^{-1}$
    - If an inverse exists, the matrix is called *regular/invertible/nonsingular*
    - If no inverse exists, the matrix is called *singular/noninvertible*
- *Transpose*
    - The *transpose* is denoted as $A^T$
    - For $A \in \mathbb{R}^{m \times n}$, the matrix $B \in \mathbb{R}^{n \times m}$ with $b_{ij} = a_{ji}$ is called the *transpose* of A
        - $B = A^T$
- Properties of Inverses and Traponses
    - Inverses
        - $AA^{-1} = I = A^{-1}A$
        - $(AB)^{-1} = B^{-1}A^{-1}
        - $(A + B)^{-1} \ne A^{-1} + B^{-1}$
    - Transposes
        - $(A^T)^T = A$
        - $(A + B)^T = A^T + B^T$
        - $(AB)^T = B^TA^T$

### Multiplication by a Scalar



### Compact Representations of Systems of Linear Equations



## Solving Systems of Linear Equations



### Particular and General Solution



### Elementary Transformations



### The Minus-1 Trick



### Algorithms for Solving a System of Linear Equations



## Vector Spaces



### Groups



### Vector Spaces



### Vector Subspaces



## Linear Independence



## Basis and Rank



### Generating Set and Basis



### Rank



## Linear Mappings



### Matrix Representation of Linear Mappings



### Basis Change



### Image and Kernel



## 2.8 - Affine Spaces



### 2.8.1 - Affice Subspaces



### 2.8.2 - Affice Mappings