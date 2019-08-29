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
- *Symmetric Matrix*
    - A matrix $A \in \mathbb{R}^{n \times n}$ is symmetric if $A = A^T$
    - Only $(n, n)$-matrices can be symmetric or *square matrices*
    - If ***A*** is invertible, then so is $A^T$
        - $(A^{-1})^T = (A^T)^{-1} =: A^{-T}$
    - The sum of symmetric matrices $\textbf{A, B} \in \mathbb{R}^{n \times n}$ is always symmetric
    - The product of symmetric matrices is always defined but not always symmetric

### Multiplication by a Scalar

- When multiplying by a scalar:
    - $\textbf{A} \in \mathbb{R}^{m \times n}$ and $\lambda \in \mathbb{R}$, then $\lambda\textbf{A} = \textbf{K}$
        - $K_{ij} = \lambda a_{ij}$
    - $\lambda$ scales each element of $\textbf{A}$
- Associativity holds when multiplying by a scalar
    - $(\lambda\psi)C = \lambda(\psi C)$, when $C \in \mathbb{R}^{m \times n}$
    - $\lambda(BC) = (\lambda B)C = B(\lambda C) = (BC)\lambda$, when $\textbf{B} \in \mathbb{R}^{m \times n}, \textbf{C} \in \mathbb{R}^{n \times k}$
    - $(\lambda C)^T = C^T\lambda^T = C^T\lambda = \lambda C^T$ since $\lambda = \lambda^T$ for all $\lambda \in \mathbb{R}$
- Distributivity holds when multiplying by a scalar
    - $(\lambda + \psi)\textbf{C} = \lambda\textbf{C} + \psi\textbf{C}$, when $\textbf{C} \in \mathbb{R}^{m \times n}$
    - $\lambda(\textbf{B} + \textbf{C}) = \lambda\textbf{B} + \lambda\textbf{C}$, when $\textbf{B,C} \in \mathbb{R}^{m \times n}$

### Compact Representations of Systems of Linear Equations

- Consider the system of linear equations:

$$ 2x_1 + 3x_2 + 5x_3 = 1 \\
4x_1 - 2x_2 - 7x_3 = 8 \\
9x_1 + 5x_2 - 3x_3 = 2$$

- Using the rules for matrix multiplication, we can write this system in a compact form:

$$ \left[\begin{array}{ccc}
2 & 3 & 5 \\
4 & -2 & -7 \\
9 & 5 & -3
\end{array}\right]
\left[\begin{array}{ccc}
x_1 \\
x_2 \\
x_3
\end{array}\right] = 
\left[\begin{array}{ccc}
1 \\
8 \\
2
\end{array}\right]
$$

- $x_1$ scales the first column, $x_2$ scales the second, and $x_3$ scales the third
- Generally, a system of linear equations can be compactly represented in their matrix form as $\textit{\textbf{Ax}} = \textit{\textbf{b}}$

## Solving Systems of Linear Equations


### Particular and General Solution

$$\left[\begin{array}{ccc}
1 & 0 & 8 & -4 \\
0 & 1 & 2 & 12
\end{array}\right]

\left[\begin{array}{ccc}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{array}\right]

=

\left[\begin{array}{ccc}
42 \\
8
\end{array}\right]
$$


- In general, we want to find scalars $x_1,...,x_4$, such that $\sum_{i=1}^4x_ic_i = \textbf{b}$
    - $c_i$ is the *i*th column of the matrix
    - $\textbf{b} is the right-hand side (the vector the system is equal to)
- The general approach is:
    - Find a particular solution to $Ax = b$
    - Find all solutions to $Ax = 0$
    - Combine the solutions from steps 1, and 2 to the general solution
- The key to transforming matrices to be solvable is Gaussian elimination
    - Gaussian elemination transforms linear equations using elementary transformations
    - After the transformations, we can apply the three steps to the simple form


### Elementary Transformations

- *Elementary Transformations* are the key to solving a system of linear equations and effectively keep the solution set the same, but transform the equation system into a simpler form
    - Exchange of two equations (rows in a matrix representing the system)
    - Multiplication of an equation (row) with a constant $\lambda \in \mathbb{R}\\\{0}$
    - Addition of two equations (rows)
- *Row-Echelon Form*
    - All rows that contain only zeros are at the bottom of the matrix
    - All rows that contain at least one non-zero element are on top of rows that contain only zeros
    - Looking at non-zero rows only, the first non-zero number from the left (also called the *pivot* or the *leading coefficient*) is always strictly to the right of the pivot of the row above it
- *Basic Variable*: corresponds to the pivots in the row-echelon form
- *Free Variaible*: corresponds to the other variables that aren't the basic variables
- *Reduced Row-Echelon Form*
    - It is in row-echelon form
    - Every pivot is 1
    - The pivot is the only non-zero entry in its column
- *Gaussian Elimination*: is an algorithm that performs elementary transformations to bring a sytem of linear equations into reduced row-echelon form


### The Minus-1 Trick

- This trick assumes that matrix $\textit{\textbf{A}}$ is in reduced row-echelon form
- The matrix $\textit{\textbf{A}}$ is augmented to become an $n \times n$-augmented matrix $\tilde{A}$ by adding $n - k$ rows of the form $\left[\begin{array}{ccc}0 & ... & 0 & -1 & 0 & ... & 0\end{array}\right]$ so that the diagonals of the augmented matrix $\tilde{A}$ contain either 1 or -1
- The columns of $\tilde{A}$ that contain the -1 as pivots are the solutions of the linear equation system $Ax = $
- These columns form a basis of the solution space, or *kernel* or *null space*, for $Ax = 0$
- Example:

$$A = \left[\begin{array}{ccc}
1 & 3 & 0 & 0 & 3 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4
\end{array}\right]$$

The augmented matrix becomes:

$$\tilde{A} = \left[\begin{array}{ccc}
1 & 3 & 0 & 0 & 3 \\
0 & -1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4 \\
0 & 0 & 0 & 0 & -1
\end{array}\right]$$

From this form, we can read out the solutions of $Ax = 0$ by taking the columns of $\tilde{A}$, which contain -1 on the diagonal

$$\left\{x \in \mathbb{R}^5: x = \lambda_1 
\left[\begin{array}{ccc}
3 \\
-1 \\
0 \\
0 \\
0 \end{array}\right] + 
\lambda_2 \left[\begin{array}{ccc}
3 \\
0 \\
9 \\
-4 \\
-1 \end{array}\right] , \lambda_1, \lambda_2 \in \mathbb{R}\right\}$$

- *Calculating the Inverse*
    - To compute the inverse $A^{-1}$ of $\mathbb{R}^{n \times n}$, we need to find a matrix $X$ that satisfies $AX = I_n$
    - Then, $X = A^{-1}$
    - This can be writting as a set of simultaneous linear eauations $AX = I_n$, where we solve for $X = \left[x_1\|...\|x_n\right]$

### Algorithms for Solving a System of Linear Equations

- We may be able to determine the inverse $A^{-1}$ such that the solution of $Ax = b$ is given as $x = A^{-1}b$
    - Only possible if ***A*** is a square matrix and invertible
    - ***A*** needs to have linearly independent columns

$$Ax = b \longleftrightarrow A^TAx = A^Tb \longleftrightarrow x = (A^TA)^{-1}A^Tb$$

- The *Moore-Penrose pseudo-inverse* $(A^TA)^{-1}A^T$ can be used to determine the solution that solve $Ax = b$
    - This is also the mininum norm least-squares solution
- Gaussian elimination plays an important role in computing determinants, checking linear independence, computing the inverse, compurting the rank of a matrix, and determining the basis of a vector space
- In practice, systems of many linear equations are solved indirectly by either stationary iterative methods, such as the Richardson method, the Jacobi method, the GauB-Seidel method, and the successive over-relaxation method, or Krylov subspace methods, such as conjugate gradients, generalized minimal residual, or biconjugate gradients
    - Let $x_*$ be a solution of $Ax = b$
    - The key idea of iterative methods is to set up an iteration of the form $x^{k+1} = Cx^k + d$ for a suitable $C$ and $d$ that reduces the residual error $\Vert x^{k + 1} - x_* \Vert$ in every iteration and convergers to $x_*$


## Vector Spaces

- A set of elements and an operation defined on these elements that keeps some structure of the set intact

### Groups

- Consider a set $\mathcal{G}$ and an operation $\otimes$: $\mathcal{G} \times \mathcal{G} \rightarrow \mathcal{G}$ defined on $\mathcal{G}$
    - Then $\mathcal{G} := (\mathcal{G}, \otimes)$ is called a group if the following hold:
        - Closure of $\mathcal{G}$ under $\otimes$: $\forall x,y \in \mathcal{G}: x \otimes y \in \mathcal{G}$
            - The closure of group G under tensor product: for all x, y in group G: x tensor product y is in group G
        - Associativity: $\forall x,y,z \in \mathcal{G} : (x \otimes y) \otimes z = x \otimes (y \otimes z)$
            - For all
        - Neutral element: $\existse \in \mathcal{G} \forall x \in \mathcal{G}: x \otimes e = x$ and $e \otimes x = x$


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