---
template: default
use_math: true
---

# Linear Algebra

## Scalars, Vectors, Matrices, and Tensors

- **Scalars**
: A single number
- Written in italics with lowercase variables names
- Can be integers, real numbers, rational numbers, etc.
- Examples: $\textit{a, n, x}$

- **Vectors**
: An 1-D array of numbers  
    - Vectors are given lowercase variable names in bold typeface
    - Can be real, binary, integer, etc
    - Elements of the vector are identified by writing its name in italic typeface with a subscript
        - The first element of $\textbf{x}$ is $\textit{x}_1$
    - Example:
    $$x = \left[\begin{array}{ccc}x_1 \\ x_2 \\ \vdots \\ x_n \end{array}\right]$$
    - Can be thought of as matrices that contain only one column therefore the transpose of a vector is a matrix with only one row

- **Matrices**
: A 2-D array of numbers  
    - Matrices are given uppercase variable names with bold typeface, such as $\textbf{A}$
    - Elements of a matrix are identified using its name in talic but not bold font, and the indices are listed with separating commas
        - The first element \(upper left\) of matrix $\textbf{A}$ is identified by $\textit{A}_{1,1}$
    - Example:
    $$\textbf{A} = \left[\begin{array}{ccc} 
    \textit{A}_{1,1} & \textit{A}_{1,2} \\
    \textit{A}_{2,1} & \textit{A}_{2,2}
    \end{array}\right]$$

- **Tensors**
: An array of numbers arranged on a regular grid with a variable number of axes
    - May have:
        - Zero dimensions and be a scalar
        - One dimension and be a vector
        - Two dimensions are be a matrix
        - Or more dimensions
    - Tensors are given uppercase variables names with Bold Sans Serif typeface, such as $\mathsf{A}$
    - To identify the element of $\mathsf{A}$ at coordinates $(\textit{i,j,k})$, it is written as $\mathsf{A}_{i,j,k}$

- **Transpose**
: Is the mirror image of the matrix across a diagonal line called the main diagonal
    - The transpose of matrix $\textbf{A}$ is denoted as $\textbf{A}^T$ and is defined such that $\left(\textbf{A}^T\right)\_{i,j} = \textbf{A}_{j,i}$

- **Main Diagonal**
: The line of a matrix running down and to the right starting from its upper left corner

- Matrices can be added to each other as long as they have the same shape
    - $\textbf{C} = \textbf{A} + \textbf{B}$ where $\textit{C}\_{i,j} = \textit{A}\_{i,j} + \textit{B}\_{i,j}$
- We can also add a scalar to a matrix or multiply a matrix by a scalar by performing that operation on each element of a matrix
    - $\textbf{D} = \textit{a} \cdot \textbf{B} + \textit{c}$ where $\textit{D}\_{i,j} = \textit{a} \cdot \textit{B}\_{i,j} + \textit{c}$
- A matrix and a vector can also be added together to yield another matrix
    - $\textbf{C} = \textbf{A} + \textbf{b}$ where $\textit{C}\_{i,j} = \textit{A}\_{i,j} + \textit{b}\_j$

- **Broadcasting**
: The implicit copying of **b** to many locations

## Multiplying Matrices and Vectors

- **Matrix Product**
: The result of when two matrices are multiplied together
    - $\textbf{A}$ must have the same number of columns as $\textbf{B}$ has rows
    - If $\textbf{A}$ is of shape $\textit{m} \times \textit{n}$ and $\textbf{B}$ is of shape $\textit{n} \times \textit{p}$, then $\textbf{C}$ is of shape $\textit{m} \times \textit{p}$
    - Can be written as: $\textbf{C} = \textbf{AB}$
    - The product operation is defined by:

    $$\textit{C}_{i,j} = \sum_k \textit{A}_{i,k}\textit{B}_{k,j}$$

- **Element-wise Product (Hadamard Product)**
: A matrix contraining the product of the individual elements of two matrices multiplied by each other
    - Denoted $\textbf{A} \odot \textbf{B}$

- **Dot Product**
    - The dot product between two vectors $\textbf{x}$ and $\textbf{y}$ or the same dimensions is the matrix product $\textbf{x}^T\textbf{y}$

- Properties of Matrices
    - Distributive Property: $\textbf{A(B + C) = AB + AC}$
    - Associative Property: $\textbf{A(BC) = (AB)C}$
- Properties of Vectors
    - Commutative Property: $\textbf{x}^T\textbf{y} = \textbf{y}^T\textbf{x}$

## Identity and Inverse Matrices

- **Matrix Inversion**
: Powerful tool that enables us to analytically solve a system of equations

- **Identity Matrix**
: A matrix that does not change any vector when we multiply that vector by that matrix
    - Denoted by $\textbf{I}_n$, formally $\textbf{I}_n \in \mathbb{R}^{n \times n}$
    - For all vector **x** in the real number space, **x** multiplied by the identity matrix yields **x**

    $$\forall \textbf{x} \in \mathbb{R}^n, \textbf{I}_n\textbf{x} = \textbf{x}$$


- **Matrix Inverse**
: When multiplied by a matrix yields the identity matrix
    - Denoted as $\textbf{A}^{-1}$ and is defined as the matrix such that:

    $$\textbf{A}^{-1}\textbf{A} = \textbf{I}_n$$

    - Can be used to solve equations by (dependent upon $A^{-1} existing):

    $$\textbf{Ax = b} \\
    \textbf{A}^{-1}\textbf{Ax} = \textbf{A}^{-1}\textbf{b} \\
    \textbf{I}_nx = \textbf{A}^{-1}b \\
    \textbf{x} = \textbf{A}^{-1}b$$

## Linear Dependence and Span

- For $\textbf{A}^{-1}$ to exist, $\textbf{Ax = b}$ must have one solution for every value of $\textbf{b}$
- It is only possible for a system of equations to have no solutions, infitely many solutions, or one solution for every value of $\textbf{b}$
- If both $\textbf{x}$ and $\textbf{y}$ are solutions then $ \textbf{z} = \alpha\textbf{x} + (1 - \alpha)\textbf{y}$

- **Origin**
: The point specified by the vector of all zeros

- To analyze how many solutions the equation has, the columns of $\textbf{A}$ can be thought of as different directions we can travel in from the origin, then determine how many ways there are of reach $\textbf{b}$. Each element of $\textbf{x}$ specifies how far we should travel in each of these directions with $\textit{x}_i$ specifying how far to move in the direction of column *i*

$$\textbf{Ax} = \sum_i \textit{x}_i \textbf{A}_{:,i}$$

- **Linear Combination**
: Some set of vectors $\{v^{(1)},...,v^{(n)}}$ given by multiplying each vector $\textbf{v}^{(i)}$ by a corresponding scalar coefficient and adding the results
    
    $$\sum_i\textit{c}_i\textbf{v}^{(i)}$$

- **Span**
: The set of all points obtainable by linear combination of the original vectors

- **Column Space (Range)**
: Testing whether $\textbf{b}$ is in the span of columns of $\textbf{A}$
    - In order for the system $\textbf{Ax = b}$ to have a solution for all values of $\textbf{b} \in \mathbb{R}^m$, it is required that that column space of $\textbf{A}$ be all of $\mathbb{R}^m$

- **Linear Dependence**
: When the column space is just a line and fails to encompass all of $\mathb{R}^2$ even when there are two columns

- **Linearly Independent**
: When no vector in the set is a linear combination of the other vectors

- For $\textbf{Ax = b}$ to have a solution for every value of $\textbf{b}$, the column space of the matrix must contain at least one set of *m* linearly independent columns
- For a matrix to have an inverse, we need to ensure that $\textbf{Ax = b}$ has at most one solution for each value of $\textbf{b}$

**Square Matrix**
:

- **Singular**
: A square matrix with linearly independent columns

## Norms

- **Norm**
: A function used to measure the size of vectors
    - Formally, for $p \in \mathbb{R}, p \ge 1$, the $\textit{L}^p$ norm is given by : 

    $$||\textbf{x}|| = \left(\sum_i|x_i|^p\right)^{\frac{1}{p}}$$

    - Norms are functions mapping vectors to non-negative values
    - The norm of a vector $\textit{x}$ measures the distance from the origin to the point $\textit{x}$
    - Norms are any functions *f* that satisfy the following proprties:
        - $f(x) = 0 \Rightarrow \textit{x} = 0$
        - $f(x + y) \le f(x) + f(y)$ (the triangle inequality)
        - $\forall\alpha \in \mathbb{R}, f(\alpha x) = \|\alpha\|f(x)$

- **Euclidean Norm**
: This is the $L^2$ norm which is the Euclidean distance from the origin to the point identified by $\textit{x}$
    - The size of a vector can be measured using $\textit{x}^Tx$
    - The squared $L^2$ norm is more convenient to work with mathematically and computationally because each derivative of the squared $L^2$ norm with respect to each element of $x$ depends only on the corresponding element of $x$, while all the derivatives of the $L^2$ norm depend on the entire vector

- The $L^1$ norm is used when it is important to discriminate between elements that are exactly zero and elements that are small but nonzero
    - The $L^1$ norm grows at the same rate in all locations but retains mathematical simplicity

    $$||\textbf{x}||_1 = \sum_i|x_i|$$

- **Max Norm**
: The $L^\infty$ simplifies to the absolute value of the element with the largest magnitude in the vector

    $$||\textbf{x}||_\infty = \max_i |x_i|$$

- **Frobenius Norm**
: The most common way to measure the size of a matrix

    $$ ||A ||_F = \sqrt{\sum_{i,j} A^2_{i,j}} $$

    - This is analogous to the $L^2$ norm of a vector

- The dot product of two vectors can be rewritten in terms of norms:

$$ x^Ty = ||x||_2 ||y||_2 \cos \theta $$

## Special Kinds of Matrices and Vectors

- **Diagonal Matrices**
: Consist mostly of zeros and have nonzero entries only along the main diagonal
    - Formally, a matrix $\textbf{D}$ is diagonal if and only if $\textit{D}_{i,j} = 0$ for all $i \ne j$
    - Not all diagonal matrices need to be square but rectangular diagonal matrices do not have inverses

- **Symmetric Matrix**
: Any matrix that is equal to its own transpose
    - $\textbf{A} = \textbf{A}^T$

- **Unit Vector**
: A vector with a unit norm
    - $\|\|x\|\|_{2} = 1$

**Unit Norm**
:

- **Orthogonal Vectors**
    - A vector $\textbf{x}$ and vector $\textbf{y}$ are orthogonal to each other if $x^Ty = 0$
    - If both vectors have a nonzero norm, this means that they are at a 90 degree angle to each other

- **Orthonormal Vectors**
: Vectors that are not only orthogonal but also have unit norms

- **Orthogonal Matrix**
: A square matrix whose rows are mutually orthonormal and whose columns are mutually orthonormal
    - $\textbf{A}^T\textbf{A} = \textbf{AA}^T = \textbf{I}$ 

## Eigendecomposition

- **Eigendecomposition**
: The process of decomposing a matrix into a set of eigenvectors and eigenvalues
    - Given by $\textbf{A} = \textbf{V}\text{diag}\(\lambda\)\textbf{V}^{-1}$

- **Eigenvector**
: A nonzero vector $\textbf{v}$ such that when you multiply it by a matrix, the matrix is only altered by the scale of $\textbf{v}$
    - $\textbf{Av} = \lambda\textbf{v}$
    - The scalar $\lambda$ is known as the eigenvalue corresponding to the eigenvector

- **Eigenvalue**
: The scalar that is the eigenvalue of the corresponding eigenvector

- **Decompose**
: Split matrices into their respective eigenvalues and eigenvectors

- Every real symmetric matrix can be decomposed into an expression using only real-valued eigenvectors and eigenvalues: $\textbf{A} = Q \Lambda Q^T$
    - **Q** is an orthogonal matrix composed of eigenvectors of A
    - $\Lambda$ is a diagonal matrix

- The eigendecomposition of a matrix tells us:
    - The matrix is singular if and only if any of the eigenvalues are zero

- **Positive Definite**
: A matrix whose eigenvalues are all positive

- **Positive Semidefinite**
: A matrix whose eigenvalues are all positive or zero
    - Guarantee that $\forall x, x^T\textbf{A}x \ge 0$

- **Negative Definite**
: A matrix whose eigenvalues are all negative

- **Negative Semidefinite**
: A matrix whose eigenvalues are all negative or zero
    - Guarantee that $x^T\textbf{A}x = 0 \Rightarrow x = 0$

## Singular Value Decomposition

**Singular Value Decomposition (SVD)**
: Provides a way to factorize a matrix into singular vectors and singular values
    - SVD provides the same information eigendecomposition revealed but is more generally applicable
    - Written as: $\textbf{A} = \textbf{UDV}^T$
        - $\textbf{U}$ and $\textbf{V}$ are both orthogonal matrices
        - Matrix $\textbf{D}$ is a diagonal matrix but not necessarily square

**Singular Values**
: The elements along the diagonal of matrix $\textbf{D}$

**Left-Singular Vectors**
: The columns of matrix $\textbf{U}$

**Right-Singular Vectors**
: The columns of matrix $\textbf{V}$

- The singular value decomposition of ***A*** can be interpreted in terms of the eigendecomposition of functions of ***A***
    - The left-singular vectors of ***A*** are the eigenvectors of $\textbf{AA}^T$
    - The right-singular vectors of ***A*** are the eigenvectors of $\textbf{A}^T\textbf{A}$

## The Moore-Penrose Pseudoinverse

- Matrix inversion is not defined for matrices that are not square
- The **Moore-Penrose pseudoinverse** can be used in cases when:
    - Matrix ***A** is taller than it is wide, hence it is possible for the equation $\textbf{x = By}$
    - Matrix ***A*** is wider than it is tall, hence there could be multiple possible solutions for the equation $\textbf{x = By}$
- The pseudoinverse of ***A*** is defined as a matrix:

$$\textbf{A}^+ = \lim_{\alpha \to 0} (\textbf{A}^T\textbf{A} + \alpha \textbf{I})^{-1}\textbf{A}^T$$

- Practical algorithms for computing the pseudoinverse are based on:

$$\textbf{A}^+ = \textbf{VD}^+\textbf{U}^T$$

- $\textbf{U, D, V}$ are the singular value decomposition of $\textbf{A}$ and the pseudoinverse of $\textbf{D}^+$ of diagonal matrix $\textbf{D}$ is obtained by taking the receiprocal of its nonzero elements then taking the transpose of the resulting matrix

## The Trace Operator

- Gives the sum of all the diagonal entries of a matrix

$$Tr(\textbf{A}) = \sum_i \textbf{A}_{i,j}$$

## The Determinant

- **Determinant**
: A function that maps matrices to real scalars and is equal to the product of all the eigenvalues of the matrix
    - If the determinant is 0, then space is contracted completely along at least one dimension causing it to lose all its volume
    - If the determinant is 1, then the transformation preserves volume

## Example: Principal Components Analysis (PCA)

- PCA is defined by our choice of the decoding function
- We will want to find some encoding function the produces the code for an input, $f(x) = c$ and a decoding function that produces the reconstructed input given its code $x \approx g(f(x))$
- The first thing that needs to be done in PCA is to figure out how to generate the optimal code point $c^*$ for each input point $x$
    - One way to do this is to minimize the distance between the input point $x$ and its reconstruction $g(c^*)$
    - We can measure the distance using the $L^2$ norm

    $$c^* = \text{arg}_c\text{min}||x - g(c)||_2$$

    - We can switch to the squared $L^2$ norm because both are minimized by the same value of $c$

    $$c^* = \text{arg}_c\text{min}||x - g(c)||^2_2$$

    - The function being minimized simplifies to

    $$(x - g(c))^T(x - g(c))$$