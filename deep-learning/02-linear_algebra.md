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

**Origin**
: The point specified by the vector of all zeros

- To analyze how many solutions the equation has, the columns of $\textbf{A}$ can be thought of as different directions we can travel in from the origin, then determine how many ways there are of reach $\textbf{b}$. Each element of $\textbf{x}$ specifies how far we should travel in each of these directions with $\textit{x}_i$ specifying how far to move in the direction of column *i*

$$\textbf{Ax} = \sum_i \textit{x}_i \textbf{A}_{:,i}$$

**Linear Combination**
: Some set of vectors $\{v^{(1)},...,v^{(n)}}$ given by multiplying each vector $\textbf{v}^{(i)}$ by a corresponding scalar coefficient and adding the results
    
    $$\sum_i\textit{c}_i\textbf{v}^{(i)}$$

**Span**
: The set of all points obtainable by linear combination of the original vectors

**Column Space (Range)**
: Testing whether $\textbf{b}$ is in the span of columns of $\textbf{A}$
    - In order for the system $\textbf{Ax = b}$ to have a solution for all values of $\textbf{b} \in \mathbb{R}^m$, it is required that that column space of $\textbf{A}$ be all of $\mathbb{R}^m$

**Linear Dependence**
: When the column space is just a line and fails to encompass all of $\mathb{R}^2$ even when there are two columns

**Linearly Independent**
: When no vector in the set is a linear combination of the other vectors

- For $\textbf{Ax = b}$ to have a solution for every value of $\textbf{b}$, the column space of the matrix must contain at least one set of *m* linearly independent columns
- For a matrix to have an inverse, we need to ensure that $\textbf{Ax = b}$ has at most one solution for each value of $\textbf{b}$

**Square Matrix**
:

**Singular**
: A square matrix with linearly independent columns

## Norms

**Norm**
:

**Euclidean Norm**
:

**Max Norm**
:

**Frobenius Norm**
:

## Special Kinds of Matrices and Vectors

**Diagonal Matrices**
:

**Symmetric Matrix**
:

**Unit Vector**
:

**Unit Norm**
:

**Orthogonal Vectors**
:

**Orthonormal Vectors**
:

**Orthogonal Matrix**
:

## Eigendecomposition

**Eigendecomposition**
:

**Eigenvector**
:

**Eigenvalue**
:

**Decompose**
:

## Singular Value Decomposition

**Singular Value Decomposition**
:

**Singular Vectors**
:

**Singular Values**
:

**Left-Singular Vectors**
:

**Right-Singular Vectors**
:

## The Moore-Penrose Pseudoinverse

## The Trace Operator

## The Determinant

## Example: Principal Components Analysis (PCA)