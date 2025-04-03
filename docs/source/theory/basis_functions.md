# Basis Functions Theory

This document provides the mathematical theory behind the various basis functions used in Kolmogorov-Arnold Networks.

## Introduction to Basis Functions

Basis functions are a set of functions that can be linearly combined to approximate other functions within a specific function space. In the context of KANs, basis functions are used to represent the single-variable functions that appear in the Kolmogorov-Arnold representation theorem.

The choice of basis functions significantly affects the network's ability to approximate different types of functions efficiently. This library supports several families of basis functions, each with specific properties and domains of effectiveness.

## Orthogonal Polynomials

Orthogonal polynomials form an important class of basis functions used in KANs. A set of polynomials $\{P_n(x)\}$ is orthogonal with respect to a weight function $w(x)$ on an interval $[a,b]$ if:

$$\int_a^b P_m(x) P_n(x) w(x) dx = 0 \quad \text{for} \quad m \neq n$$

This orthogonality property makes these polynomials particularly useful for function approximation. The library supports several families of orthogonal polynomials:

### Chebyshev Polynomials (First Kind)

Chebyshev polynomials of the first kind, denoted $T_n(x)$, are orthogonal with respect to the weight function $w(x) = 1/\sqrt{1-x^2}$ on the interval $[-1, 1]$.

They are defined by the recurrence relation:
$$T_0(x) = 1$$
$$T_1(x) = x$$
$$T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)$$

Chebyshev polynomials minimize the maximum error in approximation, making them excellent for approximating functions on bounded intervals.

### Chebyshev Polynomials (Second Kind)

Chebyshev polynomials of the second kind, denoted $U_n(x)$, are orthogonal with respect to the weight function $w(x) = \sqrt{1-x^2}$ on the interval $[-1, 1]$.

They are defined by the recurrence relation:
$$U_0(x) = 1$$
$$U_1(x) = 2x$$
$$U_{n+1}(x) = 2xU_n(x) - U_{n-1}(x)$$

### Legendre Polynomials

Legendre polynomials, denoted $P_n(x)$, are orthogonal with respect to the weight function $w(x) = 1$ on the interval $[-1, 1]$.

They are defined by the recurrence relation:
$$P_0(x) = 1$$
$$P_1(x) = x$$
$$P_{n+1}(x) = \frac{(2n+1)xP_n(x) - nP_{n-1}(x)}{n+1}$$

Legendre polynomials are particularly useful for problems with uniform weighting across the input domain.

### Gegenbauer Polynomials

Gegenbauer polynomials (also known as ultraspherical polynomials), denoted $C_n^{(\alpha)}(x)$, are orthogonal with respect to the weight function $w(x) = (1-x^2)^{\alpha-1/2}$ on the interval $[-1, 1]$, where $\alpha > -1/2$.

They are defined by the recurrence relation:
$$C_0^{(\alpha)}(x) = 1$$
$$C_1^{(\alpha)}(x) = 2\alpha x$$
$$C_{n+1}^{(\alpha)}(x) = \frac{2(n+\alpha)xC_n^{(\alpha)}(x) - (n+2\alpha-1)C_{n-1}^{(\alpha)}(x)}{n+1}$$

Gegenbauer polynomials generalize both Chebyshev and Legendre polynomials: $C_n^{(0)}(x)$ are related to Chebyshev polynomials of the first kind, and $C_n^{(1/2)}(x)$ are proportional to Legendre polynomials.

### Hermite Polynomials

Hermite polynomials, denoted $H_n(x)$, are orthogonal with respect to the weight function $w(x) = e^{-x^2}$ on the interval $(-\infty, \infty)$.

They are defined by the recurrence relation:
$$H_0(x) = 1$$
$$H_1(x) = 2x$$
$$H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x)$$

Hermite polynomials are particularly useful for approximating functions on unbounded domains, especially those with Gaussian-like behavior.

### Laguerre Polynomials

Laguerre polynomials, denoted $L_n(x)$, are orthogonal with respect to the weight function $w(x) = e^{-x}$ on the interval $[0, \infty)$.

They are defined by the recurrence relation:
$$L_0(x) = 1$$
$$L_1(x) = 1 - x$$
$$L_{n+1}(x) = \frac{(2n+1-x)L_n(x) - nL_{n-1}(x)}{n+1}$$

Laguerre polynomials are particularly useful for approximating functions on semi-infinite domains.

### Jacobi Polynomials

Jacobi polynomials, denoted $P_n^{(\alpha,\beta)}(x)$, are orthogonal with respect to the weight function $w(x) = (1-x)^\alpha(1+x)^\beta$ on the interval $[-1, 1]$, where $\alpha, \beta > -1$.

They generalize many other orthogonal polynomials:
- $P_n^{(0,0)}(x)$ are Legendre polynomials
- $P_n^{(\alpha,\alpha)}(x)$ are closely related to Gegenbauer polynomials
- $P_n^{(-1/2,-1/2)}(x)$ are related to Chebyshev polynomials of the first kind
- $P_n^{(1/2,1/2)}(x)$ are related to Chebyshev polynomials of the second kind

## Fourier Basis

The Fourier basis uses sine and cosine functions to represent periodic functions. For a function on $[-\pi, \pi]$, the Fourier basis consists of:
$$\{1, \cos(x), \sin(x), \cos(2x), \sin(2x), \ldots, \cos(nx), \sin(nx), \ldots\}$$

This basis is particularly effective for approximating periodic functions and is implemented in the `FourierBasis` class.

## B-Splines

B-splines (basis splines) are piecewise polynomial functions with compact support. They provide local control, meaning changes to a single control point affect only a small region of the function.

A B-spline of degree $k$ is defined recursively:

$$B_{i,0}(x) = \begin{cases} 1 & \text{if } t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$B_{i,k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i,k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(x)$$

where $t_i$ are knot values that define the domain partition.

B-splines are particularly useful for approximating functions with local features or discontinuities.

## Radial Basis Functions

Radial Basis Functions (RBFs) are functions whose value depends only on the distance from a center point. The Gaussian RBF is a common choice:

$$\phi(x) = \exp\left(-\frac{\|x-c\|^2}{2\sigma^2}\right)$$

where $c$ is the center and $\sigma$ controls the width.

RBFs provide local approximation capabilities and are particularly useful for scattered data interpolation.

## Choosing the Right Basis Functions

The choice of basis functions should be guided by the properties of the function being approximated:

1. **Domain characteristics**: 
   - Bounded domain: Chebyshev, Legendre, or Jacobi polynomials
   - Unbounded domain: Hermite polynomials
   - Semi-infinite domain: Laguerre polynomials
   - Periodic functions: Fourier basis

2. **Smoothness**:
   - Smooth functions: Orthogonal polynomials
   - Functions with discontinuities or local features: B-splines or RBFs

3. **Approximation objectives**:
   - Minimizing maximum error: Chebyshev polynomials
   - Uniform approximation: Legendre polynomials
   - Local control: B-splines or RBFs

The PyTorch KAN library provides implementations of all these basis functions, allowing users to choose the most appropriate one for their specific problem.