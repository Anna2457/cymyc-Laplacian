# Computing the Eigenvalue Crossings and Attractor Points for the Quintic Threefold with cymyc
 
**Anna Geho, PHYS 795 Spring 2026**
 
---
 
## 1. Introduction
 
This project investigates the eigenvalue spectrum of the Laplacian on the Calabi-Yau quintic threefold. Although spectrum contains fundamental information about string compactification on CYs, it cannot be found analytically for the quintic. However, these quantities can be computed analytically for the lower-dimensional torus, yielding insights into the quintic case. Ahmed and Ruehle's 2023 paper "Level Crossings, Attractor Points and Complex Multiplication" (arXiv:2304.00027) studies the complex structure moduli dependence of the Laplacian spectrum on the torus, showing that eigenvalue crossings, where two eigenvalue branches meet, seem to be connected to complex multiplication (CM) and attractor points on tori. These results concerning the attractor points and eigenvalue crossings are then extended using numerical techniques to the quintic. 
 
The goal of this project is to reproduce Figure 8 of Ahmed and Ruehle's paper, illustrating the $\psi$ values at which the eigenvalues cross and whether they are also attractor points. The numerical implementation is done in the cymyc framework, which is a JAX-based library that can numerically approximate the Ricci-flat CY metric with neural networks. 
 
## 2. Background on Ahmed and Ruehle
 
### 2.1 Eigenvalue Crossings
 
Ahmed and Ruehle's paper [1] begins with noting that eigenmodes can become lighter or heavier, leading to eigenvalue crossings. These points are also significant in number theory and the study of BPS black holes, and the paper investigates these points for the torus, the quartic K3, and the quintic threefold:
 
**Torus.** For the one-parameter family of complex tori $z_0^3 + z_1^3 + z_2^3  - 3\psi \ z_0 z_1 z_2 = 0$, the scalar Laplacian spectrum is known exactly. Eigenvalue crossings occur at values of $\tau$ where the torus has complex multiplication (CM). This is where $\tau$ is a quadratic irrationality satisfying $a\tau^2 + b\tau + c = 0$ with $a, b, c \in \mathbb{Z}$. The CM points also are attractor points.
 
**Quartic K3.** The CM of a quartic hypersurface in $\mathbb{P}^3$, defined by $z_0^4 + z_1^4 + z_2^4 + z_3^4 - 4\psi \, z_0 z_1 z_2 z_3 = 0$, is where the Picard rank is 20 instead of the general 19. Ahmed and Ruehle computed the scalar Laplacian spectrum numerically over different $\psi$ values and showed that the eigenvalues cross at the CM.
 
**Quintic threefold.** Lastly, the concept of the CM is not well defined mathematically for the CY quintic threefold, defined as $z_0^5 + z_1^5 + z_2^5 + z_3^5 + z_4^5 - 5\psi \, z_0 z_1 z_2 z_3 z_4 = 0$. However, there is numerical evicence that the eigenvalue crossings correspond to rank-one attractor points related to BPS black holes (in other words, points in moduli space where the central charge $|Z(q, \Pi(\psi))|$ is minimized for some integer charge vector $q$).
 
### 2.2 The Attractor Mechanism
 
In four-dimensional $\mathcal{N}=2$ theories resulting from type IIB string theory compactified on a Calabi-Yau threefold $X$, BPS black holes carry charges $q = (p^0, p^1, q_0, q_1) \in H^3(X, \mathbb{Z})$ under the vector multiplet fields. The complex structure moduli on the CY manifold are determined based on the charges. The attractor points minimize the BPS mass $|Z|^2 / e^{-K}$, with $Z = q \cdot \Pi$ being the central charge and $K$ being the Kähler potential on the CY.
 
## 3. Mathematics of the Scalar Laplacian
 
### 3.1 Periods of the Quintic
 
The periods of the holomorphic 3-form $\Omega$, whcih are $\Pi = (X^0, X^1, F_0, F_1)^T$, satisfy the Picard-Fuchs equation
``` math
\left[\theta^4 - 5^5 z \,(\theta + \tfrac{1}{5})(\theta + \tfrac{2}{5})(\theta + \tfrac{3}{5})(\theta + \tfrac{4}{5})\right] f(z) = 0
```
where $\theta$ depends on $\psi$. The periods depend on the solutions to the Frobenius equation and a matrix $M$ encoding Euler characteristic and the Apéry constant $\zeta(3)$:
``` math
\Pi(z) = M \cdot \boldsymbol{\omega}(z).
```
 
The central charge for a charge vector $q$ is then $Z = q \cdot \Pi(\psi)$, and the BPS mass squared (normalized by the Kähler potential) is
``` math
Z(q; \psi, \bar{\psi}) = e^{\mathcal{K}/2} \, q \cdot \Pi, \qquad \mathcal{K} = -\ln\left[i \int_X \Omega \wedge \bar{\Omega}\right] = -\ln\left[-i \, \bar{\Pi}^T \Sigma \, \Pi\right]
```
where $\Sigma$ is the symplectic pairing matrix. At an attractor point, $\psi$ minimizes a fixed charge $q$.

### 3.2 Basis Functions
 
The eigenmodes of the scalar Laplacian on the CY are expanded in a finite basis of eigenfunctions on the quintic: 

``` math
\{\alpha_A\} = \frac{s_\alpha^{(k_\phi)} \bar{s}_\beta^{(k_\phi)}}{(|z_0|^2 + \cdots + |z_n|^2)^{k_\phi}}
```
where 
```math
s_\alpha^{(k_\phi)} \in H^0(\mathcal{O}_{\mathbb{P}^2}(k_\phi))
```
are sections of the line bundle $O_{\mathbb{P}^2}(k_\phi)$. For degree the quintic, the number of monomials is $\binom{7}{4}$, giving a basis of size $\binom{7}{4}^2$. So, $k = 3$ yields 35 monomials and 1225 basis functions.
 
## 4. Numerical Methodology
 
### 4.1 Point Sampling and Computing the metric. 
 
The cymyc library [3] provides a JAX framework for computing the Ricci-flat metric for CY manifolds with neural networks. The cymyc point sampling method was used to generate points on the quintic threefold, and the Ricci-flat metric was approximated over 50 epochs, achieving a Monge-Ampère loss of 0.05. The model was trained on 100,000 points, with a learning rate for the Adam optimizer of 0.0001 and a batch size of 1024. A basis degree of k=3 was used, allowing 1225 basis functions. Future work will implement Fourier features for improved metric accuracy, but the standard cymyc method was used for speed in developing the code to compute eigenvalue crossings. 
 
### 4.2 Laplacian
 
For each sampled point $p_i$ on the quintic with weight $w_i$, pullback $J_i$, and metric $g_i$:
 
1. The metric was inverted: $g_i^{-1}$.
2. Compute the metrics pulled into ambient space coordinates, $G_{1,i}$ and $G_{2,i}$, from $J_i$ and $g_i^{-1}$.
3. Evaluate the basis functions $e_\alpha(z_i)$ and their Wirtinger derivatives $\partial e_\alpha / \partial z$, $\partial e_\alpha / \partial \bar{z}$ with JAX automatic differentiation (`jacfwd`).
4. Accumulate the per-point stiffness and overlap contributions for the matrices $L$ and $S$:

 ```math
 L = \sum_i (\kappa \, w_i / N) \, L^{(i)}, S = \sum_i (\kappa \, w_i / N) \, S^{(i)}.
```
### 4.3 Calculating the Eigenvalues
 
Both $L$ and $S$ are theoretically Hermitian but must be corrected because of numerical error. Hermiticity is ensured using $L \leftarrow \frac{1}{2}(L + L^\dagger)$, and the same method for $S$. The generalized eigenvalue problem $L \mathbf{c} = \lambda S \mathbf{c}$ is then solved with `scipy.linalg.eigvalsh`.

### 4.4 Period Computation and Attractor Search
 
To identify attractor points, the period vector was computed $\Pi(\psi)$ by:
 
1. Evaluating the Frobenius power series $\omega_k(z)$ and using the formula the digamma and polygamma functions (```scipy.special.digamma/polgamma```).
2. Solving for the initial initial conditions for the Picard-Fuchs ODE using finite difference methods and numerically integrating using Runge-Kutta with the standard ```solve_ivp``` function. 
3. Lastly, transforming from the Frobenius basis to the symplectic basis using the matrix $M$.

The algorithm then searches for $\psi$ values that minimize $Z$ for each charge vector $q$. 

 
### 4.5 Finding Eigenvalue Crossings
 
The $\psi$-sweep computes the eigenvalues for $\psi$ vaules around the crossings already identified by Ahmed and Ruehle: $\psi \in [5.30, 5.60]$, $[6.50, 7.00]$, and $[8.90, 9.30]$, with 30 sample points for each range. At each $\psi$ value, the metric found using the neural network, the Laplacian is computed, and the eigenvalues are extracted. The closest pair of eigenvalue branches at the midpoint of each window is found by finding the minimum gap in the sorted spectrum (excluding the trivial zero mode).
 
### 4.6 Attractor Points
 
The lower half of Figure 8 in Ahmed and Ruehle shows a contour plot $Z$ as a function of $\psi - \psi^*$ for the charge vector that best fits each crossing. The minimum of $Z$ near $\psi^*$ indicates that the crossing corresponds to a rank-one attractor point. The three approximate crossing/attractor locations are $\psi^* \approx 5.52$, $6.82$, and $9.17$.
 
### 4.7 Sources of Numerical Error
 
There are multiple sources of error when computing the spectrum of the Laplacian. First, the Ricci-flat metric approximation is not perfectly accurate. The basis functions are also truncated, in this case using k=3, so higher eigenvalues are less accurate. The error in these approximations must be taken into account when interpreting the eigenvalue results since they are key to the computation. 

 
## 5. Conclusion
 
The eigenvalue crossings of the quintic threefold were numerically shown by Ahmed and Ruehle to be attractor points, and the goal of this project was to reproduce this result using the cymyc codebase for the metric and point sampling computations. The final result will include comparing the results from the Ahmed and Ruehle paper to these results using adaptive Fourier features to compute the metric. 
 
## References
 
1. M. Ahmed and F. Ruehle, "Level Crossings, Attractor Points and Complex Multiplication," JHEP 06 (2023) 164, arXiv:2304.00027.
2. A. Ashmore et al., "Numerical Spectra of the Laplacian for Line Bundles on Calabi-Yau Hypersurfaces," JHEP 07 (2023) 164, arXiv:2305.08901.
3. P. Berglund et al., "cymyc -- Calabi-Yau Metrics, Yukawas, and Curvature," arXiv: 2410.19728 2025.
