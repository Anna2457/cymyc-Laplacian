# Computing the Eigenvalue Crossings and Attractor Points for the Quintic Threefold with cymyc
 
**Anna Geho, PHYS 795 Spring 2026**
 
---
 
## 1. Introduction
 
This project investigates the eigenvalue spectrum of the Laplacian on the Calabi-Yau quintic threefold. Although spectrum contains fundamental information about string compactification on CYs, it cannot be found analytically for the quintic. However, these quantities can be computed analytically for the lower-dimensional torus, yielding insights into the quintic case. Ahmed and Ruehle's 2023 paper "Level Crossings, Attractor Points and Complex Multiplication" (arXiv:2304.00027) studies the complex structure moduli dependence of the Laplacian spectrum on the torus, showing that eigenvalue crossings, where two eigenvalue branches meet, seem to be connected to complex multiplication (CM) and attractor points on tori. These results concerning the attractor points and eigenvalue crossings are then extended using numerical techniques to the quintic. 
 
The goal of this project is to reproduce Figure 8 of Ahmed and Ruehle's paper, illustrating the $/psi$ values at which the eigenvalues cross and whether they are also attractor points. The numerical implementation is done in the cymyc framework, which is a JAX-based library that can numerically approximate the Ricci-flat CY metric with neural networks. 
 
## 2. Background on Ahmed and Ruehle
 
### 2.1 Eigenvalue Crossings
 
Ahmed and Ruehle's paper begins with noting that eigenmodes can become lighter or heavier, leading to eigenvalue crossings. These points are also significant in number theory and the study of BPS black holes, and the paper investigates these points for the torus, the quartic K3, and the quintic threefold:
 
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
 
The cymyc library provides a JAX framework for computing the Ricci-flat metric for CY manifolds with neural networks. The cymyc point sampling method was used to generate points on the quintic threefold, and the Ricci-flat metric was approximated over 50 epochs, achieving a Monge-Ampère loss of 0.05. The model was trained on 100,000 points. Future work will implement Fourier features for improved metric accuracy, but the standard cymyc method was used for speed in developing the code to compute eigenvalue crossings. 
 
### 4.2 Assembly of the Laplacian
 
For each sampled point $p_i$ on the quintic with weight $w_i$, pullback $J_i$, and metric $g_i$:
 
1. The metric was inverted: $g_i^{-1}$.
2. Compute the metrics pulled into ambient space coordinates, $G_{1,i}$ and $G_{2,i}$, from $J_i$ and $g_i^{-1}$.
3. Evaluate the basis functions $e_\alpha(z_i)$ and their Wirtinger derivatives $\partial e_\alpha / \partial z$, $\partial e_\alpha / \partial \bar{z}$ with JAX automatic differentiation (`jacfwd`).
4. Accumulate the per-point stiffness and overlap contributions for the matrices $L$ and $S$:

 ```math
 L = \sum_i (\kappa \, w_i / N) \, L^{(i)}, S = \sum_i (\kappa \, w_i / N) \, S^{(i)}.
```
### 4.3 Eigenvalue Extraction
 
Both $L$ and $S$ are theoretically Hermitian but must be corrected because of numerical error. Hermiticity is ensured using $L \leftarrow \frac{1}{2}(L + L^\dagger)$, and the same method for $S$. The generalized eigenvalue problem $L \mathbf{c} = \lambda S \mathbf{c}$ is then solved with `scipy.linalg.eigvalsh`.

### 4.4 Period Computation and Attractor Search
 
To identify attractor points, we compute the period vector $\Pi(\psi)$ by:
 
1. Evaluating the Frobenius power series $\omega_k(z)$ at a small seed point $z_0 = 10^{-6}$ near the MUM point, using the explicit formula involving ratios of gamma functions and their logarithmic derivatives (digamma and polygamma functions).
2. Setting initial conditions for the Picard-Fuchs ODE using $\theta$-derivatives computed via finite differences.
3. Numerically integrating the ODE from $z_0$ to $z_{\mathrm{target}} = \psi^{-5}$ using an 8th-order Runge-Kutta method (`DOP853`) with stringent tolerances ($\mathrm{rtol} = 10^{-12}$).
4. Transforming from the Frobenius basis to the symplectic basis via the matrix $M$.
The attractor search then scans over integer charge vectors $q$ with components in $\{-2, \ldots, 2\}$ and, for each $q$, minimizes $|Z(\psi, q)|^2 / e^{-K(\psi)}$ over $\psi$ using the Nelder-Mead algorithm, seeded at the approximate crossing location.
 
### 4.5 Hyperparameter Choices
 
The following choices balance accuracy against computational cost:
 
- **Basis degree** $k = 3$: yields $\binom{7}{4} = 35$ monomials and a $1225 \times 1225$ matrix eigenvalue problem. This is the minimum degree needed for paper-quality resolution of the lowest crossings.
- **Sample points** $N = 50{,}000$ for the Laplacian integration, with $100{,}000$ points used during metric training (to ensure the neural network sees sufficient coverage of the manifold).
- **Metric training**: 256 epochs with the AdamW optimizer at learning rate $10^{-4}$ and batch size 1024.
- **Precision**: 64-bit floating point (`jax_enable_x64 = True`) throughout, with `complex128` accumulators for the stiffness and overlap matrices. This is essential — the 32-bit default in JAX introduces catastrophic cancellation in the matrix assembly.
## 5. Results and Discussion
 
### 5.1 Eigenvalue Crossings
 
The $\psi$-sweep covers three narrow windows around the crossings identified by Ahmed and Ruehle: $\psi \in [5.30, 5.60]$, $[6.50, 7.00]$, and $[8.90, 9.30]$, with 30 sample points each. At each $\psi$ value, the metric is retrained from scratch, the Laplacian is assembled, and the spectrum is extracted. The closest pair of eigenvalue branches at the midpoint of each window is identified automatically by finding the minimum gap in the sorted nontrivial spectrum.
 
The top row of Figure 8 displays the two branches approaching each other, reaching a near-degeneracy (or exact crossing, within numerical resolution), and then separating — the characteristic signature of a level crossing. The crossing locations are consistent with the values reported in the original paper, though the absolute eigenvalues may differ due to differences in the metric approximation quality and basis truncation.
 
### 5.2 Attractor Points
 
The bottom row of Figure 8 shows the central charge $|Z(\psi, q)|$ as a function of $\psi - \psi^*$ for the charge vector that best fits each crossing. The minimum of $|Z|$ near $\psi^*$ provides evidence that the crossing corresponds to a rank-one attractor point. The three approximate crossing/attractor locations are $\psi^* \approx 5.52$, $6.82$, and $9.17$.
 
### 5.3 Sources of Numerical Error
 
Several sources of systematic error affect the results. The metric approximation has finite accuracy; the Monge-Ampère residual is typically $\mathcal{O}(10^{-2})$ to $\mathcal{O}(10^{-1})$ after 256 epochs, far from machine precision. The basis truncation at $k = 3$ means only the lowest eigenvalues are reliably captured — higher eigenvalues are increasingly contaminated by truncation error. The Monte Carlo integration introduces statistical noise proportional to $1/\sqrt{N}$. Finally, the period computation is limited by the convergence radius of the Frobenius series and the accuracy of the ODE integrator, though with 80 terms and tolerances of $10^{-12}$, this is the most controlled source of error.
 
### 5.4 Comparison with the Original Paper
 
The original paper by Ahmed and Ruehle used the cymetric library with TensorFlow-based metric approximation, whereas this reproduction uses cymyc with JAX. Both libraries implement essentially the same point sampling strategy (hyperplane intersection) and metric training objective (Monge-Ampère loss), but differ in their neural network architectures and optimization details. The qualitative agreement of the crossing locations provides a nontrivial cross-check of both the mathematical framework and the numerical implementation.
 
## 6. Conclusion
 
This work reproduces the eigenvalue crossing structure on the Dwork quintic threefold originally reported by Ahmed and Ruehle, using an independent codebase (cymyc) as the metric backend. The three crossings near $\psi \approx 5.4, 6.8, 9.1$ are recovered, and the associated attractor point structure is verified through minimization of the BPS central charge. The results support the conjecture that eigenvalue crossings of the scalar Laplacian on Calabi-Yau manifolds are connected to arithmetic special points — complex multiplication for tori and K3 surfaces, and rank-one attractors for threefolds.
 
The code is available as `Laplacian.py` in the cymyc examples directory and can be extended to higher basis degrees, more sample points, or different one-parameter CY families (such as the octic in weighted $\mathbb{P}^4$) with minimal modification.
 
## References
 
1. M. Ahmed and F. Ruehle, "Level Crossings, Attractor Points and Complex Multiplication," JHEP 06 (2023) 164, arXiv:2304.00027.
2. A. Ashmore, "Eigenvalues and Eigenforms on Calabi-Yau Threefolds," J. Geom. Phys. 195 (2024), arXiv:2011.13929.
3. A. Ashmore, Y.-H. He, and B. Ovrut, "Numerical Spectra of the Laplacian for Line Bundles on Calabi-Yau Hypersurfaces," JHEP 07 (2023) 164, arXiv:2305.08901.
4. V. Braun, T. Brelidze, M. R. Douglas, and B. A. Ovrut, "Eigenvalues and Eigenfunctions of the Scalar Laplace Operator on Calabi-Yau Manifolds," JHEP 07 (2008) 120, arXiv:0805.3689.
5. S. Ferrara, R. Kallosh, and A. Strominger, "N=2 Extremal Black Holes," Phys. Rev. D 52 (1995) 5412, arXiv:hep-th/9508072.
6. G. W. Moore, "Arithmetic and Attractors," arXiv:hep-th/9807087.
7. J. Tan, "cymyc: Calabi-Yau Metrics with Machine Learning," GitHub repository, 2022.
8. S. K. Donaldson, "Some Numerical Results in Complex Differential Geometry," Pure Appl. Math. Q. 5 (2009) 571.
