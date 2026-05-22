# Computing the Eigenvalue Crossings and Attractor Points for the Quintic Threefold with cymyc
 
**Anna Geho, PHYS 795 Spring 2026**
 
---
 
## 1. Introduction
 
This project investigates the eigenvalue spectrum of the Laplacian on the Calabi-Yau quintic threefold. Although spectrum contains fundamental information about string compactification on CYs, it cannot be found analytically for the quintic. However, these quantities can be computed analytically for the lower-dimensional torus, yielding insights into the quintic case. Ahmed and Ruehle's 2023 paper "Level Crossings, Attractor Points and Complex Multiplication" (arXiv:2304.00027) studies the complex structure moduli dependence of the Laplacian spectrum on the torus, showing that eigenvalue crossings, where two eigenvalue branches meet, seem to be connected to complex multiplication (CM) and attractor points on tori. These results concerning the attractor points and eigenvalue crossings are then extended using numerical techniques to the quintic. 
 
The goal of this project is to reproduce Figure 8 of Ahmed and Ruehle's paper, illustrating the $/psi$ values at which the eigenvalues cross and whether they are also attractor points. The numerical implementation is done in the cymyc framework, which is a JAX-based library that can numerically approximate the Ricci-flat CY metric with neural networks. 
 
## 2. Background on Ahmed and Ruehle
 
### 2.1 Prior Work on Numerical CY Spectra
 
The numerical computation of Laplacian eigenvalues on Calabi-Yau manifolds has a relatively short history. The foundational contribution is due to Braun, Brelidze, Douglas, and Ovrut (arXiv:0805.3689), who in 2008 gave the first explicit numerical computation of eigenvalues and eigenfunctions of the scalar Laplace-Beltrami operator on Calabi-Yau threefolds, including the Fermat quintic and certain $\mathbb{Z}_5 \times \mathbb{Z}_5$ quotients. Their approach used Donaldson's algorithm for balanced metrics together with a finite-element-style expansion in a polynomial basis.
 
Ashmore (arXiv:2011.13929) extended this program significantly by computing the spectrum of the full Laplace-de Rham operator acting on $(p,q)$-forms, not just the scalar Laplacian. His algorithm, validated against exact results on $\mathbb{P}^3$, provided the first numerical eigenvalue calculations for bundle-valued forms on the Fermat quintic. A follow-up by Ashmore, Cross, and Mayerson (arXiv:2305.08901) further extended the computation to line bundles over Calabi-Yau hypersurfaces.
 
All of these works computed the spectrum at a single point in complex structure moduli space — typically the Fermat point. The central innovation of Ahmed and Ruehle was to track how the spectrum varies as a function of the moduli, revealing a rich structure of eigenvalue crossings.
 
### 2.2 The Level Crossing Conjecture
 
Ahmed and Ruehle's paper begins with the key observation that eigenvalue crossings are not generic — they represent special loci in moduli space where the accidental degeneracy of eigenvalues reflects an enhanced arithmetic structure of the underlying geometry. The paper builds its case through a hierarchy of examples of increasing complexity.
 
**Torus.** For the one-parameter family of complex tori $\mathbb{C}/(\mathbb{Z} + \tau\mathbb{Z})$, the scalar Laplacian spectrum is known exactly. Eigenvalue crossings occur at values of $\tau$ where the torus has complex multiplication — that is, where $\tau$ is a quadratic irrationality satisfying $a\tau^2 + b\tau + c = 0$ with $a, b, c \in \mathbb{Z}$. At CM points, the endomorphism ring of the torus enlarges from $\mathbb{Z}$ to an order in an imaginary quadratic field, and the extra symmetry forces eigenvalue degeneracies beyond what the lattice symmetry group predicts.
 
**Quartic K3.** The quartic hypersurface in $\mathbb{P}^3$, defined by $z_0^4 + z_1^4 + z_2^4 + z_3^4 + 4\psi \, z_0 z_1 z_2 z_3 = 0$, provides a two-complex-dimensional testing ground. Here the notion of CM generalizes: a K3 surface has CM when its Picard rank jumps from the generic value of 19 (for this family) to the maximum of 20. Ahmed and Ruehle computed the scalar Laplacian spectrum numerically for a range of $\psi$ values and identified eigenvalue crossings that coincide with known CM points of the quartic K3.
 
**Quintic threefold.** The Dwork quintic, $z_0^5 + z_1^5 + z_2^5 + z_3^5 + z_4^5 + 5\psi \, z_0 z_1 z_2 z_3 z_4 = 0$, is the main object of study. For Calabi-Yau threefolds, the notion of CM is less well-developed, but the attractor mechanism from $\mathcal{N}=2$ supergravity provides a natural analogue. Ahmed and Ruehle conjectured — and provided numerical evidence — that eigenvalue crossings on the quintic correspond to rank-one attractor points, i.e., points in moduli space where the central charge $|Z(q, \Pi(\psi))|$ is minimized for some integer charge vector $q$.
 
### 2.3 The Attractor Mechanism
 
In four-dimensional $\mathcal{N}=2$ supergravity arising from type IIB string theory compactified on a Calabi-Yau threefold $X$, BPS black holes carry charges $q = (p^0, p^1, q_0, q_1) \in H^3(X, \mathbb{Z})$ under the graviphoton and vector multiplet gauge fields. The attractor mechanism, discovered by Ferrara, Kallosh, and Strominger, states that the complex structure moduli flow along the black hole solution from spatial infinity to fixed values at the horizon, determined entirely by the charges. These fixed points — attractor points — minimize the BPS mass $|Z|^2 / e^{-K}$, where $Z = q \cdot \Pi$ is the central charge and $K$ is the Kähler potential of the special geometry on moduli space.
 
Moore observed that attractor points on CY threefolds should be "arithmetic" — their periods satisfy algebraic relations. For the torus, this reduces to the classical statement that CM points are quadratic irrationalities. For higher-dimensional CY manifolds, the precise arithmetic characterization remains an open problem, though it is expected to be related to the Hodge conjecture.
 
## 3. Mathematical Framework
 
### 3.1 Special Geometry and Periods of the Quintic
 
The complex structure moduli space of the quintic is one-dimensional, parameterized by $\psi$. The holomorphic 3-form $\Omega$ depends on $\psi$, and its periods $\Pi = (X^0, X^1, F_0, F_1)^T$ satisfy the Picard-Fuchs equation
$$
\left[\theta^4 - 5^5 z \,(\theta + \tfrac{1}{5})(\theta + \tfrac{2}{5})(\theta + \tfrac{3}{5})(\theta + \tfrac{4}{5})\right] f(z) = 0,
$$
where $\theta = z \frac{d}{dz}$ and $z = \psi^{-5}$ is the algebraic coordinate. The maximally unipotent monodromy (MUM) point at $z = 0$ ($\psi \to \infty$) admits four Frobenius solutions with increasing logarithmic order:
$$
\omega_k(z) = \frac{1}{k!} \frac{\partial^k}{\partial \rho^k} \bigg|_{\rho=0} \sum_{n=0}^\infty \frac{\Gamma(5(n+\rho)+1)}{\Gamma(n+\rho+1)^5} \, z^{n+\rho}.
$$
These are related to the symplectic period vector by a constant matrix $M$ encoding the topological data of the quintic (triple intersection number $\kappa = 5$, Euler characteristic $\chi = -200$, and the Apéry constant $\zeta(3)$):
$$
\Pi(z) = M \cdot \boldsymbol{\omega}(z).
$$
 
The central charge for a charge vector $q$ is then $Z = q \cdot \Pi(\psi)$, and the BPS mass squared (normalized by the Kähler potential) is
$$
|Z|^2 / e^{-K} = \frac{|q \cdot \Pi|^2}{-i \, \bar{\Pi}^T \Sigma \, \Pi},
$$
where $\Sigma$ is the symplectic pairing matrix. Attractor points minimize this quantity over $\psi$ for fixed integer $q$.
 
### 3.2 The Scalar Laplacian on a CY Manifold
 
Let $(X, g)$ be a compact Kähler manifold of complex dimension $n$ with local complex coordinates $z^i$, $\bar{z}^{\bar{\jmath}}$. The scalar Laplacian acting on a function $\phi$ is
$$
\Delta \phi = -g^{i\bar{\jmath}} \nabla_i \nabla_{\bar{\jmath}} \phi - g^{\bar{\jmath}i} \nabla_{\bar{\jmath}} \nabla_i \phi.
$$
To solve the eigenvalue problem $\Delta \phi = \lambda \phi$ numerically, one expands $\phi$ in a finite basis $\{e_\alpha\}$ of rational functions on $X$ and converts the PDE to a generalized eigenvalue problem $L \mathbf{c} = \lambda S \mathbf{c}$, where
$$
L_{\alpha\beta} = \int_X \left[\overline{\partial_{\bar{\jmath}} e_\alpha} \, G_1^{\bar{\jmath}i} \, \partial_{\bar{\imath}} e_\beta + \overline{\partial_i e_\alpha} \, G_2^{i\bar{\jmath}} \, \partial_{\bar{\jmath}} e_\beta\right] \mathrm{dvol}
$$
is the stiffness matrix,
$$
S_{\alpha\beta} = \int_X \bar{e}_\alpha \, e_\beta \, \mathrm{dvol}
$$
is the overlap (mass) matrix, and $G_1 = J^T g^{-1} \bar{J}$ with $J$ the embedding Jacobian from the CY to ambient projective space. The Wirtinger derivatives $\partial/\partial z$ and $\partial/\partial \bar{z}$ are computed from real Jacobians via $\partial f/\partial z = \frac{1}{2}(\partial f/\partial x - i \, \partial f/\partial y)$.
 
### 3.3 Basis Functions
 
The finite basis consists of degree-$k$ rational functions on $\mathbb{P}^4$ restricted to the quintic:
$$
e_{\alpha\beta}(z) = \frac{z^\alpha \bar{z}^\beta}{|z|^{2k}},
$$
where $\alpha$, $\beta$ are multi-indices with $|\alpha| = |\beta| = k$ and $z$ denotes the homogeneous coordinates. These are well-defined on projective space (they have the correct scaling weight) and form a complete basis in the limit $k \to \infty$. For degree $k$ in 5 homogeneous variables, the number of monomials is $\binom{k+4}{4}$, giving a basis of size $\binom{k+4}{4}^2$. At $k = 3$, this yields 35 monomials and a $1225 \times 1225$ matrix system.
 
## 4. Computational Methodology
 
### 4.1 The cymyc Library
 
The cymyc library (Tan, 2022) provides a JAX-based framework for computing approximate Ricci-flat metrics on Calabi-Yau manifolds via neural network regression. The key components used in this work are:
 
**Point sampling.** The `DworkQuintic` class generates uniformly distributed points on the quintic hypersurface via a hyperplane intersection method: random lines in $\mathbb{P}^4$ are intersected with the defining equation $Q(z) = 0$, and the resulting roots — computed via companion matrices — give points on $X$. Integration weights $w_i = \Omega \wedge \bar{\Omega} / dA_{\mathrm{ref}}$ are computed via the Poincaré residue map, ensuring that weighted sums approximate integrals against the holomorphic volume form.
 
**Metric approximation.** The `RicciFlatMetric` class trains a neural network correction $h_{i\bar{\jmath}}$ to the Fubini-Study metric $g^{\mathrm{FS}}_{i\bar{\jmath}}$ by minimizing the Monge-Ampère loss $\| \det(g^{\mathrm{FS}} + h) / \det(g^{\mathrm{FS}}) - \kappa \,\Omega \wedge \bar{\Omega} / \omega^n_{\mathrm{FS}} \|$. The predicted metric is pulled back to the CY via $g_{\mathrm{pred}} = J^T (g_{\mathrm{FS}} + h) \bar{J}$, where $J$ is the embedding Jacobian. The cymyc Fubini-Study metric includes a $1/\pi$ normalization: $g^{\mathrm{FS}}_{i\bar{\jmath}} = (\delta_{ij} |z|^2 - \bar{z}_i z_j) / (\pi |z|^4)$.
 
### 4.2 Assembly of the Laplacian
 
For each sampled point $p_i$ on the quintic with weight $w_i$, pullback $J_i$, and metric $g_i$:
 
1. Invert the metric: $g_i^{-1}$.
2. Compute $G_{1,i} = J_i^T \, g_i^{-1} \, \bar{J}_i$ and $G_{2,i} = \bar{G}_{1,i}$.
3. Evaluate the basis functions $e_\alpha(z_i)$ and their Wirtinger derivatives $\partial e_\alpha / \partial z$, $\partial e_\alpha / \partial \bar{z}$ via JAX automatic differentiation (`jacfwd`).
4. Accumulate the per-point stiffness and overlap contributions:
$$
L^{(i)}_{\alpha\beta} = \overline{(\partial_{\bar{z}} e_\alpha)}_j \, G_1^{jk} \, (\partial_{\bar{z}} e_\beta)_k + \overline{(\partial_z e_\alpha)}_j \, G_2^{jk} \, (\partial_z e_\beta)_k, \qquad S^{(i)}_{\alpha\beta} = \bar{e}_\alpha \, e_\beta.
$$
5. Form the global matrices: $L = \sum_i (\kappa \, w_i / N) \, L^{(i)}$, $S = \sum_i (\kappa \, w_i / N) \, S^{(i)}$.
The factor $\kappa \, w_i / N$ ensures the Monte Carlo sum approximates the integral $\int_X \cdot \, \omega^n / n!$ with the correct normalization ($\kappa = 5$ for the quintic).
 
### 4.3 Eigenvalue Extraction
 
Both $L$ and $S$ are Hermitian by construction, but numerical noise breaks this symmetry slightly. We enforce Hermiticity via $L \leftarrow \frac{1}{2}(L + L^\dagger)$ and similarly for $S$. The generalized eigenvalue problem $L \mathbf{c} = \lambda S \mathbf{c}$ is then solved with `scipy.linalg.eigvalsh`, which exploits the Hermitian-positive-definite structure of $S$ via Cholesky factorization. The lowest eigenvalue $\lambda_0 \approx 0$ corresponds to the constant mode and serves as a consistency check. We track the first 80 nontrivial eigenvalues across moduli space.
 
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
