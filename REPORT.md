# Report: Steepest Descent and Modified Newton Method

## 1. Objective

The goal is to numerically solve the unconstrained minimization problem

\[
\min_{x \in \mathbb{R}^2} f(x_1, x_2) = x_1^2 + x_2^4.
\]

The function is nonnegative for all `(x_1, x_2)` and reaches its unique global minimum at `(0, 0)`.

The numerical experiments in this repository start from

\[
x^{(0)} = (1, 1).
\]

## 2. Derivatives of the Objective Function

For

\[
f(x_1, x_2) = x_1^2 + x_2^4,
\]

the gradient is

\[
\nabla f(x_1, x_2)
=
\begin{bmatrix}
2x_1 \\
4x_2^3
\end{bmatrix},
\]

and the Hessian is

\[
\nabla^2 f(x_1, x_2)
=
\begin{bmatrix}
2 & 0 \\
0 & 12x_2^2
\end{bmatrix}.
\]

The Hessian is positive semidefinite for all `(x_1, x_2)`. However, it becomes singular when `x_2 = 0`, so a modified Newton strategy is useful in practice.

## 3. Steepest Descent Method

The Steepest Descent method chooses the negative gradient as the search direction:

\[
p_k = -\nabla f(x_k).
\]

The step size is obtained from an exact line search:

\[
\alpha_k = \arg\min_{\alpha \ge 0} f(x_k + \alpha p_k).
\]

The iterate is then updated by

\[
x_{k+1} = x_k + \alpha_k p_k.
\]

### Pseudocode

1. Choose an initial point `x_0`.
2. For `k = 0, 1, 2, ...`:
3. Compute `p_k = -\nabla f(x_k)`.
4. Compute `\alpha_k = \arg\min_{\alpha \ge 0} f(x_k + \alpha p_k)`.
5. Set `x_{k+1} = x_k + \alpha_k p_k`.
6. Stop when the step becomes sufficiently small.

## 4. Modified Newton Method

The Newton direction is usually computed from

\[
\nabla^2 f(x_k) p_k = -\nabla f(x_k).
\]

Since the Hessian here may be singular near the minimizer, the modified Newton method replaces the Hessian by

\[
B_k = \nabla^2 f(x_k) + \mu_k I,
\]

where `\mu_k \ge 0` is chosen so that `B_k` is positive definite. The search direction becomes

\[
p_k = -B_k^{-1}\nabla f(x_k).
\]

Again, the step length is selected through exact line search:

\[
\alpha_k = \arg\min_{\alpha \ge 0} f(x_k + \alpha p_k).
\]

The update is

\[
x_{k+1} = x_k + \alpha_k p_k.
\]

### Pseudocode

1. Choose an initial point `x_0`.
2. For `k = 0, 1, 2, ...`:
3. Form the Hessian `H_k = \nabla^2 f(x_k)`.
4. Compute a positive-definite approximation `B_k = H_k + \mu_k I`.
5. Solve `B_k p_k = -\nabla f(x_k)`.
6. Compute `\alpha_k = \arg\min_{\alpha \ge 0} f(x_k + \alpha p_k)`.
7. Set `x_{k+1} = x_k + \alpha_k p_k`.
8. Stop when the step becomes sufficiently small.

## 5. Implementation Details

The Python program `optimization_methods.py` includes:

- the objective function
- its gradient and Hessian
- exact line search using the directional derivative
- Steepest Descent
- Modified Newton with Hessian regularization
- CSV export of all iterates
- SVG contour plots of the optimization paths

The line search is implemented by bracketing and bisection on the directional derivative `\phi'(\alpha)`, where

\[
\phi(\alpha) = f(x_k + \alpha p_k).
\]

This gives a stable and reproducible exact line search for the present convex objective.

## 6. Visual Output

The following plots are generated automatically:

1. Contour plot with Steepest Descent iterates.
2. Contour plot with Modified Newton iterates.
3. Combined contour plot showing both iterate paths.
4. Convergence history for the objective value and gradient norm.

These plots help visualize how the two methods move toward the minimizer in the `x_1-x_2` plane.

## 7. Discussion

The function `x_1^2 + x_2^4` is convex, but the curvature is highly anisotropic:

- in the `x_1` direction the behavior is quadratic
- in the `x_2` direction the behavior is quartic

This means that first-order and second-order methods behave differently:

- Steepest Descent follows the negative gradient and usually takes more iterations.
- Modified Newton uses curvature information and generally reaches the minimizer more quickly.

Near the minimizer, the Hessian loses rank because the second derivative with respect to `x_2` tends to zero. The diagonal regularization in the modified Newton method prevents numerical instability and preserves a descent direction.

## 8. Numerical Outcome

For the run stored in this repository, the following results were obtained:

| Method | Iterations | Final point | Final objective value | Final gradient norm |
| --- | ---: | --- | ---: | ---: |
| Steepest Descent | 3015 | `(0.000000, -0.000356)` | `1.5973e-14` | `1.7972e-10` |
| Modified Newton | 14 | `(0.000000, 0.000224)` | `2.5202e-15` | `4.4992e-11` |

The numerical evidence confirms that both methods approach the minimizer `(0, 0)`. However, the Steepest Descent method needs thousands of iterations, while the Modified Newton method reaches essentially the same accuracy in only a small number of steps.

## 9. Conclusion

Both algorithms successfully solve the minimization problem. The contour plots and convergence histories show that:

- both methods converge to `(0, 0)`
- the function value approaches zero
- the Modified Newton method is more efficient in terms of iteration count

This repository provides both the mathematical explanation and a reproducible Python implementation for the given problem.
