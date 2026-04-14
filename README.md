# Numerical Minimization of `f(x_1, x_2) = x_1^2 + x_2^4`

This repository contains a Python implementation and short report for two iterative optimization methods applied to the minimization problem

\[
f(x_1, x_2) = x_1^2 + x_2^4,
\]

starting from the initial point

\[
x^{(0)} = (1, 1).
\]

The implemented methods are:

- Steepest Descent
- Modified Newton Method

Both methods use an exact line search along the chosen search direction, and the code produces contour plots showing the path of the iterates in the `x_1-x_2` plane.

## Problem Details

- Objective function:
  `f(x_1, x_2) = x_1^2 + x_2^4`
- Gradient:
  `\nabla f(x_1, x_2) = [2x_1, 4x_2^3]^T`
- Hessian:
  `\nabla^2 f(x_1, x_2) = [[2, 0], [0, 12x_2^2]]`
- True minimizer:
  `(0, 0)`
- Minimum value:
  `f(0, 0) = 0`

## Algorithms

### 1. Steepest Descent

At each iteration `k`:

1. Start with the current point `x_k`.
2. Compute the descent direction
   `p_k = -\nabla f(x_k)`.
3. Compute the step size by exact line search
   `\alpha_k = \arg\min_{\alpha \ge 0} f(x_k + \alpha p_k)`.
4. Update the iterate
   `x_{k+1} = x_k + \alpha_k p_k`.
5. Stop when the relative step size is sufficiently small.

### 2. Modified Newton Method

At each iteration `k`:

1. Start with the current point `x_k`.
2. Build the Hessian `H_k = \nabla^2 f(x_k)`.
3. Since the Hessian can become singular near `x_2 = 0`, form a regularized matrix
   `B_k = H_k + \mu_k I`,
   where `\mu_k` is chosen so that `B_k` is positive definite.
4. Compute the search direction
   `p_k = -B_k^{-1}\nabla f(x_k)`.
5. Compute the step size by exact line search
   `\alpha_k = \arg\min_{\alpha \ge 0} f(x_k + \alpha p_k)`.
6. Update the iterate
   `x_{k+1} = x_k + \alpha_k p_k`.

## Repository Contents

- `optimization_methods.py`:
  full Python implementation
- `REPORT.md`:
  detailed mathematical report and observations
- `plots/`:
  contour plots and convergence figures
- `results/`:
  iteration tables and numerical summaries

## How to Run

Use Python 3 with `numpy` and `matplotlib` installed.

```bash
python optimization_methods.py
```

The script will generate:

- `plots/steepest_descent_contour.svg`
- `plots/modified_newton_contour.svg`
- `plots/iterate_paths_comparison.svg`
- `plots/convergence_history.svg`
- `results/steepest_descent_history.csv`
- `results/steepest_descent_history_sampled.csv`
- `results/modified_newton_history.csv`
- `results/modified_newton_history_sampled.csv`
- `results/summary.txt`

## Numerical Results From The Current Run

Using the starting point `(1, 1)`, the generated summary is:

| Method | Iterations | Final point | Final objective value | Final gradient norm |
| --- | ---: | --- | ---: | ---: |
| Steepest Descent | 3015 | `(0.000000, -0.000356)` | `1.5973e-14` | `1.7972e-10` |
| Modified Newton | 14 | `(0.000000, 0.000224)` | `2.5202e-15` | `4.4992e-11` |

These values show that both methods move to the true minimizer, but the Modified Newton method reaches a comparable accuracy in far fewer iterations.

## Plot Outputs

When `optimization_methods.py` is executed, it writes the following SVG figures into `plots/`:

- `steepest_descent_contour.svg`
- `modified_newton_contour.svg`
- `iterate_paths_comparison.svg`
- `convergence_history.svg`

These figures show the contour map of `f(x_1, x_2) = x_1^2 + x_2^4`, the iterate path on the `x_1-x_2` plane, and the convergence history of the objective value and gradient norm.

## Expected Behavior

- Steepest Descent converges steadily but typically requires more iterations.
- Modified Newton converges much faster because it uses second-order curvature information.
- The contour plots clearly show both methods approaching the minimizer `(0, 0)`.

## Notes

- The line search is implemented by solving for the zero of the directional derivative on `\alpha \ge 0`.
- The Modified Newton Method adds a small positive diagonal shift when the Hessian is not sufficiently positive definite.
- The implementation is written specifically for the given two-variable optimization problem, while still keeping the code clean and easy to follow.
- For very long iteration histories, the contour plots use a lightly sampled path for readability, while the full iterate list is preserved in the CSV files.
- Sampled CSV files are also exported for quick inspection in addition to the full iteration histories.
