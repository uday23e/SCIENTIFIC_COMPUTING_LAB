from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"

plt.rcParams["svg.fonttype"] = "none"


@dataclass
class OptimizationResult:
    method: str
    iterates: np.ndarray
    values: np.ndarray
    gradient_norms: np.ndarray
    step_sizes: np.ndarray
    shifts: np.ndarray

    @property
    def iterations(self) -> int:
        return max(0, len(self.iterates) - 1)

    @property
    def minimizer(self) -> np.ndarray:
        return self.iterates[-1]

    @property
    def minimum_value(self) -> float:
        return float(self.values[-1])


def f(x: np.ndarray) -> float:
    return float(x[0] ** 2 + x[1] ** 4)


def grad_f(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * x[0], 4.0 * x[1] ** 3], dtype=float)


def hess_f(x: np.ndarray) -> np.ndarray:
    return np.array([[2.0, 0.0], [0.0, 12.0 * x[1] ** 2]], dtype=float)


def directional_derivative(alpha: float, x: np.ndarray, direction: np.ndarray) -> float:
    return float(grad_f(x + alpha * direction) @ direction)


def exact_line_search(
    x: np.ndarray,
    direction: np.ndarray,
    initial_upper: float = 1.0,
    tol: float = 1e-12,
    max_expand: int = 60,
    max_iter: int = 200,
) -> float:
    derivative_at_zero = directional_derivative(0.0, x, direction)
    if derivative_at_zero >= 0.0:
        return 0.0

    lower = 0.0
    upper = initial_upper
    derivative_upper = directional_derivative(upper, x, direction)

    expands = 0
    while derivative_upper < 0.0 and expands < max_expand:
        upper *= 2.0
        derivative_upper = directional_derivative(upper, x, direction)
        expands += 1

    if derivative_upper < 0.0:
        return upper

    for _ in range(max_iter):
        midpoint = 0.5 * (lower + upper)
        derivative_midpoint = directional_derivative(midpoint, x, direction)

        if abs(derivative_midpoint) < tol or (upper - lower) < tol * max(1.0, upper):
            return midpoint

        if derivative_midpoint > 0.0:
            upper = midpoint
        else:
            lower = midpoint

    return 0.5 * (lower + upper)


def modified_newton_direction(x: np.ndarray, delta: float = 1e-6) -> tuple[np.ndarray, float]:
    gradient = grad_f(x)
    hessian = hess_f(x)
    smallest_eigenvalue = float(np.min(np.linalg.eigvalsh(hessian)))
    shift = max(0.0, delta - smallest_eigenvalue)
    regularized_hessian = hessian + shift * np.eye(len(x))
    direction = -np.linalg.solve(regularized_hessian, gradient)
    return direction, shift


def steepest_descent(
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> OptimizationResult:
    x = np.array(x0, dtype=float)
    iterates = [x.copy()]
    values = [f(x)]
    gradient_norms = [np.linalg.norm(grad_f(x))]
    step_sizes: list[float] = []
    shifts: list[float] = []

    for _ in range(max_iter):
        gradient = grad_f(x)
        if np.linalg.norm(gradient) < tol:
            break

        direction = -gradient
        alpha = exact_line_search(x, direction)
        x_next = x + alpha * direction

        step_sizes.append(alpha)
        shifts.append(0.0)
        iterates.append(x_next.copy())
        values.append(f(x_next))
        gradient_norms.append(np.linalg.norm(grad_f(x_next)))

        if np.linalg.norm(x_next - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_next
            break

        x = x_next

    return OptimizationResult(
        method="Steepest Descent",
        iterates=np.array(iterates),
        values=np.array(values),
        gradient_norms=np.array(gradient_norms),
        step_sizes=np.array(step_sizes),
        shifts=np.array(shifts),
    )


def modified_newton(
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> OptimizationResult:
    x = np.array(x0, dtype=float)
    iterates = [x.copy()]
    values = [f(x)]
    gradient_norms = [np.linalg.norm(grad_f(x))]
    step_sizes: list[float] = []
    shifts: list[float] = []

    for _ in range(max_iter):
        gradient = grad_f(x)
        if np.linalg.norm(gradient) < tol:
            break

        direction, shift = modified_newton_direction(x)
        alpha = exact_line_search(x, direction)
        x_next = x + alpha * direction

        step_sizes.append(alpha)
        shifts.append(shift)
        iterates.append(x_next.copy())
        values.append(f(x_next))
        gradient_norms.append(np.linalg.norm(grad_f(x_next)))

        if np.linalg.norm(x_next - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_next
            break

        x = x_next

    return OptimizationResult(
        method="Modified Newton",
        iterates=np.array(iterates),
        values=np.array(values),
        gradient_norms=np.array(gradient_norms),
        step_sizes=np.array(step_sizes),
        shifts=np.array(shifts),
    )


def export_history(result: OptimizationResult) -> None:
    file_name = result.method.lower().replace(" ", "_") + "_history.csv"
    output_path = RESULTS_DIR / file_name
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["iteration", "x1", "x2", "f(x)", "gradient_norm", "alpha", "hessian_shift"]
        )
        for index, point in enumerate(result.iterates):
            alpha = result.step_sizes[index - 1] if index > 0 else ""
            shift = result.shifts[index - 1] if index > 0 else ""
            writer.writerow(
                [
                    index,
                    f"{point[0]:.12e}",
                    f"{point[1]:.12e}",
                    f"{result.values[index]:.12e}",
                    f"{result.gradient_norms[index]:.12e}",
                    alpha if alpha == "" else f"{alpha:.12e}",
                    shift if shift == "" else f"{shift:.12e}",
                ]
            )


def export_sampled_history(result: OptimizationResult, max_rows: int = 200) -> None:
    file_name = result.method.lower().replace(" ", "_") + "_history_sampled.csv"
    output_path = RESULTS_DIR / file_name
    total_rows = len(result.iterates)
    stride = max(1, total_rows // max_rows)
    indices = list(range(0, total_rows, stride))
    if indices[-1] != total_rows - 1:
        indices.append(total_rows - 1)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["iteration", "x1", "x2", "f(x)", "gradient_norm", "alpha", "hessian_shift"]
        )
        for index in indices:
            point = result.iterates[index]
            alpha = result.step_sizes[index - 1] if index > 0 else ""
            shift = result.shifts[index - 1] if index > 0 else ""
            writer.writerow(
                [
                    index,
                    f"{point[0]:.12e}",
                    f"{point[1]:.12e}",
                    f"{result.values[index]:.12e}",
                    f"{result.gradient_norms[index]:.12e}",
                    alpha if alpha == "" else f"{alpha:.12e}",
                    shift if shift == "" else f"{shift:.12e}",
                ]
            )


def export_summary(results: list[OptimizationResult]) -> None:
    output_path = RESULTS_DIR / "summary.txt"
    lines = []
    for result in results:
        lines.append(result.method)
        lines.append(f"Iterations: {result.iterations}")
        lines.append(
            "Final point: "
            f"({result.minimizer[0]:.12e}, {result.minimizer[1]:.12e})"
        )
        lines.append(f"Final objective value: {result.minimum_value:.12e}")
        lines.append(f"Final gradient norm: {result.gradient_norms[-1]:.12e}")
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def make_contour_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = np.linspace(-0.15, 1.05, 180)
    x2 = np.linspace(-0.15, 1.05, 180)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    z_grid = x1_grid**2 + x2_grid**4
    return x1_grid, x2_grid, z_grid


def sample_path(points: np.ndarray, max_points: int = 320) -> np.ndarray:
    if len(points) <= max_points:
        return points
    stride = max(1, len(points) // (max_points - 1))
    sampled = points[::stride]
    if not np.array_equal(sampled[-1], points[-1]):
        sampled = np.vstack([sampled, points[-1]])
    return sampled


def sample_series(values: np.ndarray, max_points: int = 320) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= max_points:
        indices = np.arange(len(values))
        return indices, values
    stride = max(1, len(values) // (max_points - 1))
    indices = np.arange(0, len(values), stride)
    sampled = values[::stride]
    if indices[-1] != len(values) - 1:
        indices = np.append(indices, len(values) - 1)
        sampled = np.append(sampled, values[-1])
    return indices, sampled


def plot_contours_with_iterates(result: OptimizationResult, file_name: str, color: str) -> None:
    x1_grid, x2_grid, z_grid = make_contour_grid()
    levels = np.geomspace(1e-8, float(np.max(z_grid)), 12)

    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    axis.contour(x1_grid, x2_grid, z_grid, levels=levels, cmap="viridis", linewidths=0.9)

    iterates = sample_path(result.iterates)
    axis.plot(
        iterates[:, 0],
        iterates[:, 1],
        marker="o",
        color=color,
        linewidth=2.0,
        markersize=4.5,
        label=result.method,
    )
    axis.scatter(iterates[0, 0], iterates[0, 1], color="black", s=48, label="Start")
    axis.scatter(0.0, 0.0, color="gold", edgecolor="black", s=80, label="True minimizer")
    axis.set_title(f"{result.method}: contour plot with iterates")
    axis.set_xlabel("$x_1$")
    axis.set_ylabel("$x_2$")
    axis.set_xlim(-0.05, 1.05)
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / file_name, format="svg")
    plt.close(figure)


def plot_path_comparison(results: list[OptimizationResult]) -> None:
    x1_grid, x2_grid, z_grid = make_contour_grid()
    levels = np.geomspace(1e-8, float(np.max(z_grid)), 12)

    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    axis.contour(x1_grid, x2_grid, z_grid, levels=levels, cmap="Greys", alpha=0.7, linewidths=0.9)

    colors = {"Steepest Descent": "#c0392b", "Modified Newton": "#0b6e4f"}
    for result in results:
        iterates = sample_path(result.iterates)
        axis.plot(
            iterates[:, 0],
            iterates[:, 1],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=colors[result.method],
            label=f"{result.method} ({result.iterations} iterations)",
        )

    axis.scatter(1.0, 1.0, color="black", s=48, label="Start")
    axis.scatter(0.0, 0.0, color="gold", edgecolor="black", s=80, label="True minimizer")
    axis.set_title("Comparison of iterate paths on the contour map")
    axis.set_xlabel("$x_1$")
    axis.set_ylabel("$x_2$")
    axis.set_xlim(-0.05, 1.05)
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "iterate_paths_comparison.svg", format="svg")
    plt.close(figure)


def plot_convergence_history(results: list[OptimizationResult]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = {"Steepest Descent": "#c0392b", "Modified Newton": "#0b6e4f"}

    for result in results:
        iterations, values = sample_series(result.values)
        grad_iterations, grad_norms = sample_series(result.gradient_norms)
        axes[0].semilogy(iterations, values, marker="o", color=colors[result.method], label=result.method)
        axes[1].semilogy(
            grad_iterations,
            grad_norms,
            marker="o",
            color=colors[result.method],
            label=result.method,
        )

    axes[0].set_title("Objective value versus iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("$f(x_k)$")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Gradient norm versus iteration")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("$||\\nabla f(x_k)||$")
    axes[1].grid(alpha=0.25)

    for axis in axes:
        axis.legend()

    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "convergence_history.svg", format="svg")
    plt.close(figure)


def print_summary(results: list[OptimizationResult]) -> None:
    print("=" * 86)
    print(
        f"{'Method':<20} {'Iterations':>10} {'x1':>16} {'x2':>16} "
        f"{'f(x*)':>12} {'||grad||':>12}"
    )
    print("=" * 86)
    for result in results:
        print(
            f"{result.method:<20} {result.iterations:>10d} "
            f"{result.minimizer[0]:>16.8e} {result.minimizer[1]:>16.8e} "
            f"{result.minimum_value:>12.4e} {result.gradient_norms[-1]:>12.4e}"
        )
    print("=" * 86)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x0 = np.array([1.0, 1.0], dtype=float)
    steepest = steepest_descent(x0=x0, tol=1e-8, max_iter=5000)
    modified = modified_newton(x0=x0, tol=1e-10, max_iter=50)
    results = [steepest, modified]

    for result in results:
        export_history(result)
        export_sampled_history(result)

    export_summary(results)
    plot_contours_with_iterates(steepest, "steepest_descent_contour.svg", "#c0392b")
    plot_contours_with_iterates(modified, "modified_newton_contour.svg", "#0b6e4f")
    plot_path_comparison(results)
    plot_convergence_history(results)
    print_summary(results)


if __name__ == "__main__":
    main()
