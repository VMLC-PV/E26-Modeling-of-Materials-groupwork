import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    return Rectangle, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The relaxation method
    ### Chapter 9.1.2

    Turn a continuous boundary-value problem into a grid—and iterate to equilibrium.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 9.1: potential inside a square

    In a charge-free region, electric potential satisfies Laplace's equation:

    $$
    \nabla^2\phi =
    \frac{\partial^2\phi}{\partial x^2}
    + \frac{\partial^2\phi}{\partial y^2}=0.
    $$

    **Square box:** $1\ \mathrm{m}\times1\ \mathrm{m}$

    - Top edge: $1\ \mathrm{V}$
    - Other edges: $0\ \mathrm{V}$
    - **Find:** the potential inside the box
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From equation to grid

    Replace each second derivative with a central difference on a grid of spacing $a$:

    $$
    \frac{\phi_{i+1,j}-2\phi_{i,j}+\phi_{i-1,j}}{a^2}+\frac{\phi_{i,j+1}-2\phi_{i,j}+\phi_{i,j-1}}{a^2}=0.
    $$

    Multiplying by $a^2$ gives the discrete Laplace equation:

    $$
    \phi_{i+1,j}+\phi_{i-1,j}+\phi_{i,j+1}+\phi_{i,j-1}-4\phi_{i,j}=0.
    $$

    A differential equation becomes a local relation between neighboring grid values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The four-neighbor rule

    Rearranging gives the value at the center:

    $$
    \boxed{\phi_{i,j}=\frac{1}{4}\left(\phi_{i+1,j}+\phi_{i-1,j}+\phi_{i,j+1}+\phi_{i,j-1}\right)}
    $$

    **At equilibrium, each interior value is the average of its four nearest neighbors.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fix the boundary. Guess the interior.

    - Keep the wall voltages fixed: top $1\ \mathrm{V}$; other edges $0\ \mathrm{V}$.
    - Initialize the interior, for example, to $0\ \mathrm{V}$.
    - Repeatedly apply the four-neighbor rule.

    **Grid resolution:** $a=1\ \mathrm{cm}$ gives 100 squares (101 points) per side.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Jacobi: update from the old grid

    Every new value uses neighbors from the **previous grid**:

    $$
        \phi'_{i,j}=\frac{1}{4}\left(\phi_{i+1,j}+\phi_{i-1,j}+\phi_{i,j+1}+\phi_{i,j-1}\right).
    $$

    - Read the old grid; write to a separate new grid.
    - Update all interior points, then swap grids.
    """)
    return


@app.cell
def _(np):
    _M_example = 100
    _V_example = 1.0
    _target_example = 1e-6
    _phi_example = np.zeros((_M_example + 1, _M_example + 1), dtype=float)
    _phi_example[0, :] = _V_example
    _delta_example = 1.0
    _iteration_example = 0
    _frame_steps_example = {0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000}
    _samples_example = {}
    _frame_snapshots_example = {0: _phi_example.copy()}
    _frame_deltas_example = {0: _delta_example}
    _iteration_history_example = [0]
    _delta_history_example = [_delta_example]

    while _delta_example > _target_example:
        _phiprime_example = _phi_example.copy()
        _phiprime_example[1:-1, 1:-1] = (
            _phi_example[2:, 1:-1]
            + _phi_example[:-2, 1:-1]
            + _phi_example[1:-1, 2:]
            + _phi_example[1:-1, :-2]
        ) / 4.0
        _delta_example = float(
            np.max(np.abs(_phi_example - _phiprime_example))
        )
        _phi_example, _phiprime_example = _phiprime_example, _phi_example
        _iteration_example += 1
        _iteration_history_example.append(_iteration_example)
        _delta_history_example.append(_delta_example)
        if _iteration_example in _frame_steps_example:
            _frame_snapshots_example[_iteration_example] = _phi_example.copy()
            _frame_deltas_example[_iteration_example] = _delta_example
        if _iteration_example in {1, 10, 100, 1000, 5000, 10000}:
            _samples_example[_iteration_example] = (
                float(_phi_example[1, 50]),
                float(_phi_example[50, 50]),
            )

    _frame_snapshots_example[_iteration_example] = _phi_example.copy()
    _frame_deltas_example[_iteration_example] = _delta_example
    relaxation_steps = sorted(_frame_snapshots_example)
    relaxation_fields = _frame_snapshots_example
    relaxation_changes = _frame_deltas_example
    relaxation_iteration_history = _iteration_history_example
    relaxation_change_history = _delta_history_example
    relaxation_point_samples = _samples_example
    relaxation_final_iteration = _iteration_example
    relaxation_final_change = _delta_example
    relaxation_final_field = _phi_example
    relaxation_target = _target_example
    return (
        relaxation_change_history,
        relaxation_changes,
        relaxation_fields,
        relaxation_iteration_history,
        relaxation_steps,
        relaxation_target,
    )


@app.cell
def _(mo, relaxation_steps):
    relaxation_snapshot_index, set_relaxation_snapshot_index = mo.state(0)
    relaxation_step_slider = mo.ui.slider(
        start=0,
        stop=len(relaxation_steps) - 1,
        value=relaxation_snapshot_index(),
        step=1,
        label="Relaxation snapshot",
        show_value=True,
        on_change=set_relaxation_snapshot_index,
    )

    def _advance_relaxation_snapshot(_click_count):
        set_relaxation_snapshot_index(
            min(relaxation_snapshot_index() + 1, len(relaxation_steps) - 1)
        )
        return _click_count + 1

    relaxation_next_button = mo.ui.button(
        label="Next snapshot →",
        on_click=_advance_relaxation_snapshot,
        disabled=relaxation_snapshot_index() >= len(relaxation_steps) - 1,
        tooltip="Advance to the next saved relaxation step",
    )
    return (
        relaxation_next_button,
        relaxation_snapshot_index,
        relaxation_step_slider,
    )


@app.cell
def _(
    mo,
    plt,
    relaxation_change_history,
    relaxation_changes,
    relaxation_fields,
    relaxation_iteration_history,
    relaxation_next_button,
    relaxation_snapshot_index,
    relaxation_step_slider,
    relaxation_steps,
    relaxation_target,
):
    _selected_step_example = relaxation_steps[relaxation_snapshot_index()]
    _selected_field_example = relaxation_fields[_selected_step_example]
    _figure_example, (_field_axes_example, _change_axes_example) = plt.subplots(
        1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    _field_image_example = _field_axes_example.imshow(
        _selected_field_example,
        origin="upper",
        extent=(0, 1, 0, 1),
        vmin=0,
        vmax=1,
        cmap="magma",
        interpolation="nearest",
    )
    _field_axes_example.set(
        title=f"Potential field · sweep {_selected_step_example:,}",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    _figure_example.colorbar(
        _field_image_example, ax=_field_axes_example, label="Potential (V)"
    )
    _visible_iterations_example = [
        _step for _step in relaxation_iteration_history if 0 < _step <= _selected_step_example
    ]
    _visible_changes_example = relaxation_change_history[
        1 : _selected_step_example + 1
    ]
    _change_axes_example.plot(
        _visible_iterations_example,
        _visible_changes_example,
        color="#2878b5",
        linewidth=2,
    )
    _change_axes_example.set(
        title="Largest change per sweep",
        xlabel="Jacobi sweeps",
        ylabel="Maximum change, Δ (V)",
        xscale="log",
        yscale="log",
        xlim=(1, max(relaxation_steps[-1], 2)),
        ylim=(relaxation_target / 2, 2),
    )
    _change_axes_example.axhline(
        relaxation_target,
        color="#d1495b",
        linestyle="--",
        label=f"target = {relaxation_target:.0e} V",
    )
    _change_axes_example.plot(
        [_selected_step_example if _selected_step_example > 0 else 1],
        [relaxation_changes[_selected_step_example]],
        marker="o",
        color="#f4a261",
        zorder=3,
    )
    _change_axes_example.legend(loc="upper right")
    mo.vstack(
        [
            mo.hstack([relaxation_step_slider, relaxation_next_button], align="end"),
            mo.mpl.interactive(_figure_example),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # To be continued in 9.2
    Jacobi is simple and reliable, but convergence can be slow on fine grids.

    Each sweep updates every interior point; many sweeps may be needed before the largest change falls below $\epsilon$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overrelaxation: take a larger step

    Let $\phi'$ be the ordinary relaxed value. The update is

    $$
    \begin{aligned}
    \Delta\phi(x,y) &= \phi'(x,y)-\phi(x,y), \\
    \phi_\omega(x,y)
      &= \phi(x,y)+(1+\omega)\Delta\phi(x,y) \\
      &= (1+\omega)\phi'(x,y)-\omega\phi(x,y).
    \end{aligned}
    $$

    Here $\omega>0$ controls the overshoot: $\omega=0.9$ takes $1.9$ times the ordinary step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overrelaxed Jacobi with alternating errors

    A checkerboard alternates between high and low values. Since each point is surrounded by the opposite values, the pattern flips each round.

    With ordinary Jacobi, the error slowly fades. But an extra-large step can push the values too far: in this example, the error nearly doubles each round. Instead of settling down, the calculation moves further from a solution.
    """)
    return


@app.cell
def _(mo, np, plt):
    _interior_points_instability = 20
    _omega_instability = 0.5
    _sweeps_instability = 18
    _indices_instability = np.arange(1, _interior_points_instability + 1)
    _profile_instability = np.sin(
        np.pi * _indices_instability / (_interior_points_instability + 1)
    )
    _checkerboard_instability = np.where(
        (_indices_instability[:, None] + _indices_instability[None, :]) % 2 == 0,
        1.0,
        -1.0,
    )
    _initial_mode_instability = _checkerboard_instability * np.outer(
        _profile_instability, _profile_instability
    )
    _initial_mode_instability /= np.max(np.abs(_initial_mode_instability))

    _jacobi_error_instability = np.zeros(
        (_interior_points_instability + 2, _interior_points_instability + 2)
    )
    _jacobi_error_instability[1:-1, 1:-1] = _initial_mode_instability
    _overrelaxed_error_instability = _jacobi_error_instability.copy()
    _jacobi_max_error_instability = [1.0]
    _overrelaxed_max_error_instability = [1.0]

    for _ in range(_sweeps_instability):
        _jacobi_average_instability = (
            _jacobi_error_instability[2:, 1:-1]
            + _jacobi_error_instability[:-2, 1:-1]
            + _jacobi_error_instability[1:-1, 2:]
            + _jacobi_error_instability[1:-1, :-2]
        ) / 4.0
        _jacobi_next_instability = _jacobi_error_instability.copy()
        _jacobi_next_instability[1:-1, 1:-1] = _jacobi_average_instability
        _jacobi_error_instability = _jacobi_next_instability

        _overrelaxed_average_instability = (
            _overrelaxed_error_instability[2:, 1:-1]
            + _overrelaxed_error_instability[:-2, 1:-1]
            + _overrelaxed_error_instability[1:-1, 2:]
            + _overrelaxed_error_instability[1:-1, :-2]
        ) / 4.0
        _overrelaxed_error_instability[1:-1, 1:-1] += (
            (1.0 + _omega_instability)
            * (
                _overrelaxed_average_instability
                - _overrelaxed_error_instability[1:-1, 1:-1]
            )
        )

        _jacobi_max_error_instability.append(
            float(np.max(np.abs(_jacobi_error_instability)))
        )
        _overrelaxed_max_error_instability.append(
            float(np.max(np.abs(_overrelaxed_error_instability)))
        )

    _jacobi_eigenvalue_instability = -np.cos(
        np.pi / (_interior_points_instability + 1)
    )
    _overrelaxed_eigenvalue_instability = (
        (1.0 + _omega_instability) * _jacobi_eigenvalue_instability
        - _omega_instability
    )
    _iteration_numbers_instability = np.arange(_sweeps_instability + 1)
    _figure_instability, (
        _mode_axes_instability,
        _error_axes_instability,
    ) = plt.subplots(1, 2, figsize=(10, 4))

    _mode_image_instability = _mode_axes_instability.imshow(
        _initial_mode_instability,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    _mode_axes_instability.set(
        title="Initial checkerboard error mode",
        xlabel="Interior grid column",
        ylabel="Interior grid row",
    )
    _figure_instability.colorbar(
        _mode_image_instability,
        ax=_mode_axes_instability,
        label="Normalized error",
    )

    _error_axes_instability.semilogy(
        _iteration_numbers_instability,
        _jacobi_max_error_instability,
        marker="o",
        label="Jacobi ($\\omega=0$)",
    )
    _error_axes_instability.semilogy(
        _iteration_numbers_instability,
        _overrelaxed_max_error_instability,
        marker="o",
        label=f"Overrelaxed Jacobi ($\\omega={_omega_instability}$)",
    )
    _error_axes_instability.semilogy(
        _iteration_numbers_instability,
        np.abs(_jacobi_eigenvalue_instability) ** _iteration_numbers_instability,
        linestyle="--",
        color="#2878b5",
        alpha=0.7,
        label="Jacobi eigenvalue prediction",
    )
    _error_axes_instability.semilogy(
        _iteration_numbers_instability,
        np.abs(_overrelaxed_eigenvalue_instability)
        ** _iteration_numbers_instability,
        linestyle="--",
        color="#d1495b",
        alpha=0.7,
        label="Overrelaxed eigenvalue prediction",
    )
    _error_axes_instability.set(
        title="Maximum error after each sweep",
        xlabel="Jacobi sweeps",
        ylabel="Maximum absolute error",
    )
    _error_axes_instability.grid(True, which="both", alpha=0.25)
    _error_axes_instability.legend(fontsize=8)
    _figure_instability.tight_layout()
    mo.mpl.interactive(_figure_instability)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gauss–Seidel: reuse the newest values

    Sweep through the grid and update the same array immediately.

    - Above and left: already updated this sweep.
    - Below and right: still from the previous sweep.
    - Boundaries stay fixed; only one potential array is needed.
    """)
    return


@app.cell
def _(mo):
    from wigglystuff import PlaySlider

    gauss_seidel_sweep_step = mo.ui.anywidget(
        PlaySlider(
            value=0,
            min_value=0,
            max_value=10,
            step=1,
            interval_ms=500,
            loop=True,
            width=440,
        )
    )

    def _advance_gauss_seidel_sweep(_click_count):
        _widget = gauss_seidel_sweep_step.widget
        _widget.value = (_widget.value + 1) % 11
        return _click_count + 1

    gauss_seidel_next_step = mo.ui.button(
        label="Next step →",
        on_click=_advance_gauss_seidel_sweep,
        tooltip="Advance the Gauss–Seidel animation by one point",
    )
    return gauss_seidel_next_step, gauss_seidel_sweep_step


@app.cell
def _(Rectangle, gauss_seidel_next_step, gauss_seidel_sweep_step, mo, plt):
    # Create figure once
    if "_fig" not in globals():
        _fig, _ax = plt.subplots(figsize=(6, 6))

        _colors = {
            "boundary": "#dbeafe",
            "old": "#edf0f4",
            "current": "#f5cf68",
            "updated": "#8bd0b5",
        }

        _patches = []
        _labels = []

        # 5x5 grid, including fixed boundary
        for row in range(5):
            for col in range(5):
                boundary = (
                    row == 0 or row == 4 or
                    col == 0 or col == 4
                )

                point = None if boundary else (
                    (row - 1) * 3 + col - 1
                )

                label = (
                    "FIXED" if boundary else f"P{point + 1}"
                )

                patch = Rectangle(
                    (col, row), 0.95, 0.95,
                    facecolor=_colors[
                        "boundary" if boundary else "old"
                    ],
                    edgecolor="#8793a1",
                    linewidth=1.2
                )

                _ax.add_patch(patch)

                text = _ax.text(
                    col + 0.475,
                    row + 0.475,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold" if boundary else "normal"
                )

                _patches.append((patch, point))
                _labels.append(text)

        _ax.set(
            xlim=(-0.1, 5),
            ylim=(5, -0.1),
            aspect="equal",
            xticks=[],
            yticks=[]
        )

        _ax.set_title(
            "Gauss–Seidel: in-place row-major sweep",
            fontsize=13,
            pad=15
        )

        _status_text = _ax.text(
            0.5, -0.07, "",
            transform=_ax.transAxes,
            ha="center",
            fontsize=10
        )

        _formula_text = _ax.text(
            0.5, -0.14, "",
            transform=_ax.transAxes,
            ha="center",
            fontsize=10
        )

        _fig.tight_layout()

    # Current sweep position
    _frame = int(gauss_seidel_sweep_step.value["value"])
    _current = _frame - 1

    # Determine current point's actual grid coordinates
    if 0 <= _current < 9:
        _row = (_current // 3) + 1
        _col = (_current % 3) + 1

        # All four immediate neighbors, including boundaries
        _neighbors = {
            (_row - 1, _col),  # Above
            (_row + 1, _col),  # Below
            (_row, _col - 1),  # Left
            (_row, _col + 1),  # Right
        }
    else:
        _neighbors = set()

    # Update grid
    for patch, point in _patches:

        # Recover coordinates for EVERY cell, including boundaries
        # Rectangle coordinates are stored in the patch itself
        x, y = patch.get_xy()
        r = int(y)
        c = int(x)

        is_neighbor = (r, c) in _neighbors

        # Determine fill color independently of highlighting
        if point is None:
            status = "boundary"
        elif point < _current:
            status = "updated"
        elif point == _current:
            status = "current"
        else:
            status = "old"

        patch.set_facecolor(_colors[status])

        # Thick dark border around all four stencil neighbors
        patch.set_edgecolor(
            "#111827" if is_neighbor else "#8793a1"
        )

        patch.set_linewidth(
            3.5 if is_neighbor else (
                2.5 if status == "current" else 1.2
            )
        )

    # Update labels
    for label, (_, point) in zip(_labels, _patches):
        if point is not None:
            label.set_fontweight(
                "bold" if (
                    point == _current or
                    (
                        (point // 3 + 1, point % 3 + 1)
                        in _neighbors
                    )
                ) else "normal"
            )

    # Update explanation
    if _frame == 0:
        _status_text.set_text(
            "Ready: select Next step to begin."
        )
        _formula_text.set_text("")

    elif _frame == 10:
        _status_text.set_text(
            "Sweep complete: all 9 interior points updated."
        )
        _formula_text.set_text(
            "Boundary values remain fixed."
        )

    else:
        _status_text.set_text(
            f"Updating P{_current + 1}: dark borders show all four neighbors."
        )
        _formula_text.set_text(
            "Above + left: newest values | Below + right: old values"
        )

    mo.vstack([
        mo.md("""
        **Gauss–Seidel | In-place update**

        Replace each point with its four-neighbor average. Use the newest values already visited; retain old values elsewhere.
        """),
        mo.hstack([gauss_seidel_sweep_step, gauss_seidel_next_step]),
        mo.mpl.interactive(_fig),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Successive Overrelaxation (SOR): speed up Gauss–Seidel

    $$
    \phi \leftarrow \phi + (1+\omega)(\bar\phi_{\rm GS}-\phi)
    $$

    $\bar\phi_{\rm GS}$ is the neighbor average, using the newest values available.

    - $\omega=0$: ordinary Gauss–Seidel
    - $0<\omega<1$: step past the average to reduce error faster

    **Why it works:** Gauss–Seidel reuses each update immediately. This
    damps the checkerboard error that can grow when Jacobi overrelaxes.
    The step size matters: too much overshoot can still cause instability.
    """)
    return


@app.cell
def _(mo, np, plt):
    _comparison_points = 31
    _comparison_interior = _comparison_points - 2
    _comparison_tolerance = 1e-15
    _comparison_max_sweeps = 10_000
    _comparison_omega_sor = (
        2.0 / (1.0 + np.sin(np.pi / (_comparison_interior + 1)))
    )
    _comparison_extra_omega = _comparison_omega_sor - 1.0
    _comparison_initial = np.zeros(
        (_comparison_points, _comparison_points), dtype=float
    )
    _comparison_initial[0, :] = 1.0

    def _comparison_residual(_field):
        _laplacian = (
            _field[2:, 1:-1]
            + _field[:-2, 1:-1]
            + _field[1:-1, 2:]
            + _field[1:-1, :-2]
            - 4.0 * _field[1:-1, 1:-1]
        ) / 4.0
        return float(np.max(np.abs(_laplacian)))

    _jacobi_comparison = _comparison_initial.copy()
    _jacobi_residuals = [_comparison_residual(_jacobi_comparison)]
    for _ in range(_comparison_max_sweeps):
        _jacobi_next = _jacobi_comparison.copy()
        _jacobi_next[1:-1, 1:-1] = (
            _jacobi_comparison[2:, 1:-1]
            + _jacobi_comparison[:-2, 1:-1]
            + _jacobi_comparison[1:-1, 2:]
            + _jacobi_comparison[1:-1, :-2]
        ) / 4.0
        _jacobi_comparison = _jacobi_next
        _jacobi_residuals.append(_comparison_residual(_jacobi_comparison))
        if _jacobi_residuals[-1] <= _comparison_tolerance:
            break
    else:
        raise RuntimeError("Jacobi did not reach the residual tolerance.")

    _gauss_seidel_comparison = _comparison_initial.copy()
    _gauss_seidel_residuals = [
        _comparison_residual(_gauss_seidel_comparison)
    ]
    for _ in range(_comparison_max_sweeps):
        for _row in range(1, _comparison_points - 1):
            for _column in range(1, _comparison_points - 1):
                _gauss_seidel_comparison[_row, _column] = (
                    _gauss_seidel_comparison[_row - 1, _column]
                    + _gauss_seidel_comparison[_row + 1, _column]
                    + _gauss_seidel_comparison[_row, _column - 1]
                    + _gauss_seidel_comparison[_row, _column + 1]
                ) / 4.0
        _gauss_seidel_residuals.append(
            _comparison_residual(_gauss_seidel_comparison)
        )
        if _gauss_seidel_residuals[-1] <= _comparison_tolerance:
            break
    else:
        raise RuntimeError("Gauss–Seidel did not reach the residual tolerance.")

    _sor_comparison = _comparison_initial.copy()
    _sor_residuals = [_comparison_residual(_sor_comparison)]
    for _ in range(_comparison_max_sweeps):
        for _row in range(1, _comparison_points - 1):
            for _column in range(1, _comparison_points - 1):
                _sor_average = (
                    _sor_comparison[_row - 1, _column]
                    + _sor_comparison[_row + 1, _column]
                    + _sor_comparison[_row, _column - 1]
                    + _sor_comparison[_row, _column + 1]
                ) / 4.0
                _sor_comparison[_row, _column] += _comparison_omega_sor * (
                    _sor_average - _sor_comparison[_row, _column]
                )
        _sor_residuals.append(_comparison_residual(_sor_comparison))
        if _sor_residuals[-1] <= _comparison_tolerance:
            break
    else:
        raise RuntimeError("SOR did not reach the residual tolerance.")

    _comparison_figure, _comparison_axes = plt.subplots(figsize=(8, 5))
    for _residuals, _label, _color in (
        (_jacobi_residuals, "Jacobi", "#2878b5"),
        (_gauss_seidel_residuals, "Gauss–Seidel", "#f4a261"),
        (
            _sor_residuals,
            f"SOR (extra $\\omega={_comparison_extra_omega:.2f}$)",
            "#2a9d8f",
        ),
    ):
        _comparison_axes.semilogy(
            range(len(_residuals)),
            _residuals,
            label=f"{_label}: {len(_residuals) - 1:,} sweeps",
            color=_color,
            linewidth=2,
        )

    _comparison_axes.axhline(
        _comparison_tolerance,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"common tolerance = {_comparison_tolerance:.0e}",
    )
    _comparison_axes.set(
        title=f"Convergence on a {_comparison_points} × {_comparison_points} grid",
        xlabel="Complete sweeps",
        ylabel="Maximum discrete Laplace residual (V)",
        xlim=(0, None),
        ylim=(_comparison_tolerance / 2, None),
    )
    _comparison_axes.grid(True, which="both", alpha=0.25)
    _comparison_axes.legend()
    _comparison_figure.tight_layout()
    mo.mpl.interactive(_comparison_figure)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    - **Jacobi** is simple, but reads only the previous grid and converges
      slowly.
    - **Gauss–Seidel** reuses the latest left and above values and stores the
      field in one array.
    - **SOR** adds a controlled overshoot to that in-place update. It reaches the stopping threshold
      in far fewer sweeps than Jacobi.
    - **Overrelaxation is not automatically stable**: using the same
      overshoot with Jacobi can amplify error and prevent convergence.
    """)
    return


if __name__ == "__main__":
    app.run()
