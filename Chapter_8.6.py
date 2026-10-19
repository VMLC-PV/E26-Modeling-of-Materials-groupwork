import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Boundary-value problems: shooting, relaxation, and eigenvalues",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from wigglystuff import PlaySlider

    return PlaySlider, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 8.6 Boundary-value problems

    An **initial value problem (IVP)** gives the starting values of the
    variables at one point. A **boundary value problem (BVP)** gives conditions
    at two or more points. We may need to find a missing starting value or
    parameter so the solution meets all those conditions.

    For example, consider a ball thrown vertically. Its height $x(t)$ obeys
    $$
    \frac{d^2 x}{dt^2}=-g.
    $$
    Suppose we know its height at launch and landing,
    $$
    x(0)=0,\qquad x(10)=0,
    $$
    but not its initial velocity $x'(0)$. We must find the velocity that
    makes the ball reach the required height at $t=10$ s.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    g = 9.81
    a = 0.0
    b = 10.0
    N = 1000
    h = (b - a) / N
    target = 1e-10
    x0 = 0.0
    x_target = 0.0

    def f(r):
        return np.array([r[1], -g], dtype=float)

    def trajectory(v):
        times = np.linspace(a, b, N + 1)
        states = np.empty((N + 1, 2))
        states[0] = [x0, v]
        for i in range(N):
            r = states[i]
            k1 = h * f(r)
            k2 = h * f(r + 0.5 * k1)
            k3 = h * f(r + 0.5 * k2)
            k4 = h * f(r + k3)
            states[i + 1] = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return times, states

    def height(v):
        return trajectory(v)[1][-1, 0]

    def endpoint_residual(v):
        return height(v) - x_target

    def plot_residual(v):
        flight_time = b - a
        return x0 + v * flight_time - 0.5 * g * flight_time**2 - x_target

    def solve_initial_velocity():
        v1 = 0.01
        v2 = 1000.0
        h1 = endpoint_residual(v1)
        h2 = endpoint_residual(v2)
        if h1 * h2 >= 0:
            raise ValueError("The initial velocities do not bracket a root.")

        history = []
        for iteration in range(1, 101):
            vp = (v1 + v2) / 2
            _, states = trajectory(vp)
            hp = states[-1, 0] - x_target
            history.append(
                {
                    "lower": v1,
                    "upper": v2,
                    "midpoint": vp,
                    "residual": hp,
                    "states": states,
                }
            )
            if abs(hp) <= target:
                return vp, iteration, hp, history

            if h1 * hp < 0:
                v2, h2 = vp, hp
            else:
                v1, h1 = vp, hp

        raise RuntimeError("Bisection did not converge within 100 iterations.")

    (
        shooting_velocity,
        shooting_iterations,
        shooting_endpoint_residual,
        shooting_bisection_history,
    ) = solve_initial_velocity()

    shooting_velocity_grid = np.linspace(0.01, 1000.0, 200)
    shooting_residual_grid = plot_residual(shooting_velocity_grid)
    shooting_times, _ = trajectory(0.0)
    return (
        N,
        a,
        b,
        g,
        h,
        plot_residual,
        shooting_bisection_history,
        shooting_endpoint_residual,
        shooting_iterations,
        shooting_residual_grid,
        shooting_times,
        shooting_velocity,
        shooting_velocity_grid,
        target,
        x0,
        x_target,
    )


@app.cell(hide_code=True)
def _(PlaySlider, mo, shooting_bisection_history):
    shooting_step = mo.ui.anywidget(
        PlaySlider(
            value=0,
            min_value=0,
            max_value=len(shooting_bisection_history) - 1,
            step=1,
            interval_ms=500,
            loop=False,
        )
    )
    return (shooting_step,)


@app.cell(hide_code=True)
def _(mo):
    shooting_zoom = mo.ui.switch(label="Zoom on current step")
    return (shooting_zoom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Shooting Method

    The shooting method finds the initial velocity needed for a ball to reach a target height.

    $$
    x'=v,\qquad v'=-g
    $$

    - **1. Calculate the trajectory:** Use RK4 with $N=1000$ steps and $h=0.01\,\text{s}$.
    - **2. Compute the endpoint residual:**
      $$
      F(v_0)=x(b;v_0)-x_{\text{target}}
      $$
    - **3. Refine the initial velocity:** Apply bisection between $v_1=0.01$ and $v_2=1000\,\text{m/s}$, where residuals have opposite signs.
    - **Stopping criterion:** $|F(v_0)|<10^{-10}\,\text{m}$.

    **Goal:** Find $v_0$ such that $F(v_0)=0$.
    """)
    return


@app.cell(hide_code=True)
def _(
    a,
    b,
    mo,
    np,
    plot_residual,
    plt,
    shooting_bisection_history,
    shooting_residual_grid,
    shooting_step,
    shooting_times,
    shooting_velocity,
    shooting_velocity_grid,
    shooting_zoom,
    x0,
    x_target,
):
    _frame_play = min(
        int(shooting_step.value["value"]), len(shooting_bisection_history) - 1
    )
    _step_play = shooting_bisection_history[_frame_play]
    _states_play = _step_play["states"]
    _residual_x_play = shooting_velocity_grid
    _residual_y_play = shooting_residual_grid
    if shooting_zoom.value:
        _bracket_width_play = _step_play["upper"] - _step_play["lower"]
        _padding_play = max(0.15 * _bracket_width_play, 0.01)
        _zoom_lower_play = max(0.01, _step_play["lower"] - _padding_play)
        _zoom_upper_play = min(1000.0, _step_play["upper"] + _padding_play)
        _residual_x_play = np.linspace(_zoom_lower_play, _zoom_upper_play, 200)
        _residual_y_play = plot_residual(_residual_x_play)

    _fig_play, (_ax_residual_play, _ax_path_play) = plt.subplots(
        1, 2, figsize=(9, 4)
    )

    _ax_residual_play.plot(
        _residual_x_play,
        _residual_y_play,
        color="steelblue",
        label="Endpoint residual $F(v_0)$",
    )
    _ax_residual_play.axhline(0, color="black", linewidth=1)
    _ax_residual_play.axvspan(
        _step_play["lower"], _step_play["upper"], color="orange", alpha=0.2,
        label="Current bracket",
    )
    _ax_residual_play.scatter(
        [_step_play["midpoint"]], [_step_play["residual"]],
        color="crimson", zorder=3, label="Tested midpoint",
    )
    _ax_residual_play.set(
        xlabel="Trial initial velocity v₀ (m/s)",
        ylabel="Endpoint residual F(v₀) (m)",
        title=(
            "Current bisection step (zoomed)"
            if shooting_zoom.value
            else "Bisection narrows the velocity bracket"
        ),
    )
    if shooting_zoom.value:
        _ax_residual_play.set_xlim(_zoom_lower_play, _zoom_upper_play)
        _y_min_play = min(float(np.min(_residual_y_play)), _step_play["residual"])
        _y_max_play = max(float(np.max(_residual_y_play)), _step_play["residual"])
        _y_padding_play = max((_y_max_play - _y_min_play) * 0.1, 1e-8)
        _ax_residual_play.set_ylim(
            _y_min_play - _y_padding_play, _y_max_play + _y_padding_play
        )
    _ax_residual_play.grid(True, alpha=0.3)
    _ax_residual_play.legend()

    _ax_path_play.plot(
        shooting_times,
        _states_play[:, 0],
        color="steelblue",
        label=f"Trial v₀={_step_play['midpoint']:.6f} m/s",
    )
    _ax_path_play.axhline(
        x_target, linestyle="--", color="black", label="Target height"
    )
    _ax_path_play.scatter(
        [a, b], [x0, x_target], color="black", zorder=3
    )
    _ax_path_play.scatter(
        [b], [_states_play[-1, 0]],
        color="crimson", zorder=4, label="Predicted endpoint",
    )
    _ax_path_play.set(
        xlim=(a, b),
        xlabel="Time (s)",
        ylabel="Height (m)",
        title="Trajectory for this iteration",
    )
    _path_min_play = min(float(np.min(_states_play[:, 0])), x0, x_target)
    _path_max_play = max(float(np.max(_states_play[:, 0])), x0, x_target)
    _path_padding_play = max(
        0.1 * (_path_max_play - _path_min_play), 0.1
    )
    _ax_path_play.set_ylim(
        _path_min_play - _path_padding_play,
        _path_max_play + _path_padding_play,
    )
    _ax_path_play.grid(True, alpha=0.3)
    _ax_path_play.legend()
    plt.close(_fig_play)

    _history_rows = [
        {
            "iteration": _i + 1,
            "lower (m/s)": f"{_row['lower']:.8f}",
            "upper (m/s)": f"{_row['upper']:.8f}",
            "test v₀ (m/s)": f"{_row['midpoint']:.8f}",
            "residual (m)": f"{_row['residual']:.3e}",
        }
        for _i, _row in enumerate(shooting_bisection_history)
    ]
    _body = mo.vstack(
        [
            mo.hstack(
                [
                    mo.md("**Iteration:**"),
                    shooting_step,
                    shooting_zoom,
                ],
                justify="start",
                align="center",
            ),
            mo.hstack(
                [
                    _fig_play,
                    mo.vstack(
                        [
                            mo.stat(
                                label="Test velocity",
                                value=f"{_step_play['midpoint']:.8f} m/s",
                            ),
                            mo.stat(
                                label="Endpoint residual",
                                value=f"{_step_play['residual']:.3e} m",
                            ),
                            mo.stat(
                                label="Final velocity",
                                value=f"{shooting_velocity:.8f} m/s",
                                caption=f"{len(shooting_bisection_history)} total iterations",
                            ),
                        ]
                    ),
                ],
                widths=[0.8, 0.2],
                align="center",
            ),
            mo.accordion(
                {
                    "Bisection iteration table": mo.ui.table(
                        _history_rows, page_size=8, selection=None
                    )
                }
            ),
        ]
    )
    _body
    return


@app.cell(hide_code=True)
def _(
    a,
    b,
    g,
    mo,
    shooting_endpoint_residual,
    shooting_iterations,
    shooting_velocity,
    x0,
    x_target,
):
    _flight_time = b - a
    _exact_velocity = (
        x_target - x0 + 0.5 * g * _flight_time**2
    ) / _flight_time
    _velocity_difference = abs(shooting_velocity - _exact_velocity)
    mo.md(
        f"""
        **Check the numerical answer.** For constant acceleration, the exact
        path is $x(t)=x_0+v_0(t-a)-g(t-a)^2/2$. Applying the two height
        conditions gives $v_0={_exact_velocity:.8f}$ m/s. Bisection gives
        **{shooting_velocity:.8f} m/s**, a difference of
        **{_velocity_difference:.2e} m/s** after {shooting_iterations} steps.
        Its final endpoint residual is **{shooting_endpoint_residual:.2e} m**.
        """
    )
    return


@app.cell(hide_code=True)
def _(N, a, b, g, h, np, target, x0, x_target):
    relaxation_grid = np.linspace(a, b, N + 1)
    _profile_relax = np.linspace(x0, x_target, N + 1)
    _profile_relax[0], _profile_relax[-1] = x0, x_target
    _red_indices_relax = np.arange(1, N, 2)
    _black_indices_relax = np.arange(2, N, 2)
    _omega_relax = 2 / (1 + np.sin(np.pi / N))
    _source_relax = g * h**2
    _tolerance_relax = target
    relaxation_history = [
        {
            "iteration": 0,
            "profile": _profile_relax.copy(),
            "max_update": float("inf"),
        }
    ]

    for _iteration_relax in range(1, 10001):
        _previous_relax = _profile_relax[1:-1].copy()
        for _indices_relax in (_red_indices_relax, _black_indices_relax):
            _local_average_relax = (
                _profile_relax[_indices_relax - 1]
                + _profile_relax[_indices_relax + 1]
                + _source_relax
            ) / 2
            _profile_relax[_indices_relax] = (
                (1 - _omega_relax) * _profile_relax[_indices_relax]
                + _omega_relax * _local_average_relax
            )

        _max_update_relax = float(
            np.max(np.abs(_profile_relax[1:-1] - _previous_relax))
        )
        if _iteration_relax % 100 == 0 or _max_update_relax <= _tolerance_relax:
            relaxation_history.append(
                {
                    "iteration": _iteration_relax,
                    "profile": _profile_relax.copy(),
                    "max_update": _max_update_relax,
                }
            )
        if _max_update_relax <= _tolerance_relax:
            break
    else:
        raise RuntimeError("Relaxation did not converge within 10,000 sweeps.")

    relaxation_solution = _profile_relax.copy()
    relaxation_initial_velocity = (
        -3 * relaxation_solution[0]
        + 4 * relaxation_solution[1]
        - relaxation_solution[2]
    ) / (2 * h)
    _flight_time_relax = b - a
    _reference_velocity_relax = (
        x_target - x0 + 0.5 * g * _flight_time_relax**2
    ) / _flight_time_relax
    relaxation_reference = (
        x0
        + _reference_velocity_relax * (relaxation_grid - a)
        - 0.5 * g * (relaxation_grid - a) ** 2
    )
    relaxation_error = float(
        np.max(np.abs(relaxation_solution - relaxation_reference))
    )
    return (
        relaxation_error,
        relaxation_grid,
        relaxation_history,
        relaxation_initial_velocity,
        relaxation_reference,
        relaxation_solution,
    )


@app.cell(hide_code=True)
def _(PlaySlider, mo, relaxation_history):
    relaxation_step = mo.ui.anywidget(
        PlaySlider(
            value=0,
            min_value=0,
            max_value=len(relaxation_history) - 1,
            step=1,
            interval_ms=350,
            loop=False,
        )
    )
    return (relaxation_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Relaxation Method

    - **Initial guess:** Estimate the full trajectory while keeping both endpoint heights fixed: $(x(a)=x_0=0$, $x(b)=x_{\text{target}}=0$.
    - **Update rule:** Approximate $x''(t_i)=-g$ using neighboring grid points:
      $$
      x_i=\frac{x_{i-1}+x_{i+1}+gh^2}{2}
      $$
    - **Convergence:** Repeat updates until the maximum change per sweep is below $10^{-10}\,\text{m}$.
    - **Acceleration:** Use red-black ordering and successive over-relaxation (SOR), with $\omega$ close to 2.
    - **Initial velocity:** Estimate from the slope at the starting point once the trajectory converges.

    **Key difference:** Shooting adjusts the initial velocity; relaxation adjusts intermediate heights while fixing both endpoints. The exact trajectory (dashed curve) validates the numerical result.
    """)
    return


@app.cell(hide_code=True)
def _(
    a,
    b,
    mo,
    np,
    plt,
    relaxation_error,
    relaxation_grid,
    relaxation_history,
    relaxation_initial_velocity,
    relaxation_reference,
    relaxation_solution,
    relaxation_step,
    x0,
    x_target,
):
    _relax_frame = min(
        int(relaxation_step.value["value"]), len(relaxation_history) - 1
    )
    _relax_state = relaxation_history[_relax_frame]
    _relax_fig, _relax_ax = plt.subplots(figsize=(9, 4))
    _relax_ax.plot(
        relaxation_grid,
        _relax_state["profile"],
        label=f"Relaxation iterate {_relax_state['iteration']}",
        color="steelblue",
    )
    _relax_ax.plot(
        relaxation_grid,
        relaxation_reference,
        "--",
        label="Exact trajectory (comparison)",
        color="black",
    )
    _relax_ax.scatter(
        [a, b], [x0, x_target], color="crimson", zorder=3, label="Fixed boundaries"
    )
    _relax_ax.set(
        xlim=(a, b),
        xlabel="Time (s)",
        ylabel="Height (m)",
        title="How relaxation changes the trajectory",
    )
    _relax_ax.grid(True, alpha=0.3)
    _relax_ax.legend()
    plt.close(_relax_fig)

    mo.vstack(
        [
            mo.hstack(
                [mo.md("**Relaxation sweep:**"), relaxation_step],
                justify="start",
                align="center",
            ),
            mo.hstack(
                [
                    _relax_fig,
                    mo.vstack(
                        [
                            mo.stat(
                                label="Current iteration",
                                value=f"{_relax_state['iteration']:,}",
                            ),
                            mo.stat(
                                label="Maximum difference from exact path",
                                value=f"{relaxation_error:.2e} m",
                            ),
                            mo.stat(
                                label="Distance from final path",
                                value=(
                                    f"{np.max(np.abs(_relax_state['profile'] - relaxation_solution)):.2e} m"
                                ),
                                caption="largest height difference; convergence uses change per sweep",
                            ),
                            mo.stat(
                                label="Final initial velocity",
                                value=f"{relaxation_initial_velocity:.8f} m/s",
                                caption=(
                                    f"{relaxation_history[-1]['iteration']} sweeps; "
                                    f"update tolerance {relaxation_history[-1]['max_update']:.1e} m"
                                ),
                            ),
                        ]
                    ),
                ],
                widths=[0.8, 0.2],
                align="center",
            ),
            mo.md(
                f"The animation saves every 100th sweep "
                f"({len(relaxation_history) - 1} saved steps). The estimated "
                f"initial velocity is **{relaxation_initial_velocity:.8f} m/s**."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Eigenvalue Problems: Linear Homogeneous ODEs

    **Example: A second-order boundary-value problem**

    $$
    \psi''(x)=\frac{2m}{\hbar^2}[V(x)-E]\psi(x)
    $$

    Rewrite as two first-order ODEs:

    $$
    \frac{d\psi}{dx}=\phi,\qquad
    \frac{d\phi}{dx}=\frac{2m}{\hbar^2}[V(x)-E]\psi
    $$

    **Numerical approach: Shooting method**

    1. Apply the boundary condition $\psi(0)=0$.
    2. Guess the initial slope $\phi(0)$ and integrate to $x=L$, e.g. using Runge–Kutta.
    3. Adjust the parameter $E$ until the second boundary condition is satisfied:

    $$
    \psi(L)=0
    $$

    **Key challenge**

    Changing the initial slope only rescales the solution. Instead, vary $E$ and use a root-finding method to solve

    $$
    f(E)=\psi(L;E)=0
    $$

    The roots are the **eigenvalues** for which both boundary conditions are satisfied.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    well_electron_mass = 9.1094e-31
    well_hbar = 1.0546e-34
    well_electron_charge = 1.6022e-19
    well_width = 5.2918e-11
    well_potential_scale = 1000.0
    well_steps = 1000
    well_step = 1.0 / well_steps
    well_z = np.linspace(0.0, 1.0, well_steps + 1)
    well_energy_scale = (
        2
        * well_electron_mass
        * well_width**2
        * well_electron_charge
        / well_hbar**2
    )
    well_free_energy_scale = (
        np.pi**2
        * well_hbar**2
        / (2 * well_electron_mass * well_width**2)
        / well_electron_charge
    )

    def well_potential(positions, potential_model):
        if potential_model == "Parabolic potential (V0 = 1000 eV)":
            return well_potential_scale * positions * (positions - 1.0)
        if potential_model == "Zero potential (V = 0)":
            return np.zeros_like(positions, dtype=float)
        raise ValueError(f"Unknown potential model: {potential_model}")

    def integrate_well_energy(energy_ev, potential_model):
        state = np.empty((well_steps + 1, 2))
        state[0] = [0.0, 1.0]

        def rhs(current, z):
            psi, slope = current
            potential = well_potential(z, potential_model)
            curvature = well_energy_scale * (potential - energy_ev) * psi
            return np.array([slope, curvature])

        for index in range(well_steps):
            z = well_z[index]
            current = state[index]
            k1 = well_step * rhs(current, z)
            k2 = well_step * rhs(current + k1 / 2, z + well_step / 2)
            k3 = well_step * rhs(current + k2 / 2, z + well_step / 2)
            k4 = well_step * rhs(current + k3, z + well_step)
            state[index + 1] = current + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            scale = np.max(np.abs(state[index + 1]))
            if scale > 1e50 or (0 < scale < 1e-50):
                state[: index + 2] /= scale
        return state

    def well_node_count(state):
        signs = np.sign(state[1:-1, 0])
        signs = signs[signs != 0]
        return int(np.count_nonzero(signs[1:] != signs[:-1]))

    def solve_well_spectrum(potential_model):
        mode_count = 2
        lower_bound = (
            -well_potential_scale / 4
            if potential_model == "Parabolic potential (V0 = 1000 eV)"
            else 0.0
        )
        upper_bound = mode_count**2 * well_free_energy_scale + 100.0
        if well_node_count(
            integrate_well_energy(lower_bound, potential_model)
        ) != 0:
            raise RuntimeError("The lower energy bound does not bracket the spectrum.")
        if well_node_count(
            integrate_well_energy(upper_bound, potential_model)
        ) < mode_count:
            raise RuntimeError("The upper energy bound does not bracket two states.")

        spectrum = []
        for mode in range(mode_count):
            lower = lower_bound
            upper = upper_bound
            for _ in range(60):
                midpoint = (lower + upper) / 2
                state = integrate_well_energy(midpoint, potential_model)
                if well_node_count(state) <= mode:
                    lower = midpoint
                else:
                    upper = midpoint
                if upper - lower < 1e-7:
                    break

            energy = (lower + upper) / 2
            state = integrate_well_energy(energy, potential_model)
            wavefunction = state[:, 0]
            spectrum.append(
                {
                    "mode": mode,
                    "energy": energy,
                    "wavefunction": wavefunction,
                    "nodes": well_node_count(state),
                    "residual": float(wavefunction[-1]),
                }
            )
        return spectrum

    return (
        integrate_well_energy,
        well_potential,
        solve_well_spectrum,
        well_z,
    )


@app.cell(hide_code=True)
def _(mo):
    well_potential_selector = mo.ui.dropdown(
        options=[
            "Zero potential (V = 0)",
            "Parabolic potential (V0 = 1000 eV)",
        ],
        value="Zero potential (V = 0)",
        label="Potential model",
    )
    well_solve_button = mo.ui.run_button(label="Solve eigenstates")
    return well_potential_selector, well_solve_button


@app.cell(hide_code=True)
def _(
    mo,
    np,
    plt,
    solve_well_spectrum,
    well_potential,
    well_potential_selector,
    well_solve_button,
    well_z,
):
    _potential_model = well_potential_selector.value
    _control_panel = mo.hstack(
        [well_potential_selector, well_solve_button],
        widths=[0.72, 0.28],
        align="end",
    )
    _intro = mo.md(
        r"""
        <div style="padding: 1.5rem 1.75rem; border-radius: 18px; background: linear-gradient(120deg, #102a43, #176b87); color: white;">
          <div style="font-size: 0.75rem; letter-spacing: 0.16em; text-transform: uppercase; opacity: 0.8;">BOUND STATES / SHOOTING METHOD</div>
          <h2 style="margin: 0.45rem 0; color: white;">Find the allowed energy levels</h2>
          <div style="max-width: 48rem; opacity: 0.9;">Choose a potential, then solve for the first two states that satisfy both hard-wall boundary conditions.</div>
        </div>
        """
    )
    _controls = mo.vstack(
        [
            _control_panel,
            mo.md(
                r"""
                **Potential choices:** zero potential, \(V(x)=0\), or the
                parabolic well,
                \[
                V(x)=V_0\left(\frac{x}{L}\right)
                \left(\frac{x}{L}-1\right),\qquad V_0=1000\,\mathrm{eV}.
                \]
                """
            ),
        ],
        gap=0.5,
    )

    if not well_solve_button.value:
        _results = mo.callout(
            mo.md("Select a potential and press **Solve eigenstates** to calculate its spectrum."),
            kind="info",
        )
    else:
        _spectrum = solve_well_spectrum(_potential_model)
        _potential_ev = well_potential(well_z, _potential_model)
        _colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(_spectrum)))
        _fig, (_energy_ax, _wave_ax) = plt.subplots(
            2,
            1,
            figsize=(11.5, 7.5),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1.2]},
            constrained_layout=True,
        )
        _energy_ax.fill_between(
            well_z,
            _potential_ev,
            0,
            color="#93c5fd",
            alpha=0.35,
            label="Potential $V(x)$",
        )
        _energy_ax.plot(well_z, _potential_ev, color="#2563eb", linewidth=2.4)

        for _state, _color in zip(_spectrum, _colors):
            _energy = _state["energy"]
            _wave = _state["wavefunction"]
            _energy_ax.axhline(
                _energy,
                color=_color,
                linestyle=":",
                label=f"n={_state['mode'] + 1}: E={_energy:.2f} eV",
            )
            _wave_ax.plot(
                well_z,
                _wave,
                color=_color,
                linewidth=2,
                label=f"n={_state['mode'] + 1} ({_state['nodes']} nodes)",
            )

        _energy_ax.set(
            xlim=(0, 1),
            ylim=(
                min(float(np.min(_potential_ev)), _spectrum[0]["energy"])
                - 25,
                _spectrum[-1]["energy"] + 75,
            ),
            ylabel="Energy (eV)",
            title="Potential and first two eigenenergy levels",
        )
        _energy_ax.spines[["top", "right"]].set_visible(False)
        _energy_ax.grid(axis="y", alpha=0.2)
        _energy_ax.legend(loc="upper left", frameon=False, ncol=2)
        _wave_ax.axhline(0, color="#64748b", linewidth=1)
        _wave_ax.set(
            xlim=(0, 1),
            xlabel="Position inside the well, $x/L$",
            ylabel="Wavefunction $\\psi(x)$",
            title="Unnormalized shooting solutions",
        )
        _wave_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        _wave_ax.spines[["top", "right"]].set_visible(False)
        _wave_ax.grid(alpha=0.2)
        _wave_ax.legend(loc="upper right", frameon=False, ncol=2)
        plt.close(_fig)

        _state_cards = [
            mo.stat(
                label=f"State n={_state['mode'] + 1}",
                value=f"{_state['energy']:.2f} eV",
                caption=(
                    f"{_state['nodes']} interior nodes | "
                    f"|psi(L)|={abs(_state['residual']):.1e}"
                ),
            )
            for _state in _spectrum
        ]
        _results = mo.vstack(
            [
                mo.hstack(_state_cards, widths=[1, 1]),
                _fig,
                mo.md(
                    "The lower panel shows the RK4 shooting solutions at their "
                    "computed amplitudes, with no display normalization. An eigenstate "
                    "has a node at the far wall; the reported residual measures how "
                    "closely that boundary condition is met."
                ),
            ],
            gap=1,
        )

    mo.vstack(
        [
            _intro,
            _controls,
            _results,
        ],
        gap=1.2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
            ## Choosing a method
            ---

            | Method | What do we adjust? | How do we recognize a solution? |
            |---|---|---|
            | Shooting | The starting slope | The calculated path reaches the far boundary |
            | Relaxation | The values between the endpoints | The heights stop changing from one sweep to the next |
            | Eigenvalue shooting | The energy | The calculated wave is zero at the far wall |

            Use shooting when you can calculate the solution from a starting
            point and only need to adjust a few starting values. Use relaxation
            when you want to solve for the values across the whole grid. For a
            quantum eigenvalue problem, use shooting to find the energy that
            makes the wave meet both wall conditions.
            """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
