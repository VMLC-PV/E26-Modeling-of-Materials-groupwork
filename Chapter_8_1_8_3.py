import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Ordinary Differential Equations (8.1 & 8.3)",
    layout_file="layouts/ode_chapter8_presentation.slides.json",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ordinary Differential Equations
    ### Chapter 8 — Sections 8.1 and 8.3

    **8.1** First-order equations with one variable
    Euler → Runge–Kutta (2nd and 4th order) → infinite ranges

    **8.3** Second-order equations
    One trick turns them into first-order systems → the nonlinear pendulum
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8.1 First-order equations with one variable

    The general form:

    $$\frac{dx}{dt} = f(x,t)$$

    plus an initial (or boundary) condition, e.g. the value of $x$ at $t = a$.

    Example: $\dfrac{dx}{dt} = \dfrac{2x}{t} + \dfrac{3x^2}{t^3}$ is **nonlinear** and not separable,
    so there is no easy analytic solution. Computers don't care — the numerical techniques are the same.

    The independent variable is called $t$ here, but it doesn't have to be time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8.1.1 Euler's method

    Taylor-expand $x$ a short interval $h$ later:

    $$x(t+h) = x(t) + h\frac{dx}{dt} + \tfrac12 h^2\frac{d^2x}{dt^2}+\dots
    = x(t) + h f(x,t) + O(h^2)$$

    Drop the $h^2$ terms:

    $$\boxed{x(t+h) = x(t) + h f(x,t)}$$

    - Take the slope at $t$ and extrapolate it forward.
    - Error per step: $O(h^2)$. Over $N=(b-a)/h$ steps the total error is **$O(h)$** (first-order method).
    - Halving $h$ halves the error, but doubles the run time.
    """)
    return


@app.cell
def _(np):
    # ---- The methods: each one advances r(t) by a single step of size h ----

    def euler_step(f, r, t, h):
        return r + h * f(r, t)

    def rk2_step(f, r, t, h):
        k1 = h * f(r, t)
        k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
        return r + k2

    def rk4_step(f, r, t, h):
        k1 = h * f(r, t)
        k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
        k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
        k4 = h * f(r + k3, t + h)
        return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def integrate(step, f, r0, a, b, N):
        """Apply `step` N times from t=a to t=b. Returns N+1 times and states."""
        h = (b - a) / N
        ts = a + h * np.arange(N + 1)
        r = np.array(r0, dtype=float)
        rs = [r.copy()]
        for t in ts[:-1]:
            r = step(f, r, t, h)
            rs.append(r.copy())
        return ts, np.array(rs)

    # Example 8.1:  dx/dt = -x^3 + sin(t),  x(0) = 0
    def f_ex(x, t):
        return -(x**3) + np.sin(t)


    return euler_step, f_ex, integrate, rk2_step, rk4_step


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 8.1 — Euler's method

    $$\frac{dx}{dt} = -x^3 + \sin t,\qquad x(0)=0,\quad t\in[0,10]$$
    """)
    return


@app.cell
def _(mo):
    n_euler = mo.ui.slider(5, 500, value=20, step=5, label="N (steps)", show_value=True)
    return (n_euler,)


@app.cell
def _(euler_step, f_ex, integrate, mo, n_euler, plt):

    _t, _x = integrate(euler_step, f_ex, 0.0, 0.0, 10.0, n_euler.value)
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.plot(_t, _x)
    _ax.set_xlabel("t")
    _ax.set_ylabel("x(t)")
    _ax.set_title(f"Euler's method, N = {n_euler.value}  (h = {10 / n_euler.value:.3g})")
    _ax.set_ylim(-1, 1)


    mo.vstack([n_euler, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8.1.2 The (second-order) Runge–Kutta method

    Euler uses the slope at the *start* of the interval. A better estimate
    uses the slope at the **midpoint** $t+\tfrac12 h$ (the "midpoint method"):

    $$
    \begin{aligned}
    k_1 &= h f(x,t)\\
    k_2 &= h f\!\left(x+\tfrac12 k_1,\; t+\tfrac12 h\right)\\
    x(t+h) &= x(t) + k_2
    \end{aligned}
    $$

    - Taylor expansions around $t+\tfrac12 h$ cancel the $h^2$ term entirely.
    - Error per step: $O(h^3)$; accuracy of the whole calculation: **$O(h^2)$**.
    - Using Euler to estimate $x(t+\tfrac12 h)$ adds an error that is itself $O(h^3)$, so nothing is lost.
    - Same accuracy as Euler with far fewer steps → faster programs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8.1.3 The fourth-order Runge–Kutta method

    Taking more Taylor expansions at more points, and the right linear combination:

    $$
    \begin{aligned}
    k_1 &= h f(x,t)\\
    k_2 &= h f\!\left(x+\tfrac12 k_1,\; t+\tfrac12 h\right)\\
    k_3 &= h f\!\left(x+\tfrac12 k_2,\; t+\tfrac12 h\right)\\
    k_4 &= h f(x+k_3,\; t+h)\\
    x(t+h) &= x(t) + \tfrac16\left(k_1 + 2k_2 + 2k_3 + k_4\right)
    \end{aligned}
    $$

    - Accurate to $h^4$ per step, error $O(h^5)$; **$O(h^4)$ overall**.
    - Often the first method physicists reach for: simple to program, excellent results.
    - **Warning:** if you get a factor wrong (a missing $\tfrac12$, a wrong sign) the output usually *still looks reasonable*, but is only about as accurate as Euler. Check the code carefully!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compare the three methods

    Same equation as Example 8.1, with a coarse step count. The "exact" curve is
    RK4 with 20 000 steps.
    """)
    return


@app.cell
def _(mo):
    n_cmp = mo.ui.slider(10, 200, value=20, step=2, label="N (steps)", show_value=True)
    show_euler = mo.ui.checkbox(value=True, label="Euler")
    show_rk2 = mo.ui.checkbox(value=True, label="RK2")
    show_rk4 = mo.ui.checkbox(value=True, label="RK4")
    return n_cmp, show_euler, show_rk2, show_rk4


@app.cell
def _(
    euler_step,
    f_ex,
    integrate,
    mo,
    n_cmp,
    plt,
    rk2_step,
    rk4_step,
    show_euler,
    show_rk2,
    show_rk4,
):
    _tr, _xr = integrate(rk4_step, f_ex, 0.0, 0.0, 10.0, 20000)
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_tr, _xr, color="lightgray", lw=4, label="reference")
    for _on, _name, _step in [
        (show_euler.value, "Euler", euler_step),
        (show_rk2.value, "RK2", rk2_step),
        (show_rk4.value, "RK4", rk4_step),
    ]:
        if _on:
            _t, _x = integrate(_step, f_ex, 0.0, 0.0, 10.0, n_cmp.value)
            _ax.plot(_t, _x, "o-", ms=3, lw=1, label=_name)
    _ax.set_xlabel("t")
    _ax.set_ylabel("x(t)")
    _ax.set_title(f"N = {n_cmp.value}")
    _ax.legend()

    mo.vstack([mo.hstack([n_cmp, show_euler, show_rk2, show_rk4], justify="start", gap=2), 
           _fig    
    ])
    return


@app.cell
def _(f_ex, integrate, mo, plt, rk4_step):
    _step_counts = (10, 50, 250)
    _t_ref, _x_ref = integrate(rk4_step, f_ex, 0.0, 0.0, 10.0, 20000)
    _x_exact = _x_ref[-1]
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_t_ref, _x_ref, color="lightgray", lw=5, label="reference (N = 20,000)")
    _error_rows = []
    for _n in _step_counts:
        _t, _x = integrate(rk4_step, f_ex, 0.0, 0.0, 10.0, _n)
        _h = 10.0 / _n
        _error = abs(_x[-1] - _x_exact)
        _ax.plot(_t, _x, "-", lw=1, label=f"N = {_n}, h = {_h:g}")
        _error_rows.append(f"| {_n} | {_h:g} | {_error:.3e} |")
    _ax.set_xlabel("t")
    _ax.set_ylabel("x(t)")
    _ax.set_title("RK4 with three step sizes")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _errors = mo.md(
        "| N (steps) | h | Absolute error at t = 10 |\n"
        "|---:|---:|---:|\n"
        + "\n".join(_error_rows)
    )

    mo.vstack([_fig, _errors])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How does the error scale with the step size?

    Error at $t=10$ versus $h$ on a log–log plot. The slope is the order of the method:
    Euler ≈ 1, RK2 ≈ 2, RK4 ≈ 4. (RK4 eventually flattens at the floor set by the reference
    solution / rounding error.)
    """)
    return


@app.cell
def _(euler_step, f_ex, integrate, np, plt, rk2_step, rk4_step):
    _, _ref = integrate(rk4_step, f_ex, 0.0, 0.0, 10.0, 20000)
    _x_exact = _ref[-1]
    _Ns = np.array([20, 40, 80, 160, 320, 640, 1280])
    _hs = 10.0 / _Ns
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _name, _step in [("Euler", euler_step), ("RK2", rk2_step), ("RK4", rk4_step)]:
        _errs = [abs(integrate(_step, f_ex, 0.0, 0.0, 10.0, int(n))[1][-1] - _x_exact) for n in _Ns]
        _ax.loglog(_hs, _errs, "o-", label=_name)
    _ax.set_xlabel("h")
    _ax.set_ylabel("|error| at t = 10")
    _ax.legend()
    _ax.grid(True, which="both", alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8.1.4 Solutions over infinite ranges

    To go all the way to $t=\infty$ we can't take infinitely many steps, so change variables:

    $$u = \frac{t}{1+t}\quad\Longleftrightarrow\quad t = \frac{u}{1-u}$$

    so $t\to\infty$ becomes $u\to 1$. By the chain rule, with $dt/du = (1-u)^{-2}$:

    $$\frac{dx}{du} = g(x,u),\qquad g(x,u) = (1-u)^{-2}\, f\!\left(x,\frac{u}{1-u}\right)$$

    Solve for $u\in[0,1)$ with the usual RK4, then map $u$ back to $t$.

    **Example 8.4:** $\dfrac{dx}{dt} = \dfrac{1}{x^2+t^2}$, $x(0)=1$ gives
    $\dfrac{dx}{du} = \dfrac{1}{x^2(1-u)^2 + u^2}$.

    Other substitutions (trigonometric, hyperbolic, ...) can work better for particular problems.
    """)
    return


@app.cell
def _(mo):
    t_max = mo.ui.slider(10, 200, value=80, step=10, label="show t up to", show_value=True)
    n_inf = mo.ui.slider(20, 400, value=100, step=10, label="N (steps in u)", show_value=True)
    return n_inf, t_max


@app.cell
def _(integrate, mo, n_inf, plt, rk4_step, t_max):
    def g_inf(x, u):
        return 1.0 / (x**2 * (1 - u) ** 2 + u**2)

    _u, _x = integrate(rk4_step, g_inf, 1.0, 0.0, 1.0, n_inf.value)
    _u, _x = _u[:-1], _x[:-1]  # drop u = 1, where t = infinity
    _t = _u / (1 - _u)
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.plot(_t, _x)
    _ax.set_xlim(0, t_max.value)
    _ax.set_xlabel("t")
    _ax.set_ylabel("x(t)")
    _ax.set_title("Solution of dx/dt = 1/(x² + t²), computed out to t = ∞")


    mo.vstack([
        mo.hstack([t_max, n_inf], justify="start", gap=2), 
        _fig
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8.3 Second-order differential equations

    First-order equations are rare in physics. Most are second-order or higher:

    $$\frac{d^2x}{dt^2} = f\!\left(x,\frac{dx}{dt},t\right)$$

    $f$ can be any function, e.g. $\dfrac{d^2x}{dt^2} = \dfrac1x\left(\dfrac{dx}{dt}\right)^2 + 2\dfrac{dx}{dt} - x^3e^{-4t}$.

    ### The trick

    Define a new variable $y$:

    $$\frac{dx}{dt} = y \qquad\text{and}\qquad \frac{dy}{dt} = f(x,y,t)$$

    Two **first-order** equations, equivalent to the one second-order equation.
    Now use the simultaneous-equation methods with a vector $\mathbf r = (x, y)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generalizing

    **Third order:** $\dfrac{d^3x}{dt^3} = f\!\left(x,\dfrac{dx}{dt},\dfrac{d^2x}{dt^2},t\right)$
    becomes three first-order equations:

    $$\frac{dx}{dt}=y,\qquad \frac{dy}{dt}=z,\qquad \frac{dz}{dt}=f(x,y,z,t)$$

    **Vector form (any number of dependent variables):**

    $$\frac{d^2\mathbf r}{dt^2} = \mathbf f\!\left(\mathbf r,\frac{d\mathbf r}{dt},t\right)
    \;\Longleftrightarrow\;
    \frac{d\mathbf r}{dt}=\mathbf s,\quad \frac{d\mathbf s}{dt}=\mathbf f(\mathbf r,\mathbf s,t)$$

    In general, $n$ equations of order $m$ become $m\times n$ simultaneous first-order equations,
    solvable with the standard methods (the vector RK4 is just Eq. 8.33 with arrays).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Example 8.6 - The nonlinear pendulum

    An arm of length $\ell$, bob of mass $m$, angle $\theta$ from the vertical.
    The tangential component of gravity is $mg\sin\theta$, so Newton's second law gives

    $$m\ell\frac{d^2\theta}{dt^2} = -mg\sin\theta
    \quad\Longrightarrow\quad
    \frac{d^2\theta}{dt^2} = -\frac{g}{\ell}\sin\theta$$

    Nonlinear → no exact solution. Define $\omega = d\theta/dt$:

    $$\frac{d\theta}{dt} = \omega,\qquad \frac{d\omega}{dt} = -\frac{g}{\ell}\sin\theta$$

    Combine into $\mathbf r=(\theta,\omega)$ and apply vector RK4. We only care about $\theta$ - we just ignore $\omega$ when plotting.
    """)
    return


@app.cell
def _(np):
    g_grav = 9.81

    def make_pendulum(length):
        def f_pend(r, t):
            theta, omega = r
            return np.array([omega, -(g_grav / length) * np.sin(theta)], float)
        return f_pend

    return g_grav, make_pendulum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it: release the pendulum from rest

    Set the initial angle. For small angles the pendulum is nearly harmonic
    (dashed line: the linear approximation $\theta_0\cos\sqrt{g/\ell}\,t$);
    near 180° the motion is strongly non-sinusoidal and slower.
    """)
    return


@app.cell
def _(mo):
    theta0_deg = mo.ui.slider(1, 179, value=179, step=1, label="θ₀ (degrees)", show_value=True)
    length_cm = mo.ui.slider(5, 100, value=10, step=5, label="ℓ (cm)", show_value=True)
    tmax_pend = mo.ui.slider(1, 20, value=10, step=1, label="t max (s)", show_value=True)
    show_linear = mo.ui.checkbox(value=True, label="show linear approximation")
    return length_cm, show_linear, theta0_deg, tmax_pend


@app.cell
def _(
    g_grav,
    integrate,
    length_cm,
    make_pendulum,
    mo,
    np,
    plt,
    rk4_step,
    show_linear,
    theta0_deg,
    tmax_pend,
):
    _l = length_cm.value / 100.0
    _th0 = np.radians(theta0_deg.value)
    _N = int(tmax_pend.value / 0.002)
    t_pend, r_pend = integrate(rk4_step, make_pendulum(_l), [_th0, 0.0], 0.0, float(tmax_pend.value), _N)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    _a1.plot(t_pend, r_pend[:, 0], label="nonlinear (RK4)")
    if show_linear.value:
        _a1.plot(t_pend, _th0 * np.cos(np.sqrt(g_grav / _l) * t_pend), "--", label="small-angle approx.")
    _a1.set_xlabel("t (s)")
    _a1.set_ylabel("θ (rad)")
    _a1.legend(loc="upper right")
    _a2.plot(r_pend[:, 0], r_pend[:, 1])
    _a2.set_xlabel("θ")
    _a2.set_ylabel("ω")
    _a2.set_title("phase space")
    _fig.tight_layout()

    mo.vstack([
        mo.hstack([theta0_deg, length_cm, tmax_pend, show_linear], justify="start", gap=2, wrap=True), 
        _fig
    ])
    return r_pend, t_pend


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Snapshot of the pendulum at a chosen time
    """)
    return


@app.cell
def _(mo, tmax_pend):
    t_snap = mo.ui.slider(0.0, float(tmax_pend.value), value=0.0, step=0.01, label="t (s)", show_value=True)
    return (t_snap,)


@app.cell
def _(length_cm, mo, np, plt, r_pend, t_pend, t_snap):
    _l = length_cm.value / 100.0
    _i = int(np.searchsorted(t_pend, t_snap.value))
    _i = min(_i, len(t_pend) - 1)
    _th = r_pend[_i, 0]
    _bx, _by = _l * np.sin(_th), -_l * np.cos(_th)
    _fig, _ax = plt.subplots(figsize=(3.6, 3.6))
    _ax.plot([0, _bx], [0, _by], "k-", lw=2)
    _ax.plot([0], [0], "ko")
    _ax.plot([_bx], [_by], "o", ms=18, color="tab:red")
    _ax.set_xlim(-1.2 * _l, 1.2 * _l)
    _ax.set_ylim(-1.2 * _l, 1.2 * _l)
    _ax.set_aspect("equal")
    _ax.set_title(f"θ = {np.degrees(_th):.1f}°")
    _ax.axis("off")


    mo.vstack([t_snap, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    - **Euler's method** is the simplest, but only first-order accurate. It is rarely used, since RK is almost as easy and far better.
    - **Runge–Kutta** methods use slopes at intermediate points: RK2 (midpoint) is $O(h^2)$, **RK4 is $O(h^4)$** and is the workhorse.
    - A good RK4 with 20 points beats Euler with hundreds, but check every factor in your code.
    - **Infinite range:** substitute $u=t/(1+t)$ and solve for $u\in[0,1)$.
    - **Second-order (and higher) equations:** introduce $y=dx/dt$ (etc.) to get a system of first-order equations, then use vector RK4.
    - Example: the **nonlinear pendulum**, $\ddot\theta=-(g/\ell)\sin\theta$, solved as $\mathbf r=(\theta,\omega)$.
    """)
    return


if __name__ == "__main__":
    app.run()
