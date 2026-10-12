import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    return mo, np, plt, solve_ivp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Section 8.5 — Other methods for differential equations

    Presentation for Modeling of Materials, chapter 8 of Newman's
    *Computational Physics*.

    The fourth-order Runge–Kutta method is a robust workhorse, but it is not the
    only way to solve an ordinary differential equation. This section looks at
    methods that do something Runge–Kutta cannot, and builds up to one of the most
    accurate ODE solvers there is:

    | Subsection | Method | Main idea |
    |---|---|---|
    | 8.5.1 | Leapfrog | a second-order method whose midpoints "leap over" each other |
    | 8.5.2 | Time-reversal symmetry | why leapfrog conserves energy over long runs |
    | 8.5.3 | Verlet | leapfrog specialised to equations of motion, with energy at integer steps |
    | 8.5.4 | Modified midpoint | leapfrog with a cleaned-up last step, whose error has only even powers of $h$ |
    | 8.5.5 | Bulirsch–Stoer | modified midpoint plus Richardson extrapolation |
    | 8.5.6 | Interval size | choosing the big step $H$ automatically |

    Two properties of leapfrog drive the whole section: it is **time-reversal
    symmetric** (useful when energy conservation matters), and its error is an **even
    function of the step size** (the ingredient that makes extrapolation so powerful).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.5.1 The leapfrog method

    ---

    Take a first-order equation in one variable,

    $$
    \frac{dx}{dt} = f(x,t). \qquad (8.58)
    $$

    The second-order Runge–Kutta method (Section 8.1.2) estimates the midpoint value
    with an Euler half step, then uses the slope at that midpoint for the full step:

    $$
    \begin{aligned}
    x(t+\tfrac{1}{2}h) &= x(t) + \tfrac{1}{2}h\,f(x,t), \\
    x(t+h) &= x(t) + h\,f\big(x(t+\tfrac{1}{2}h),\,t+\tfrac{1}{2}h\big).
    \end{aligned}
    \qquad (8.59)
    $$

    Each step is accurate to order $h^2$ with an $h^3$ error, and the accumulated error
    after many steps is $\mathcal{O}(h^2)$.

    **Leapfrog** starts the same way, but afterwards the midpoint is *not* recomputed
    from the full-step value. Instead each midpoint is obtained from the previous
    midpoint, using the slope at the full step in between:

    $$
    \begin{aligned}
    x(t+h) &= x(t) + h\,f\big(x(t+\tfrac{1}{2}h),\,t+\tfrac{1}{2}h\big), \\
    x(t+\tfrac{3}{2}h) &= x(t+\tfrac{1}{2}h) + h\,f\big(x(t+h),\,t+h\big).
    \end{aligned}
    \qquad (8.62)
    $$

    Applying these two lines repeatedly gives the full solution. The positions
    $x(t+kh)$ and the midpoints $x(t+(k+\tfrac12)h)$ are two interleaved sequences, and
    each one uses the other to take its step, so each "leaps over" the other. Exactly
    as with Runge–Kutta, the equations carry over to vectors,
    $\mathbf r(t+h)=\mathbf r(t)+h\,\mathbf f(\mathbf r(t+\tfrac12h),t+\tfrac12h)$ and so on,
    so higher-order equations are handled by first rewriting them as first-order systems
    (Section 8.3). The sketch below compares the two stencils (cf. Fig. 8.9).
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    _fig, _axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    _K = 4
    _full = np.arange(_K + 1, dtype=float)
    _mid = np.arange(_K) + 0.5

    def _arc(ax, x1, x2, rad, color):
        ax.annotate(
            "",
            xy=(x2, 0),
            xytext=(x1, 0),
            arrowprops=dict(arrowstyle="-|>", connectionstyle=f"arc3,rad={rad}", color=color, lw=1.6),
            zorder=2,
        )

    _blue, _orange = "#1d4ed8", "#c2410c"
    for _ax, _kind in zip(_axes, ["rk2", "leap"]):
        _ax.scatter(_full, np.zeros_like(_full), s=90, color="black", zorder=4)
        _ax.scatter(_mid, np.zeros_like(_mid), s=90, facecolor="white", edgecolor="black", zorder=4)
        for _k in range(_K):
            if _kind == "rk2":
                _arc(_ax, _k, _k + 0.5, -0.9, _blue)
                _arc(_ax, _k, _k + 1, 0.55, _orange)
            else:
                if _k == 0:
                    _arc(_ax, 0, 0.5, -0.9, _blue)
                else:
                    _arc(_ax, _k - 0.5, _k + 0.5, -0.55, _blue)
                _arc(_ax, _k, _k + 1, 0.55, _orange)
        _ax.set_ylim(-0.75, 0.85)
        _ax.set_yticks([])
        for _side in ("left", "right", "top"):
            _ax.spines[_side].set_visible(False)
    _axes[0].set_title("(a) second-order Runge-Kutta: every step restarts from a filled point", fontsize=11, loc="left")
    _axes[1].set_title("(b) leapfrog: after the first half step the midpoints chain onto each other", fontsize=11, loc="left")
    _axes[1].set_xticks(np.arange(_K + 1))
    _axes[1].set_xticklabels(["t", "t+h", "t+2h", "t+3h", "t+4h"])
    _axes[1].text(0.0, -0.65, "filled = x at whole steps,  open = x at half steps,  orange arc = step taken with the slope at the point it passes over",
                  fontsize=9, color="0.3")
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Leapfrog is accurate to the same order as second-order Runge–Kutta ($h^3$ per step,
    $h^2$ overall), and fourth-order Runge–Kutta is substantially more accurate for
    little extra effort. So why bother? Because of the two properties mentioned above,
    which the next subsections develop.

    The cells below set up the test problems and implement the methods used throughout
    the notebook. The nonlinear pendulum is the running example,

    $$
    \frac{d\theta}{dt}=\omega, \qquad \frac{d\omega}{dt}=-\frac{g}{\ell}\sin\theta, \qquad (8.66)
    $$

    with $\ell = 10\,$cm. Its total energy per unit mass is
    $E=\tfrac12\ell^2\omega^2-g\ell\cos\theta$.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    g_pend = 9.81  # m s^-2
    l_pend = 0.1  # m

    def pendulum_rhs(r, t=0.0):
        """Right-hand side for r = (theta, omega)."""
        return np.array([r[1], -(g_pend / l_pend) * np.sin(r[0])], dtype=float)

    def pendulum_energy(states):
        """Total energy per unit mass for an array of (theta, omega) rows."""
        theta, omega = states[:, 0], states[:, 1]
        return 0.5 * l_pend**2 * omega**2 - g_pend * l_pend * np.cos(theta)

    return g_pend, l_pend, pendulum_energy, pendulum_rhs


@app.cell(hide_code=True)
def _(np):
    def rk2_solve(f, r0, t0, t1, N):
        """Second-order Runge-Kutta, Eq. (8.59), N steps from t0 to t1."""
        h = (t1 - t0) / N
        r = np.array(r0, dtype=float)
        out = np.empty((N + 1, r.size))
        out[0] = r
        for k in range(N):
            t = t0 + k * h
            k1 = h * f(r, t)
            k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
            r = r + k2
            out[k + 1] = r
        return out

    def rk4_solve(f, r0, t0, t1, N):
        """Fourth-order Runge-Kutta, N steps from t0 to t1."""
        h = (t1 - t0) / N
        r = np.array(r0, dtype=float)
        out = np.empty((N + 1, r.size))
        out[0] = r
        for k in range(N):
            t = t0 + k * h
            k1 = h * f(r, t)
            k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
            k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
            k4 = h * f(r + k3, t + h)
            r = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            out[k + 1] = r
        return out

    def leapfrog_solve(f, r0, t0, t1, N):
        """Leapfrog, Eq. (8.62) applied to the vector r.

        Returns r at t0 + k h and r at t0 + (k + 1/2) h, both for k = 0..N."""
        h = (t1 - t0) / N
        r = np.array(r0, dtype=float)
        r_half = r + 0.5 * h * f(r, t0)  # Euler half step to get started
        out = np.empty((N + 1, r.size))
        out_half = np.empty_like(out)
        out[0], out_half[0] = r, r_half
        for k in range(N):
            t = t0 + k * h
            r = r + h * f(r_half, t + 0.5 * h)
            r_half = r_half + h * f(r, t + h)
            out[k + 1], out_half[k + 1] = r, r_half
        return out, out_half

    def leapfrog_backward(f, r_end, r_half_end, t_end, h, N):
        """Run the leapfrog algorithm with step -h, Eqs. (8.64)-(8.65).

        (r_end, r_half_end) = (x(t_end), x(t_end + h/2)). Returns the whole path and
        the half-step values, ordered from t_end backwards."""
        r = np.array(r_end, dtype=float)
        r_half = np.array(r_half_end, dtype=float)
        path = np.empty((N + 1, r.size))
        path_half = np.empty_like(path)
        path[0], path_half[0] = r, r_half
        for k in range(N):
            t = t_end - k * h
            r_half = r_half - h * f(r, t)
            r = r - h * f(r_half, t - 0.5 * h)
            path[k + 1], path_half[k + 1] = r, r_half
        return path, path_half

    def verlet_solve(acc, x0, v0, t0, t1, N):
        """Verlet method for x'' = acc(x, t), Eqs. (8.74)-(8.75).

        Returns x and v at the whole steps t0 + k h (v via Eq. 8.73)."""
        h = (t1 - t0) / N
        x = np.array(x0, dtype=float)
        v_half = np.array(v0, dtype=float) + 0.5 * h * acc(x, t0)
        xs = np.empty((N + 1, x.size))
        vs = np.empty_like(xs)
        xs[0], vs[0] = x, v0
        for k in range(N):
            t = t0 + k * h
            x = x + h * v_half
            kk = h * acc(x, t + h)
            vs[k + 1] = v_half + 0.5 * kk
            v_half = v_half + kk
            xs[k + 1] = x
        return xs, vs

    return (
        leapfrog_backward,
        leapfrog_solve,
        rk2_solve,
        rk4_solve,
        verlet_solve,
    )


@app.cell(hide_code=True)
def _(leapfrog_solve, mo, np, plt, rk2_solve):
    # Convergence test on x'' = -x, i.e. r = (x, v), exact solution x = cos t.
    def _osc(r, t=0.0):
        return np.array([r[1], -r[0]])

    _hs = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])
    _err_rk2, _err_lf = [], []
    for _h in _hs:
        _N = int(round(10 / _h))
        _a = rk2_solve(_osc, [1.0, 0.0], 0, 10, _N)[-1, 0]
        _b = leapfrog_solve(_osc, [1.0, 0.0], 0, 10, _N)[0][-1, 0]
        _err_rk2.append(abs(_a - np.cos(10)))
        _err_lf.append(abs(_b - np.cos(10)))

    _slope_rk2 = np.polyfit(np.log(_hs), np.log(_err_rk2), 1)[0]
    _slope_lf = np.polyfit(np.log(_hs), np.log(_err_lf), 1)[0]

    _fig, _ax = plt.subplots(figsize=(6.5, 4.2))
    _ax.loglog(_hs, _err_rk2, "o-", label="second-order Runge-Kutta")
    _ax.loglog(_hs, _err_lf, "s-", label="leapfrog")
    _ax.loglog(_hs, 0.5 * _hs**2 * (_err_lf[0] / (0.5 * _hs[0] ** 2)), "k--", lw=1, label="slope 2")
    _ax.set(xlabel="step size h", ylabel="error in x(10)", title="Both methods are second order")
    _ax.legend()
    _ax.grid(True, which="both", alpha=0.3)

    mo.vstack(
        [
            mo.md(
                "**Check of the order.** For $\\ddot x=-x$ with $x(0)=1$, the error at $t=10$ "
                "falls by a factor of four every time $h$ is halved for both methods. The fitted "
                f"slopes on the log-log plot are **{_slope_rk2:.2f}** (Runge-Kutta) and "
                f"**{_slope_lf:.2f}** (leapfrog), i.e. both are second order."
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.5.2 Time-reversal symmetry and energy conservation

    ---

    The state of a leapfrog calculation at any time $t_1$ is specified by the *two*
    values $x(t_1)$ and $x(t_1+\tfrac12h)$. Suppose we continue to a later time $t_2$
    and then run the same algorithm backwards, i.e. with step $-h$. Setting
    $h\to-h$ in Eq. (8.62) gives

    $$
    \begin{aligned}
    x(t-h) &= x(t) - h\,f\big(x(t-\tfrac12h),\,t-\tfrac12h\big), \\
    x(t-\tfrac32h) &= x(t-\tfrac12h) - h\,f\big(x(t-h),\,t-h\big),
    \end{aligned}
    \qquad (8.64)
    $$

    and shifting $t\to t+\tfrac32h$:

    $$
    \begin{aligned}
    x(t+\tfrac12h) &= x(t+\tfrac32h) - h\,f\big(x(t+h),\,t+h\big), \\
    x(t) &= x(t+h) - h\,f\big(x(t+\tfrac12h),\,t+\tfrac12h\big).
    \end{aligned}
    \qquad (8.65)
    $$

    These are *exactly* the forward equations with the sign of the $hf$ term flipped:
    every operation is undone. So the backward run retraces the forward steps and
    recovers the starting values (up to rounding error). Second-order Runge–Kutta does
    not have this property: putting $h\to-h$ in (8.59) does not undo the forward step.

    Time reversal is not just a curiosity. Consider the frictionless pendulum over one
    full swing, starting at the far end of its motion. Physically the swing is itself
    time-reversal symmetric, so the backward solution is the *same* motion as the
    forward one. Suppose the energy had drifted up during the forward swing. Then the
    backward run, which passes through exactly the same values of $\theta$ and
    $\omega$ in reverse order, would see it drift *down* (the backward run retraces
    the forward one) and also *up* (it is the same motion, so it behaves the same as the
    forward one). That is a contradiction, and the same argument rules out a drift
    down. The only possibility is that **the energy at the end of a full swing equals
    the energy at the start**.

    Two caveats: the argument does not make the solution exact (only
    second-order accurate), and it only applies over *whole periods*; within a swing
    the energy may fluctuate.

    **Demonstration 1: running the algorithm backwards.** Pick a number of steps,
    a step size and a starting angle. The pendulum is integrated forward with each
    method, and then backward from the final state with step $-h$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rev_steps = mo.ui.slider(100, 1000, value=500, step=100, label="Number of steps")
    rev_h = mo.ui.slider(0.002, 0.01, value=0.006, step=0.002, label="Step size h (s)")
    rev_theta = mo.ui.slider(30, 179, value=120, step=1, label="Starting angle (degrees)")
    return rev_h, rev_steps, rev_theta


@app.cell(hide_code=True)
def _(
    leapfrog_backward,
    leapfrog_solve,
    mo,
    np,
    pendulum_rhs,
    plt,
    rev_h,
    rev_steps,
    rev_theta,
    rk2_solve,
):
    _h = float(rev_h.value)
    _N = int(rev_steps.value)
    _r0 = np.array([np.radians(rev_theta.value), 0.0])
    _T = _N * _h
    _t = np.linspace(0, _T, _N + 1)

    _lf, _lf_half = leapfrog_solve(pendulum_rhs, _r0, 0, _T, _N)
    _lf_back, _lf_back_half = leapfrog_backward(pendulum_rhs, _lf[-1], _lf_half[-1], _T, _h, _N)
    _lf_err = np.max(np.abs(_lf_back[-1] - _r0))

    _rk = rk2_solve(pendulum_rhs, _r0, 0, _T, _N)
    _rk_back = rk2_solve(pendulum_rhs, _rk[-1], _T, 0, _N)
    _rk_err = np.max(np.abs(_rk_back[-1] - _r0))

    _lf_dev = np.abs(_lf_back[::-1, 0] - _lf[:, 0]) + 1e-18
    _rk_dev = np.abs(_rk_back[::-1, 0] - _rk[:, 0]) + 1e-18

    _fig, _axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    _axes[0, 0].plot(_t, _lf[:, 0], "k-", lw=2, label="forward")
    _axes[0, 0].plot(_T - _t, _lf_back[:, 0], "o", ms=3, color="#1d4ed8", markevery=max(1, _N // 40), label="backward (step -h)")
    _axes[0, 0].set_title("Leapfrog: backward run retraces the forward one")
    _axes[0, 1].plot(_t, _rk[:, 0], "k-", lw=2, label="forward")
    _axes[0, 1].plot(_T - _t, _rk_back[:, 0], "o", ms=3, color="#c2410c", markevery=max(1, _N // 40), label="backward (step -h)")
    _axes[0, 1].set_title("Second-order Runge-Kutta: it does not")
    _axes[1, 0].semilogy(_t, _lf_dev, color="#1d4ed8")
    _axes[1, 1].semilogy(_t, _rk_dev, color="#c2410c")
    for _ax in _axes[1]:
        _ax.set_ylim(1e-17, 1e1)
        _ax.set_xlabel("t (s)")
    _axes[1, 0].set_ylabel("|backward - forward| in theta")
    _axes[0, 0].set_ylabel("theta (rad)")
    for _ax in _axes.ravel():
        _ax.grid(alpha=0.3)
    _axes[0, 0].legend()
    _axes[0, 1].legend()
    _fig.tight_layout()

    mo.vstack(
        [
            rev_steps,
            rev_h,
            rev_theta,
            mo.hstack(
                [
                    mo.stat(label="Leapfrog: error in recovered start", value=f"{_lf_err:.1e}"),
                    mo.stat(label="Runge-Kutta: error in recovered start", value=f"{_rk_err:.1e}"),
                ],
                justify="center",
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The leapfrog backward run recovers the initial $(\theta,\omega)$ to rounding error,
    no matter how coarse the step. Runge–Kutta returns to somewhere near the start,
    with an error of the size of the method's own truncation error, which grows
    quickly with $h$. The upper panels can look alike because a small error is hard
    to see on a plot of $\theta$ itself; the lower panels plot the deviation between the
    backward and forward solutions on a log scale, which makes the difference clear.

    **Demonstration 2: energy of the nonlinear pendulum (cf. Fig. 8.10).** We
    use the pendulum released at $179^\circ$ and the *second-order* Runge–Kutta
    method (deliberately, to make the drift easy to see). The top panel shows the
    change in total energy for Runge–Kutta, the bottom panel for leapfrog. Note that
    the vertical scales differ.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    fig810_theta = mo.ui.slider(30, 179, value=179, step=1, label="Starting angle (degrees)")
    fig810_h = mo.ui.slider(0.002, 0.02, value=0.01, step=0.002, label="Step size h (s)")
    fig810_T = mo.ui.slider(5, 20, value=10, step=1, label="Duration (s)")
    return fig810_T, fig810_h, fig810_theta


@app.cell(hide_code=True)
def _(
    fig810_T,
    fig810_h,
    fig810_theta,
    leapfrog_solve,
    mo,
    np,
    pendulum_energy,
    pendulum_rhs,
    plt,
    rk2_solve,
):
    _h = float(fig810_h.value)
    _T = float(fig810_T.value)
    _N = int(round(_T / _h))
    _r0 = np.array([np.radians(fig810_theta.value), 0.0])
    _t = np.linspace(0, _T, _N + 1)

    _rk = rk2_solve(pendulum_rhs, _r0, 0, _T, _N)
    _lf = leapfrog_solve(pendulum_rhs, _r0, 0, _T, _N)[0]
    _dE_rk = pendulum_energy(_rk) - pendulum_energy(_rk)[0]
    _dE_lf = pendulum_energy(_lf) - pendulum_energy(_lf)[0]

    _fig, _axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    _axes[0].plot(_t, _dE_rk, color="0.15")
    _axes[0].set_ylabel("2nd-order RK\nchange in energy")
    _axes[1].plot(_t, _dE_lf, color="0.15")
    _axes[1].set_ylabel("Leapfrog\nchange in energy")
    _axes[1].set_xlabel("Time (s)")
    for _ax in _axes:
        _ax.grid(alpha=0.3)
    _fig.tight_layout()

    mo.vstack(
        [
            fig810_theta,
            fig810_h,
            fig810_T,
            mo.hstack(
                [
                    mo.stat(label="Runge-Kutta: energy drift over the run", value=f"{_dE_rk[-1]:.2e}"),
                    mo.stat(label="Leapfrog: peak-to-peak oscillation", value=f"{np.ptp(_dE_lf):.2e}"),
                ],
                justify="center",
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With these settings ($179^\circ$, $h=0.01$, 10 s) the Runge–Kutta energy
    creeps steadily upward, while the leapfrog energy oscillates with a much larger
    amplitude (the vertical scale of the lower panel is five times larger) but keeps
    coming back to the same value. Leapfrog is *less* accurate in the short term, and
    better in the long term.

    The long-time behaviour is tested properly in Section 8.5.3, once the Verlet
    method makes it easy to compute the energy at whole steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.5.3 The Verlet method

    ---

    Equations of motion from Newton's second law have the form
    $\ddot x = f(x,t)$, i.e. as a first-order pair

    $$
    \frac{dx}{dt}=v, \qquad \frac{dv}{dt}=f(x,t). \qquad (8.68)
    $$

    The right-hand side of the first equation depends only on $v$, and that of the
    second only on $x$. Applying leapfrog to the vector $(x,v)$ would carry both
    variables at all whole and half steps, which is wasteful: each equation only ever
    needs one of them at each time. Writing leapfrog out explicitly, given $x(t)$ and
    $v(t+\tfrac12h)$:

    $$
    x(t+h)=x(t)+h\,v(t+\tfrac12h), \qquad
    v(t+\tfrac32h)=v(t+\tfrac12h)+h\,f\big(x(t+h),t+h\big). \qquad (8.70\text{–}8.71)
    $$

    Here $x$ is only ever known at whole steps, and $v$ only at half steps, with half
    the work of the vector version.

    The price is that quantities depending on both, such as the total energy, cannot
    be evaluated directly, since $x$ and $v$ are never known at the same time. The fix
    is an extra half step with Euler's method:

    $$
    v(t+h)=v(t+\tfrac12h)+\tfrac12h\,f\big(x(t+h),t+h\big). \qquad (8.73)
    $$

    To get started we need the first half step, $v(t+\tfrac12h)=v(t)+\tfrac12h\,f(x(t),t)$
    (8.74). The complete **Verlet method** in $d$ dimensions, for $\ddot{\mathbf r}=\mathbf f(\mathbf r,t)$,
    is then

    $$
    \begin{aligned}
    \mathbf v(t+\tfrac12h) &= \mathbf v(t)+\tfrac12h\,\mathbf f(\mathbf r(t),t) && \text{(first step only)},\\[2pt]
    \mathbf r(t+h) &= \mathbf r(t)+h\,\mathbf v(t+\tfrac12h), \\
    \mathbf k &= h\,\mathbf f(\mathbf r(t+h),t+h), \\
    \mathbf v(t+h) &= \mathbf v(t+\tfrac12h)+\tfrac12\mathbf k, \\
    \mathbf v(t+\tfrac32h) &= \mathbf v(t+\tfrac12h)+\mathbf k.
    \end{aligned}
    \qquad (8.78)
    $$

    This is implemented in `verlet_solve` above: it returns $\mathbf r$ and $\mathbf v$
    at the whole steps, and the half-step velocity is carried along only inside the
    loop.

    ### Exercise 8.11 — leapfrog on a nonlinear equation

    Solve

    $$
    \frac{d^2x}{dt^2}-\left(\frac{dx}{dt}\right)^2+x+5=0
    $$

    with $x(0)=1$, $\dot x(0)=0$, from $t=0$ to $50$ in steps of $h=0.001$.
    Rewriting it as $\dot x=v$, $\dot v = v^2 - x - 5$ shows that the right-hand
    side of the second equation depends on $v$ as well as $x$. The Verlet shortcut
    above needs $\dot v=f(x,t)$ only, so here the full vector leapfrog method of
    Section 8.5.1 is used, as the exercise asks.
    """)
    return


@app.cell(hide_code=True)
def _(leapfrog_solve, mo, np, plt):
    def _ex811_rhs(r, t=0.0):
        x, v = r
        return np.array([v, v**2 - x - 5.0])

    _h = 0.001
    _N = int(round(50 / _h))
    _sol = leapfrog_solve(_ex811_rhs, [1.0, 0.0], 0, 50, _N)[0]
    _t = np.linspace(0, 50, _N + 1)

    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    _ax.plot(_t, _sol[:, 0], lw=1.2)
    _ax.set(xlabel="t", ylabel="x(t)", title="Exercise 8.11: leapfrog solution, h = 0.001")
    _ax.grid(alpha=0.3)
    mo.vstack(
        [
            mo.md(
                f"The solution oscillates between $x\\approx{_sol[:, 0].min():.2f}$ and "
                f"$x={_sol[:, 0].max():.2f}$ (the starting value) and stays bounded over the "
                "whole run."
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 8.12 — the orbit of the Earth

    The equations of motion of a body orbiting the Sun are
    $\ddot{\mathbf r}=-GM\,\mathbf r/r^3$, with $G=6.6738\times10^{-11}$ m$^3$ kg$^{-1}$ s$^{-2}$
    and $M=1.9891\times10^{30}$ kg. They have the Verlet form $\ddot{\mathbf r}=\mathbf f(\mathbf r)$.
    The Earth starts at perihelion, at distance $1.4710\times10^{11}$ m, moving
    tangentially at $3.0287\times10^{4}$ m s$^{-1}$. Its mass is $m=5.9722\times10^{24}$ kg.

    **(a)** the orbit, with a time step of one hour; **(b)** kinetic, potential and total
    energy against time; **(c)** the total energy alone, to see that it returns to its
    starting value at the end of every orbit.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    G_const = 6.6738e-11  # m^3 kg^-1 s^-2
    M_sun = 1.9891e30  # kg
    m_earth = 5.9722e24  # kg
    year_s = 365.25 * 24 * 3600.0  # s

    def gravity_acc(r, t=0.0):
        """Acceleration for the orbit equation d2r/dt2 = -G M r / |r|^3."""
        return -G_const * M_sun * r / np.linalg.norm(r) ** 3

    def orbit_rhs(s, t=0.0):
        """First-order form, s = (x, y, vx, vy)."""
        return np.concatenate([s[2:], gravity_acc(s[:2])])

    earth_r0 = np.array([1.4710e11, 0.0])
    earth_v0 = np.array([0.0, 3.0287e4])
    return (
        G_const,
        M_sun,
        earth_r0,
        earth_v0,
        gravity_acc,
        m_earth,
        orbit_rhs,
        year_s,
    )


@app.cell(hide_code=True)
def _(mo):
    orbit_years = mo.ui.slider(1, 5, value=3, step=1, label="Years simulated")
    orbit_h_hours = mo.ui.slider(1, 24, value=1, step=1, label="Time step h (hours)")
    return orbit_h_hours, orbit_years


@app.cell(hide_code=True)
def _(
    G_const,
    M_sun,
    earth_r0,
    earth_v0,
    gravity_acc,
    m_earth,
    mo,
    np,
    orbit_h_hours,
    orbit_years,
    plt,
    verlet_solve,
):
    _h = orbit_h_hours.value * 3600.0
    _T = orbit_years.value * 365.25 * 24 * 3600.0
    _N = int(round(_T / _h))
    _xs, _vs = verlet_solve(gravity_acc, earth_r0, earth_v0, 0.0, _N * _h, _N)
    _r = np.linalg.norm(_xs, axis=1)
    _KE = 0.5 * m_earth * np.sum(_vs**2, axis=1)
    _PE = -G_const * M_sun * m_earth / _r
    _E = _KE + _PE
    _t_days = np.arange(_N + 1) * _h / 86400.0
    _dE = (_E - _E[0]) / abs(_E[0])
    _peri = np.where((_r[1:-1] < _r[:-2]) & (_r[1:-1] < _r[2:]))[0] + 1

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.3))
    _axes[0].plot(_xs[:, 0] / 1e11, _xs[:, 1] / 1e11, lw=1)
    _axes[0].plot([0], [0], "o", color="gold", mec="k", ms=10)
    _axes[0].set(xlabel="x (10$^{11}$ m)", ylabel="y (10$^{11}$ m)", title="(a) Orbit (Verlet)")
    _axes[0].set_aspect("equal")
    _axes[1].plot(_t_days, _KE / 1e33, label="kinetic")
    _axes[1].plot(_t_days, _PE / 1e33, label="potential")
    _axes[1].plot(_t_days, _E / 1e33, "k", label="total")
    _axes[1].set(xlabel="t (days)", ylabel="energy (10$^{33}$ J)", title="(b) Energies")
    _axes[1].legend()
    _axes[2].plot(_t_days, _dE, lw=1)
    _axes[2].plot(_t_days[_peri], _dE[_peri], "o", color="crimson", ms=6, label="perihelion passages")
    _axes[2].set(xlabel="t (days)", ylabel="(E - E$_0$) / |E$_0$|", title="(c) Total energy alone")
    _axes[2].legend()
    for _ax in _axes[1:]:
        _ax.grid(alpha=0.3)
    _fig.tight_layout()

    _peri_txt = ", ".join(f"{_dE[_i]:.1e}" for _i in _peri[:6]) if len(_peri) else "no full orbit yet"
    mo.vstack(
        [
            orbit_years,
            orbit_h_hours,
            mo.hstack(
                [
                    mo.stat(label="Steps taken", value=f"{_N:,}"),
                    mo.stat(label="Largest energy deviation (relative)", value=f"{np.max(np.abs(_dE)):.1e}"),
                    mo.stat(label="Closest / farthest (10^11 m)", value=f"{_r.min() / 1e11:.3f} / {_r.max() / 1e11:.3f}"),
                ],
                justify="center",
            ),
            mo.center(_fig),
            mo.md(
                f"Relative energy deviation at the perihelion passages: {_peri_txt}. "
                "Within an orbit the energy varies (panel c), but at each return to the same "
                "point of the orbit it is back to the starting value, up to a tiny residual "
                "from floating-point rounding."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In panel (b) the kinetic and potential energies vary a lot over the orbit (the
    Earth moves faster near perihelion), but their sum is constant on that scale; the
    tiny remaining variation shows up only in panel (c). Because the orbit is close to
    circular, the orbit in panel (a) is only very slightly non-circular, as the
    exercise says.

    ### Why this matters: long runs

    Return to the energy question for much longer simulations. Verlet's energy error
    stays bounded however long the calculation runs, whereas methods that do not have
    time-reversal symmetry slowly drift. First the Earth's orbit with a coarse
    time step, comparing Verlet to fourth-order Runge–Kutta:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    long_h_days = mo.ui.slider(5, 20, value=10, step=5, label="Time step h (days)")
    long_years = mo.ui.slider(100, 500, value=300, step=100, label="Years simulated")
    return long_h_days, long_years


@app.cell(hide_code=True)
def _(
    G_const,
    M_sun,
    earth_r0,
    earth_v0,
    gravity_acc,
    long_h_days,
    long_years,
    m_earth,
    mo,
    np,
    orbit_rhs,
    plt,
    rk4_solve,
    verlet_solve,
    year_s,
):
    _h = long_h_days.value * 86400.0
    _N = int(round(long_years.value * year_s / _h))
    _xs, _vs = verlet_solve(gravity_acc, earth_r0, earth_v0, 0.0, _N * _h, _N)
    _rk = rk4_solve(orbit_rhs, np.concatenate([earth_r0, earth_v0]), 0.0, _N * _h, _N)

    def _energy(pos, vel):
        return 0.5 * m_earth * np.sum(vel**2, axis=1) - G_const * M_sun * m_earth / np.linalg.norm(pos, axis=1)

    _Ev = _energy(_xs, _vs)
    _Er = _energy(_rk[:, :2], _rk[:, 2:])
    _t_years = np.arange(_N + 1) * _h / year_s

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.plot(_t_years, (_Er - _Er[0]) / abs(_Er[0]), label="4th-order Runge-Kutta")
    _ax.plot(_t_years, (_Ev - _Ev[0]) / abs(_Ev[0]), label="Verlet")
    _ax.set(xlabel="t (years)", ylabel="(E - E$_0$) / |E$_0$|", title="Energy error of the Earth's orbit, long run")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig.tight_layout()

    mo.vstack(
        [
            long_h_days,
            long_years,
            mo.hstack(
                [
                    mo.stat(label="Runge-Kutta: energy error at the end", value=f"{(_Er[-1] - _Er[0]) / abs(_Er[0]):.1e}"),
                    mo.stat(label="Verlet: largest energy error anywhere", value=f"{np.max(np.abs((_Ev - _Ev[0]) / abs(_Ev[0]))):.1e}"),
                ],
                justify="center",
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Runge–Kutta, although it is the more accurate method *per step*, drifts away from the
    true energy and keeps going, while the Verlet error just oscillates at a small,
    fixed size. This is why the planet "runs forever" in a Verlet calculation while a
    Runge–Kutta solution would eventually spiral in or out.

    A second long-run test on the pendulum is more subtle, and uses three methods at the
    same time: second-order Runge–Kutta, leapfrog applied directly to $(\theta,\omega)$
    as in Section 8.5.1, and Verlet.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pend_long_theta = mo.ui.slider(20, 120, value=60, step=10, label="Starting angle (degrees)")
    pend_long_T = mo.ui.slider(50, 1000, value=200, step=50, label="Duration (s)")
    return pend_long_T, pend_long_theta


@app.cell(hide_code=True)
def _(
    g_pend,
    l_pend,
    leapfrog_solve,
    mo,
    np,
    pend_long_T,
    pend_long_theta,
    pendulum_energy,
    pendulum_rhs,
    plt,
    rk2_solve,
    verlet_solve,
):
    _h = 0.01
    _T = float(pend_long_T.value)
    _N = int(round(_T / _h))
    _r0 = np.array([np.radians(pend_long_theta.value), 0.0])
    _t = np.linspace(0, _T, _N + 1)

    _rk = rk2_solve(pendulum_rhs, _r0, 0, _T, _N)
    _lf = leapfrog_solve(pendulum_rhs, _r0, 0, _T, _N)[0]
    _xs, _vs = verlet_solve(lambda x, t: -(g_pend / l_pend) * np.sin(x), [_r0[0]], [0.0], 0, _T, _N)
    _verlet = np.column_stack([_xs[:, 0], _vs[:, 0]])

    _dE = {}
    for _name, _states in [("2nd-order RK", _rk), ("leapfrog on (theta, omega)", _lf), ("Verlet", _verlet)]:
        _e = pendulum_energy(_states)
        _dE[_name] = _e - _e[0]

    _fig, _ax = plt.subplots(figsize=(9, 4.2))
    for _name, _d in _dE.items():
        _ax.plot(_t, _d, lw=1, label=_name)
    _ax.set(xlabel="t (s)", ylabel="change in energy", title="Long-run energy of the nonlinear pendulum, h = 0.01 s")
    _ax.legend()
    _ax.grid(alpha=0.3)

    _theta_gap = np.max(np.abs(_lf[:, 0] - _verlet[:, 0]))
    mo.vstack(
        [
            pend_long_theta,
            pend_long_T,
            mo.hstack(
                [
                    mo.stat(label="Runge-Kutta: drift", value=f"{_dE['2nd-order RK'][-1]:.2e}"),
                    mo.stat(label="Raw leapfrog: largest deviation", value=f"{np.max(np.abs(_dE['leapfrog on (theta, omega)'])):.2e}"),
                    mo.stat(label="Verlet: largest deviation", value=f"{np.max(np.abs(_dE['Verlet'])):.2e}"),
                    mo.stat(label="Largest gap between theta (leapfrog) and theta (Verlet)", value=f"{_theta_gap:.1e}"),
                ],
                justify="center",
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Verlet's energy stays bounded; Runge–Kutta drifts steadily; and leapfrog applied
    directly to the vector $(\theta,\omega)$ falls somewhere in between: in these runs its
    energy error was *not* bounded for long durations, which looks at first sight
    like it contradicts the time-reversal argument.

    The resolution is visible in the last statistic: the values of $\theta$ at whole
    steps from the vector leapfrog and from Verlet are **identical**. In the vector
    version both $\theta$ and $\omega$ are carried at whole *and* half steps, but the
    equations only ever couple $\theta$ at whole steps to $\omega$ at half steps (and the
    reverse), so the whole-step and half-step values form two independent
    Verlet-type sequences. Energy computed from the whole-step $\theta$ and the
    whole-step $\omega$ mixes the two sequences, which slowly drift apart, since they
    started from slightly different values (the Euler half step). Verlet follows only one
    of the two and recovers $\omega$ at whole steps with Eq. (8.73), and that is the
    quantity that has the bounded energy. The bounded-energy property at the end of
    a *full* swing relies on the solution being a single self-consistent sequence, which
    is exactly what Verlet provides.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.5.4 The modified midpoint method

    ---

    The second advantage of leapfrog is subtler: after many steps, the total error is an
    **even function of $h$**. Take one small leapfrog step, which has an error $\delta(h)$.
    By time-reversal symmetry, the backward step has error $\delta(-h)=-\delta(h)$, so
    $\delta$ is an odd function of $h$, and its Taylor series has only odd powers:

    $$
    \delta(h) = c_3h^3 + c_5h^5 + c_7h^7+\cdots \qquad (8.79\text{–}8.80)
    $$

    Accumulating over $T/h$ steps costs one power of $h$, so the *total* error contains
    only even powers, $h^2, h^4, h^6,\ldots$

    There is a catch. To get leapfrog started we took an initial half step with Euler's
    method, whose error has all powers of $h$ from $h^2$ onwards. That spoils the pure
    even structure at the start.

    **Gragg's fix.** Suppose we want to go from $t$ to $t+H$ using $n$ leapfrog steps of
    size $h=H/n$. The usual final value comes from (8.62):

    $$
    x(t+H)=x(t+H-h)+h\,f\big(x(t+H-\tfrac12h),\,t+H-\tfrac12h\big). \qquad (8.81)
    $$

    But using the same trick as in (8.73), we can also reach it from the last midpoint:

    $$
    x(t+H)=x(t+H-\tfrac12h)+\tfrac12h\,f\big(x(t+H),\,t+H\big). \qquad (8.82)
    $$

    Averaging the two gives

    $$
    x(t+H)=\tfrac12\Big[x(t+H-h)+x(t+H-\tfrac12h)+h\,f\big(x(t+H-\tfrac12h),t+H-\tfrac12h\big)
    +\tfrac12h\,f\big(x(t+H),t+H\big)\Big], \qquad (8.83)
    $$

    and it turns out that the odd-order terms introduced by the Euler start cancel in this
    average, so the total error again contains only even powers of $h$. This is
    the **modified midpoint method** (also called Gragg's method). It is rarely used on its
    own, since it offers little over leapfrog or fourth-order Runge–Kutta, but it is
    the engine of the Bulirsch–Stoer method in the next subsection.

    The implementation uses two interleaved values `r1` (half
    steps) and `r2` (whole steps) and a final averaging step. For simplicity the
    right-hand side is written as $\mathbf f(\mathbf r)$ with no explicit time, which is
    no loss of generality: a time-dependent equation can be made autonomous by adding
    $t$ as an extra variable with $dt/dt=1$.
    """)
    return


@app.cell(hide_code=True)
def _():
    def leapfrog_endpoint(f, r, H, n):
        """Leapfrog from r(t) to r(t+H) in n steps, final value from Eq. (8.81) only."""
        h = H / n
        r1 = r + 0.5 * h * f(r)
        r2 = r + h * f(r1)
        for _ in range(n - 1):
            r1 = r1 + h * f(r2)
            r2 = r2 + h * f(r1)
        return r2

    def modified_midpoint(f, r, H, n):
        """Modified midpoint method, Eq. (8.83): n steps of size H/n."""
        h = H / n
        r1 = r + 0.5 * h * f(r)
        r2 = r + h * f(r1)
        for _ in range(n - 1):
            r1 = r1 + h * f(r2)
            r2 = r2 + h * f(r1)
        return 0.5 * (r1 + r2 + 0.5 * h * f(r2))

    return leapfrog_endpoint, modified_midpoint


@app.cell(hide_code=True)
def _(leapfrog_endpoint, mo, modified_midpoint, np, plt):
    # Test equation dx/dt = -x^2, x(0) = 1, exact solution x = 1/(1+t); integrate to t = 1.
    def _f(r):
        return -(r**2)

    _H = 1.0
    _exact = 1.0 / (1.0 + _H)
    _ns = np.array([2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    _e_plain = np.array([abs(leapfrog_endpoint(_f, np.array([1.0]), _H, int(n))[0] - _exact) for n in _ns])
    _e_mod = np.array([abs(modified_midpoint(_f, np.array([1.0]), _H, int(n))[0] - _exact) for n in _ns])

    _slope_plain = np.polyfit(np.log(_ns[3:]), np.log(_e_plain[3:]), 1)[0]
    _slope_mod = np.polyfit(np.log(_ns[3:]), np.log(_e_mod[3:]), 1)[0]
    _ratio = _e_plain[-1] / _e_mod[-1]

    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _ax.loglog(_ns, _e_plain, "o-", label="leapfrog, final value from (8.81)")
    _ax.loglog(_ns, _e_mod, "s-", label="modified midpoint, (8.83)")
    _ax.loglog(_ns, _e_plain[0] * (_ns[0] / _ns) ** 2, "k--", lw=1, label="slope -2")
    _ax.set(xlabel="number of steps n (h = H/n)", ylabel="error in x(t+H)", title="Error of one big step, H = 1")
    _ax.xaxis.set_minor_formatter(plt.NullFormatter())
    _ax.legend()
    _ax.grid(True, which="both", alpha=0.3)

    mo.vstack(
        [
            mo.md(
                "**Numerical check** on $dx/dt=-x^2$, $x(0)=1$, whose exact solution is $x=1/(1+t)$. "
                f"For large $n$ both errors fall as $h^2$ (fitted slopes **{_slope_plain:.2f}** and "
                f"**{_slope_mod:.2f}** against $n$), as the even-power series predicts. The modified "
                f"midpoint method has the same order but a much smaller error constant: at $n=64$ "
                f"its error is about **{_ratio:.0f} times smaller**. The real payoff of the even "
                "structure appears when we extrapolate, next."
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.5.5 The Bulirsch–Stoer method

    ---

    The method combines the modified midpoint method with **Richardson extrapolation**,
    which we met for integrals (Section 5.4) and for derivatives (Section 5.10.5). Stay
    with one equation, $dx/dt=f(x,t)$, and a big step from $t$ to $t+H$ ($H$ is not
    necessarily small).

    **Step 1: one modified midpoint step across the whole interval.** Call the result
    $R_{1,1}$, with $h_1=H$. It will be a crude estimate, but this does not matter.

    **Step 2: two steps of size $h_2=\tfrac12H$.** Call the result $R_{2,1}$. Because the
    error is even in $h$,

    $$
    x(t+H)=R_{2,1}+c_1h_2^2+\mathcal O(h_2^4), \qquad
    x(t+H)=R_{1,1}+4c_1h_2^2+\mathcal O(h_2^4), \qquad (8.85\text{–}8.86)
    $$

    using $h_1=2h_2$. Subtracting eliminates the unknown constant $c_1$:

    $$
    c_1h_2^2=\tfrac13\,(R_{2,1}-R_{1,1}), \qquad
    x(t+H)=R_{2,2}+\mathcal O(h_2^4), \quad
    R_{2,2}=R_{2,1}+\tfrac13(R_{2,1}-R_{1,1}). \qquad (8.87\text{–}8.89)
    $$

    The new estimate $R_{2,2}$ has an error of order $h^4$, two orders better than either
    estimate that went into it, and as good as fourth-order Runge–Kutta.

    **Step 3: three steps of size $h_3=\tfrac13H$,** estimate $R_{3,1}$. The same argument
    gives $R_{3,2}=R_{3,1}+\tfrac45(R_{3,1}-R_{2,1})$, with an $h^4$ error, and combining
    $R_{3,2}$ with $R_{2,2}$ eliminates the $h^4$ term as well (using $h_2=\tfrac32h_3$, so $h_2^4=\tfrac{81}{16}h_3^4$):

    $$
    R_{3,3}=R_{3,2}+\tfrac{16}{65}(R_{3,2}-R_{2,2}), \qquad x(t+H)=R_{3,3}+\mathcal O(h_3^6). \qquad (8.95)
    $$

    Three modified midpoint calculations, error of order $h^6$.

    ### The general formula

    Increase $n$ by one each time (not by doubling, as in Romberg integration), and let
    $R_{n,1}$ be the modified midpoint estimate with $n$ steps. The estimate $R_{n,m}$ has
    an error of order $h_n^{2m}$:

    $$
    x(t+H)=R_{n,m}+c_m h_n^{2m}+\mathcal O(h_n^{2m+2}). \qquad (8.96)
    $$

    Comparing this with the same formula for $R_{n-1,m}$, and using $h_{n-1}=\tfrac{n}{n-1}h_n$:

    $$
    c_mh_n^{2m}=\frac{R_{n,m}-R_{n-1,m}}{\left[n/(n-1)\right]^{2m}-1}, \qquad (8.99)
    $$

    so that

    $$
    \boxed{\,R_{n,m+1}=R_{n,m}+\frac{R_{n,m}-R_{n-1,m}}{\left[n/(n-1)\right]^{2m}-1}\,}
    \qquad x(t+H)=R_{n,m+1}+\mathcal O(h_n^{2m+2}). \qquad (8.100\text{–}8.101)
    $$

    Equation (8.101) is the fundamental equation of the method. The estimates form a
    triangle, with one row for each value of $n$; each entry depends on its two
    neighbours: $R_{n,m}$ directly to its left, and $R_{n-1,m}$ above-left in the previous
    row. Every arrow in the diagram below is one use of the formula.
    """)
    return


@app.cell(hide_code=True)
def _(mo, plt):
    _nrows = 5
    _fig, _ax = plt.subplots(figsize=(8.2, 4.6))
    _pos = {}
    for _n in range(1, _nrows + 1):
        for _m in range(1, _n + 1):
            _pos[(_n, _m)] = (_m * 1.6, -_n * 1.0)
    # arrows first so that nodes are drawn on top
    for (_n, _m), (_x, _y) in _pos.items():
        if _m >= 2:
            _left = _pos[(_n, _m - 1)]
            _diag = _pos[(_n - 1, _m - 1)]
            _ax.annotate("", xy=(_x - 0.28, _y + 0.0), xytext=(_left[0] + 0.28, _left[1]),
                         arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=1.4, shrinkA=0, shrinkB=0))
            _ax.annotate("", xy=(_x - 0.22, _y + 0.2), xytext=(_diag[0] + 0.22, _diag[1] - 0.2),
                         arrowprops=dict(arrowstyle="-|>", color="#c2410c", lw=1.4, shrinkA=0, shrinkB=0))
    for (_n, _m), (_x, _y) in _pos.items():
        _ax.text(_x, _y, f"$R_{{{_n},{_m}}}$", ha="center", va="center", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#555555"))
    for _n in range(1, _nrows + 1):
        _ax.text(0.3, -_n, f"n = {_n}", ha="left", va="center", fontsize=10, color="#444444")
    _ax.text(1.6 * 1, 0.1, "midpoint\nestimates", ha="center", va="bottom", fontsize=9, color="#444444")
    _ax.plot([], [], color="#1d4ed8", lw=1.4, label=r"$R_{n,m}$ (left neighbour, same row)")
    _ax.plot([], [], color="#c2410c", lw=1.4, label=r"$R_{n-1,m}$ (above-left neighbour)")
    _ax.legend(loc="upper right", fontsize=9, frameon=False)
    _ax.set_xlim(0, 9)
    _ax.set_ylim(-_nrows - 0.7, 1.0)
    _ax.set_title(r"Each $R_{n,m+1}$ is built from two entries: $R_{n,m}$ and $R_{n-1,m}$")
    _ax.axis("off")
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first column consists of modified midpoint estimates, and every other entry is
    a Richardson extrapolation. The quantity $c_mh_n^{2m}$ in (8.99) is the leading error
    of the current best estimate, so the method gives us an **error estimate for free**,
    and can be made *adaptive*: keep adding rows until that estimate drops below the
    target $H\Delta$, where $\Delta$ is the desired accuracy per unit time.

    **Seeing the table.** Take the pendulum released at $179^\circ$ and one big step
    $H$ from $t=0$. The table below shows the error of the $\theta$ component of each
    $R_{n,m}$ against a very accurate reference solution (from SciPy's `DOP853`
    integrator with tight tolerances). The first column is the plain modified midpoint
    method; moving right along a row adds one extrapolation each time.
    """)
    return


@app.cell(hide_code=True)
def _(modified_midpoint):
    def extrapolation_table(f, r, H, nmax):
        """Rows n = 1..nmax of the Bulirsch-Stoer triangle, Eq. (8.101).

        rows[n-1][m-1] is R_{n,m} (a state vector)."""
        rows = []
        for n in range(1, nmax + 1):
            row = [modified_midpoint(f, r, H, n)]
            for m in range(1, n):
                eps = (row[m - 1] - rows[n - 2][m - 1]) / ((n / (n - 1)) ** (2 * m) - 1)
                row.append(row[m - 1] + eps)
            rows.append(row)
        return rows

    return (extrapolation_table,)


@app.cell(hide_code=True)
def _(mo):
    table_H = mo.ui.slider(0.05, 0.4, value=0.1, step=0.05, label="Big step H (s)")
    table_rows = mo.ui.slider(3, 7, value=6, step=1, label="Rows n")
    return table_H, table_rows


@app.cell(hide_code=True)
def _(
    extrapolation_table,
    mo,
    np,
    pendulum_rhs,
    solve_ivp,
    table_H,
    table_rows,
):
    _H = float(table_H.value)
    _nmax = int(table_rows.value)
    _r0 = np.array([179 * np.pi / 180, 0.0])
    _ref = solve_ivp(lambda t, y: pendulum_rhs(y), [0, _H], _r0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
    _rows = extrapolation_table(lambda r: pendulum_rhs(r), _r0, _H, _nmax)

    _header = "| n | " + " | ".join(f"R(n,{m})" for m in range(1, _nmax + 1)) + " |"
    _sep = "|---" * (_nmax + 1) + "|"
    _lines = [_header, _sep]
    for _n, _row in enumerate(_rows, start=1):
        _cells = []
        for _m in range(1, _nmax + 1):
            if _m <= _n:
                _val = abs(_row[_m - 1][0] - _ref[0])
                _cells.append(f"**{_val:.1e}**" if _m == _n else f"{_val:.1e}")
            else:
                _cells.append("")
        _lines.append(f"| {_n} | " + " | ".join(_cells) + " |")

    mo.vstack(
        [
            table_H,
            table_rows,
            mo.md("Error of $\\theta$ in each estimate (bold = last entry of each row, the best estimate for that $n$):"),
            mo.md("\n".join(_lines)),
            mo.md(
                "Down the first column the plain modified midpoint estimates improve slowly "
                "(error $\\propto n^{-2}$). Moving right along a row, each extrapolation removes "
                "one more power of $h^2$, and the best estimates are many orders of magnitude "
                "more accurate with only a handful of midpoint steps. In the last "
                "columns the improvement levels off, once the error approaches the limit "
                "of the reference solution and double-precision rounding."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The complete algorithm

    Let $\Delta$ be the desired accuracy per unit time, and divide the whole interval
    into $N$ big steps of length $H$. For each big step:

    1. Set $n=1$ and compute $R_{1,1}$ (one modified midpoint step across $H$).
    2. Increase $n$ by one and compute a new modified midpoint estimate $R_{n,1}$ with $n$ steps.
    3. Use Eq. (8.101) to compute the rest of the row, $R_{n,2},\ldots,R_{n,n}$.
    4. Compare the error estimate (8.99) with the target $H\Delta$. If it is larger, go back to step 2. Otherwise move on to the next big step, using $R_{n,n}$ as the new starting value.

    Only the current and previous rows of the triangle are needed, so the code below
    keeps just those two. The error is taken from the
    $\theta$ component only (via `error_slice`), and the loop begins with an artificially
    large `error` so that it runs at least once. A cap `nmax` guards against a non-converging
    step (it is not triggered in Example 8.7).
    """)
    return


@app.cell(hide_code=True)
def _(modified_midpoint, np):
    def bs_step(f, r, H, delta, nmax=30, error_slice=slice(None)):
        """One Bulirsch-Stoer big step of size H.

        Returns (new state, number of rows n used, error estimate, converged flag)."""
        n = 1
        row_prev = [modified_midpoint(f, r, H, 1)]
        error = 2 * H * delta
        while error > H * delta and n < nmax:
            n += 1
            row = [modified_midpoint(f, r, H, n)]
            for m in range(1, n):
                eps = (row[m - 1] - row_prev[m - 1]) / ((n / (n - 1)) ** (2 * m) - 1)
                row.append(row[m - 1] + eps)
            error = np.max(np.abs(eps[error_slice]))
            row_prev = row
        return row_prev[-1], n, error, bool(error <= H * delta)

    return (bs_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 8.7 — the nonlinear pendulum with Bulirsch–Stoer

    Example 8.7 solves the pendulum with $\ell=10$ cm released at
    $\theta=179^\circ$ from $t=0$ to $10$ s, using $N=100$ big steps ($H=0.1$ s) and a
    required accuracy $\Delta=10^{-8}$ per unit time. Below is the same calculation. We
    count the total number of modified midpoint steps: a row with $n$ steps
    involved $1+2+\cdots+n=n(n+1)/2$ steps in total.
    """)
    return


@app.cell(hide_code=True)
def _(bs_step, mo, np, pendulum_rhs, plt, solve_ivp):
    _a, _b, _N = 0.0, 10.0, 100
    _H = (_b - _a) / _N
    _delta = 1e-8
    _r = np.array([179 * np.pi / 180, 0.0])
    _theta = [_r[0]]
    _ns = []
    _total_steps = 0
    for _ in range(_N):
        _r, _n, _err, _ok = bs_step(lambda s: pendulum_rhs(s), _r, _H, _delta, nmax=30, error_slice=slice(0, 1))
        _theta.append(_r[0])
        _ns.append(_n)
        _total_steps += _n * (_n + 1) // 2
    _t = np.linspace(_a, _b, _N + 1)
    _ref = solve_ivp(lambda tt, y: pendulum_rhs(y), [_a, _b], [179 * np.pi / 180, 0.0],
                     method="DOP853", rtol=1e-13, atol=1e-14, t_eval=_t).y[0]

    _fig, _ax = plt.subplots(figsize=(9, 3.8))
    _ax.plot(_t, _theta, "o-", ms=3, lw=1, label="Bulirsch-Stoer (N = 100 big steps)")
    _ax.set(xlabel="t (s)", ylabel="theta (rad)", title="Example 8.7: pendulum released at 179 degrees")
    _ax.legend()
    _ax.grid(alpha=0.3)

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(label="Total modified midpoint steps", value=f"{_total_steps:,}"),
                    mo.stat(label="Rows n used per big step (min-max)", value=f"{min(_ns)}-{max(_ns)}"),
                    mo.stat(label="Largest deviation from reference", value=f"{np.max(np.abs(np.array(_theta) - _ref)):.1e}"),
                ],
                justify="center",
            ),
            mo.center(_fig),
            mo.md(
                f"Example 8.7 quotes about 3800 modified midpoint steps in total for this "
                f"calculation, against about 4200 steps for adaptive Runge–Kutta at the same "
                f"accuracy, and counting the function evaluations (four per Runge–Kutta "
                f"step, two per midpoint step) that makes the Bulirsch–Stoer method about "
                f"twice as fast. Our count is **{_total_steps:,}**."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Does a tighter target accuracy cost much? The table below repeats the calculation
    for several values of $\Delta$ (the pendulum released at $179^\circ$ is very sensitive
    to small errors, since its energy is almost exactly that of the unstable upright
    position, so the deviation from the reference is larger than the per-step target
    would suggest).
    """)
    return


@app.cell(hide_code=True)
def _(bs_step, mo, np, pendulum_rhs, solve_ivp):
    _a, _b, _N = 0.0, 10.0, 100
    _H = (_b - _a) / _N
    _t = np.linspace(_a, _b, _N + 1)
    _ref = solve_ivp(lambda tt, y: pendulum_rhs(y), [_a, _b], [179 * np.pi / 180, 0.0],
                     method="DOP853", rtol=1e-13, atol=1e-14, t_eval=_t).y[0]

    _lines = ["| target accuracy | total midpoint steps | rows n per big step | largest deviation of theta |", "|---|---|---|---|"]
    for _p in [4, 6, 8, 10]:
        _delta = 10.0 ** (-_p)
        _r = np.array([179 * np.pi / 180, 0.0])
        _theta = [_r[0]]
        _steps = 0
        _nlist = []
        for _ in range(_N):
            _r, _n, _err, _ok = bs_step(lambda s: pendulum_rhs(s), _r, _H, _delta, nmax=30, error_slice=slice(0, 1))
            _theta.append(_r[0])
            _steps += _n * (_n + 1) // 2
            _nlist.append(_n)
        _dev = np.max(np.abs(np.array(_theta) - _ref))
        _lines.append(f"| $10^{{-{_p}}}$ | {_steps:,} | {min(_nlist)}-{max(_nlist)} | {_dev:.1e} |")
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Making the accuracy a million times tighter (from $10^{-4}$ to $10^{-10}$) costs far
    fewer than a million times more work: the number of rows $n$ only creeps up. That is
    the signature of an exponentially convergent scheme, and the reason
    Bulirsch–Stoer is recommended when very high accuracy is needed.

    **Limitations.** The method only returns an accurate value at the *end* of each big
    step $H$; inside the interval there are only crude midpoint estimates. It also
    relies on a series expansion in $h$, so it works well for smooth solutions but is a
    poor choice for equations with large fluctuations or divergences, where adaptive
    Runge–Kutta (Section 8.4) is better. Finally, in practice $n$ should not exceed about
    eight or ten, which keeps $H$ modest. Both limits are addressed by splitting the whole
    interval into many big steps, and in 8.5.6 by choosing those automatically.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.5.6 Interval size for the Bulirsch–Stoer method

    ---

    In Example 8.7 the big step $H$ was picked by hand. For large calculations it is
    better to let the computer choose. The scheme:

    - Divide $[a,b]$ into $N$ intervals of size $H=(b-a)/N$ (start with a small $N$, even 1).
    - On each interval run Bulirsch–Stoer, up to a maximum $n_{\max}$ (about eight).
    - If the error target $H\Delta$ is met, accept the result and move on.
    - If $n$ reaches $n_{\max}$ first, **throw the attempt away, split the interval into two halves of size $H/2$,** and apply the method to each half in turn (each of which may be split again).

    The final subintervals can have very different sizes, small where the solution
    changes rapidly and large where it is smooth:

    ```
    |--------|---|--|-|-|-|--|---------|
    ```

    This is naturally recursive: a function that tries one step, and on failure calls
    itself twice on the two halves. The implementation below takes the current time
    `t` and appends `(t, H, n)` for each accepted subinterval to a `log` list, which is
    used for the plots.
    """)
    return


@app.cell(hide_code=True)
def _(bs_step):
    def bs_interval(f, r, t, H, delta, nmax=8, error_slice=slice(None), log=None, depth=0):
        """Advance r from t to t+H, splitting the interval in two when n reaches nmax."""
        new, n, error, converged = bs_step(f, r, H, delta, nmax, error_slice)
        if converged or depth >= 25:
            if log is not None:
                log.append((t, H, n))
            return new
        mid = bs_interval(f, r, t, H / 2, delta, nmax, error_slice, log, depth + 1)
        return bs_interval(f, mid, t + H / 2, H / 2, delta, nmax, error_slice, log, depth + 1)

    return (bs_interval,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 8.13(a) — Earth's orbit with Bulirsch–Stoer

    Same orbit as Exercise 8.12, now with a positional accuracy of **1 km per year**
    ($\Delta=1000\,\text{m}/\text{year}$ in m/s) and big steps of $H=1$ week. The state
    is $(x,y,v_x,v_y)$ and the error is measured on the position components only
    (`error_slice=slice(0, 2)`). To judge the result, the position after one year is
    compared against a high-accuracy reference, and against the Verlet solution from
    Exercise 8.12 with its one-hour step.
    """)
    return


@app.cell(hide_code=True)
def _(
    bs_interval,
    earth_r0,
    earth_v0,
    gravity_acc,
    mo,
    np,
    orbit_rhs,
    plt,
    solve_ivp,
    verlet_solve,
    year_s,
):
    _week = 7 * 24 * 3600.0
    _delta = 1000.0 / year_s
    _s0 = np.concatenate([earth_r0, earth_v0])
    _nweeks = 54
    _T = _nweeks * _week

    _states = [_s0]
    _log = []
    _s = _s0.copy()
    for _k in range(_nweeks):
        _s = bs_interval(lambda y: orbit_rhs(y), _s, _k * _week, _week, _delta, nmax=8, error_slice=slice(0, 2), log=_log)
        _states.append(_s.copy())
    _states = np.array(_states)

    _ref = solve_ivp(lambda tt, y: orbit_rhs(y), [0, _T], _s0, method="DOP853", rtol=1e-13, atol=1e-3).y[:, -1]
    _bs_err = np.hypot(*(_states[-1, :2] - _ref[:2]))

    _Nh = int(round(_T / 3600.0))
    _xs, _ = verlet_solve(gravity_acc, earth_r0, earth_v0, 0.0, _Nh * 3600.0, _Nh)
    _verlet_err = np.hypot(*(_xs[-1] - _ref[:2]))

    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _ax.plot(_states[:, 0] / 1e11, _states[:, 1] / 1e11, "o-", ms=3, lw=1)
    _ax.plot([0], [0], "o", color="gold", mec="k", ms=10)
    _ax.set(xlabel="x (10$^{11}$ m)", ylabel="y (10$^{11}$ m)", title="Earth's orbit, one point per week")
    _ax.set_aspect("equal")
    _ax.grid(alpha=0.3)

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(label="Big steps", value=f"{_nweeks} weeks"),
                    mo.stat(label="Rows n used", value=f"{sorted({n for _, _, n in _log})}"),
                    mo.stat(label="Bulirsch-Stoer: position error", value=f"{_bs_err:,.0f} m"),
                    mo.stat(label="Verlet (h = 1 h): position error", value=f"{_verlet_err / 1000:,.0f} km"),
                ],
                justify="center",
            ),
            mo.center(_fig),
            mo.md(
                "The Bulirsch–Stoer solution uses only one point per week yet is far more "
                "accurate than the Verlet solution with a one-hour step: its positional error "
                "after a year is well inside the 1 km requirement, while the Verlet position "
                "has drifted by many kilometres in phase along the orbit (even though its energy "
                "is conserved so well). This is the sense in which the method gives "
                "\"significantly more accurate\" results than the Verlet method."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 8.13(b) — the orbit of Pluto

    Same program with Pluto's initial conditions: perihelion distance
    $4.4368\times10^{12}$ m and tangential velocity $6.1218\times10^{3}$ m s$^{-1}$.
    Pluto's year is 248 Earth years, so the big step can be much larger than one week.
    The slider chooses the *initial* $H$; when the extrapolation fails to converge within
    eight rows the interval is split by the procedure of 8.5.6. The energy per unit mass
    $\tfrac12v^2-GM/r$ gives a check on the calculation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pluto_H = mo.ui.dropdown(options=["5", "10", "20", "50"], value="10", label="Initial big step H (years)")
    return (pluto_H,)


@app.cell(hide_code=True)
def _(G_const, M_sun, bs_interval, mo, np, orbit_rhs, plt, pluto_H, year_s):
    _Hy = float(pluto_H.value)
    _delta = 1000.0 / year_s
    _s = np.array([4.4368e12, 0.0, 0.0, 6.1218e3])
    _total_years = 250.0
    _nbig = int(round(_total_years / _Hy))
    _states = [_s.copy()]
    _log = []
    for _k in range(_nbig):
        _s = bs_interval(lambda y: orbit_rhs(y), _s, _k * _Hy * year_s, _Hy * year_s, _delta,
                         nmax=8, error_slice=slice(0, 2), log=_log)
        _states.append(_s.copy())
    _states = np.array(_states)

    def _specific_energy(s):
        return 0.5 * (s[:, 2] ** 2 + s[:, 3] ** 2) - G_const * M_sun / np.hypot(s[:, 0], s[:, 1])

    _E = _specific_energy(_states)
    _dE = np.max(np.abs((_E - _E[0]) / abs(_E[0])))
    _r = np.hypot(_states[:, 0], _states[:, 1])
    _Hs = np.array([h for _, h, _ in _log]) / year_s

    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _ax.plot(_states[:, 0] / 1e12, _states[:, 1] / 1e12, "o-", ms=3, lw=1)
    _ax.plot([0], [0], "o", color="gold", mec="k", ms=10)
    _ax.set(xlabel="x (10$^{12}$ m)", ylabel="y (10$^{12}$ m)", title="Pluto: one point per big step")
    _ax.set_aspect("equal")
    _ax.grid(alpha=0.3)

    mo.vstack(
        [
            pluto_H,
            mo.hstack(
                [
                    mo.stat(label="Accepted intervals", value=f"{len(_log)}"),
                    mo.stat(label="Interval sizes (years)", value=f"{_Hs.min():g} to {_Hs.max():g}"),
                    mo.stat(label="Perihelion / aphelion (10^12 m)", value=f"{_r.min() / 1e12:.2f} / {_r.max() / 1e12:.2f}"),
                    mo.stat(label="Largest relative energy change", value=f"{_dE:.1e}"),
                ],
                justify="center",
            ),
            mo.center(_fig),
            mo.md(
                "The orbit is clearly much more elliptical than the Earth's: the distance "
                "to the Sun varies by almost a factor of two. Larger initial $H$ means fewer "
                "big steps, until the extrapolation stops converging and the interval is "
                "halved automatically."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Variable interval sizes in action

    Pluto's orbit is gentle enough that the interval size barely matters. A much more
    eccentric orbit shows the subdivision at work: the comet of Exercise 8.10, starting
    $4\times10^{12}$ m from the Sun with a speed of only $500$ m s$^{-1}$ (so it falls
    almost straight toward the Sun and swings very close to it). The first big step is
    2 years; wherever the extrapolation cannot converge within eight rows, the interval is
    repeatedly halved.
    """)
    return


@app.cell(hide_code=True)
def _(bs_interval, mo, np, orbit_rhs, plt, year_s):
    _delta = 1000.0 / year_s
    _H0 = 2.0 * year_s
    _s = np.array([4e12, 0.0, 0.0, 500.0])
    _log = []
    _pts = [_s[:2].copy()]
    _t = 0.0
    _T = 80.0 * year_s
    while _t < _T - 1.0:
        _before = len(_log)
        _s = bs_interval(lambda y: orbit_rhs(y), _s, _t, _H0, _delta, nmax=8, error_slice=slice(0, 2), log=_log)
        for _tt, _hh, _nn in _log[_before:]:
            _t = _tt + _hh
        _pts.append(_s[:2].copy())
    _pts = np.array(_pts)
    _ts = np.array([tt for tt, _, _ in _log]) / year_s
    _hs = np.array([hh for _, hh, _ in _log]) / year_s

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.6))
    _axes[0].plot(_pts[:, 0] / 1e12, _pts[:, 1] / 1e12, "o-", ms=3, lw=1)
    _axes[0].plot([0], [0], "o", color="gold", mec="k", ms=10)
    _axes[0].set(xlabel="x (10$^{12}$ m)", ylabel="y (10$^{12}$ m)", title="Comet orbit, points at each 2-year step")
    _axes[0].set_aspect("equal")
    _axes[0].grid(alpha=0.3)
    _axes[1].semilogy(_ts, _hs, "s-", ms=3, lw=0.8)
    _axes[1].set(xlabel="t (years)", ylabel="interval size H (years)", title="Accepted interval sizes")
    _axes[1].grid(alpha=0.3, which="both")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(label="Accepted intervals", value=f"{len(_log)}"),
                    mo.stat(label="Largest / smallest interval (years)", value=f"{_hs.max():g} / {_hs.min():.4f}"),
                ],
                justify="center",
            ),
            mo.center(_fig),
            mo.md(
                "Far from the Sun the full 2-year intervals are accepted; as the comet "
                "approaches perihelion the intervals shrink by up to three orders of "
                "magnitude, and then grow again on the way out. The interval size adapts "
                "to the solution automatically, which is the purpose of this subsection "
                "(Exercises 8.19 and 8.20 use the same idea)."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    ---

    | Method | Order (global error) | Time-reversal symmetric | Energy over long runs | Best used for |
    |---|---|---|---|---|
    | Second-order Runge–Kutta | $h^2$ | no | drifts | a simple starting point |
    | Fourth-order Runge–Kutta | $h^4$ | no | drifts (slowly) | robust general-purpose solver |
    | Leapfrog | $h^2$ | yes | oscillates, no systematic drift on a full swing | periodic physical systems |
    | Verlet | $h^2$ | yes | bounded error (verified above) | equations of motion $\ddot{\mathbf r}=\mathbf f(\mathbf r,t)$, long runs |
    | Modified midpoint | $h^2$, even powers only | yes | – | the building block of Bulirsch–Stoer |
    | Bulirsch–Stoer | $h^{2m}$ (adaptive) | – | – | smooth problems where high accuracy is needed |

    **Key points**

    - Leapfrog's two special properties are **time-reversal symmetry** (the algorithm can be run backwards and retraces itself) and an **even error series in $h$**.
    - Time-reversal symmetry forces the energy at the end of a full period back to its starting value, so a leapfrog or Verlet orbit does not spiral in or out. A Runge–Kutta orbit slowly does.
    - **Verlet** is leapfrog specialised to $\ddot{\mathbf r}=\mathbf f(\mathbf r)$, with half the work, and velocity at whole steps from the extra half step (8.73).
    - The **modified midpoint** method restores the even error series after the Euler start.
    - **Bulirsch–Stoer** extrapolates a sequence of modified midpoint results; each new column of the triangle gains two orders of accuracy, and the extrapolation gives its own error estimate, so the method is adaptive. Dividing the time range into automatically-chosen subintervals handles the limitations of the method.

    """)
    return


if __name__ == "__main__":
    app.run()
