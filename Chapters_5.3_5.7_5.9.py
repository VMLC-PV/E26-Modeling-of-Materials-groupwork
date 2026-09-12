import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad

    return mo, np, plt, quad


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sections 5.3, 5.7 & 5.9 — Choosing the number of steps, Choosing an integration method, and Multiple integrals

    Presentation for Modeling of Materials, chapter 5 of Newman's
    *Computational Physics*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Choosing the number of steps (5.3)

    ---

    Up to now, whenever we used the trapezoidal or Simpson rule, we just picked
    some value of N — 10, 100, 1000 — and checked whether the answer looked
    reasonable. That works for a demonstration, but it isn't a real strategy: if
    a calculation needs a specific accuracy (for example, six decimal places), we need a
    way to choose N automatically instead of guessing.

    The idea behind **adaptive integration** is to pick a target accuracy
    $\delta$ and keep refining the calculation until we've reached it. If we look at
    the extended trapezoidal rule with N slices of width $h=(b-a)/N$:

    $$
    I_i \simeq h_i\left[\frac{1}{2} f(a) + \frac{1}{2} f(b) + \sum_{k=1}^{N_i-1} f(a+kh_i)\right].
    $$

    We have already seen the Euler-Maclaurin formula, which gives the
    approximation error in terms of derivatives of $f$ at the endpoints — useful
    in theory, but not always practical, since we don't always know the
    derivatives of $f$ (or even a closed form for $f$ at all, if it's coming
    from measured data). Instead, we can implement a purely numerical way to
    estimate the error: compute the integral once with $N_1$ slices (giving
    $I_1$), then again with $N_2 = 2N_1$ slices (giving $I_2$). Because the
    trapezoidal error scales as $h^2$, the difference between the two answers
    gives an estimate of the error on the *better* one:

    $$
    \delta_2 = \frac{1}{3}\left(I_2 - I_1\right).
    $$

    So the strategy is: start with a small N, keep doubling it, and stop once
    this error estimate drops below the target accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The part of this that's actually worth pointing out is that doubling N
    doesn't mean starting over. If you look at the trapezoidal sum, doubling N
    just inserts one new sample point exactly halfway between each pair of old
    points — the old points are still there, just re-weighted from "interior"
    weight 1 to whatever they already had. So the new estimate can be built from
    the old one plus only the sum over the *newly added* points, which is the
    content of Eq. (5.34) below. In computational-physics terms, the sample
    points for $N_1$ are **nested** inside the sample points for $N_2$.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    # small picture of what "doubling N" looks like in practice, N1 = 4 -> 8
    _N1 = 4
    _a, _b = 0.0, 1.0
    _old_x = np.linspace(_a, _b, _N1 + 1)
    _new_x = np.linspace(_a, _b, 2 * _N1 + 1)
    _added_x = np.setdiff1d(np.round(_new_x, 10), np.round(_old_x, 10))
 
    _fig, _ax = plt.subplots(figsize=(8, 2.6))
    _ax.scatter(_old_x, np.zeros_like(_old_x), s=100, color="crimson", zorder=3, label="points already computed")
    _ax.scatter(_added_x, np.zeros_like(_added_x), s=100, color="tab:blue", marker="^", zorder=3, label="new points, added at the midpoints")
    for _x in _old_x:
        _label = r"$\frac{1}{2}$" if (_x == _a or _x == _b) else "1"
        _ax.annotate(_label, (_x, 0.06), ha="center", fontsize=11)
    for _x in _added_x:
        _ax.annotate("1", (_x, 0.06), ha="center", fontsize=11)
    _ax.set_yticks([])
    _ax.set_xlabel("x")
    _ax.set_ylim(-0.05, 0.3)
    _ax.set_title("Going from N=4 to N=8 trapezoidal slices", pad=12)
    _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
    _fig.subplots_adjust(bottom=0.38, top=0.85)
    mo.vstack(
        [
            mo.center(_fig),
            mo.md("_The number under each point is the weight it's multiplied by in the trapezoidal sum._"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adaptive trapezoidal rule

    Written out explicitly (Eq. 5.34), the new estimate
    $I_i$ can be built from the previous one $I_{i-1}$ plus the sum over just
    the newly added, odd-indexed points:

    $$
    I_i = \frac{1}{2} I_{i-1} + h_i \sum_{\substack{k \text{ odd} \\ 1 \ldots N_i-1}} f(a+kh_i).
    $$

    The full procedure which was followed directly in the code
    below:

    1. Pick a small starting $N_1$ and a target accuracy $\delta$. Compute $I_1$.
    2. Double N, apply the formula above to get $I_i$, and estimate the error
       as $\delta_i = \frac{1}{3}(I_i - I_{i-1})$.
    3. If $|\delta_i| < \delta$, stop. Otherwise repeat step 2.

    This costs almost nothing extra compared to just running the trapezoidal
    rule once at the final N — each function value is only ever computed once,
    no matter how many doublings happen along the way.
    """)
    return


@app.cell
def _(np):
    def trapezoidal_rule(f, a, b, N):
        """Extended trapezoidal rule with N slices (Eq. 5.3)."""
        x = np.linspace(a, b, N + 1)
        y = f(x)
        h = (b - a) / N
        return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))

    return (trapezoidal_rule,)


@app.cell
def _(np):
    def adaptive_trapezoidal(f, a, b, target_accuracy, N1=1, max_doublings=20):
        """Doubles N until the error estimate from Eq. 5.34 is below
        target_accuracy, reusing the previous estimate at each step instead of
        recomputing the whole sum. Returns a list of dicts (one per doubling) with
        the running history of N, the estimate I, and the error estimate."""
        N = N1
        h = (b - a) / N
        if N > 1:
            interior = np.arange(1, N)
            I_prev = h * (0.5 * f(a) + 0.5 * f(b) + np.sum(f(a + interior * h)))
        else:
            I_prev = h * (0.5 * f(a) + 0.5 * f(b))
        history = [{"N": N, "I": I_prev, "error": None}]

        for _ in range(max_doublings):
            N *= 2
            h = (b - a) / N
            odd_k = np.arange(1, N, 2)  # the newly added points
            odd_sum = np.sum(f(a + odd_k * h))
            I_new = 0.5 * history[-1]["I"] + h * odd_sum
            error = (I_new - history[-1]["I"]) / 3
            history.append({"N": N, "I": I_new, "error": error})
            if abs(error) < target_accuracy:
                break
        return history

    return (adaptive_trapezoidal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adaptive Simpson's rule

    The same reuse trick applies to Simpson's rule, just with a bit more
    bookkeeping, since Simpson's rule already treats even- and odd-indexed
    points differently. Following Eqs. (5.36)-(5.39), define

    $$
    S_i = \frac{1}{3}\left[f(a)+f(b)+2\!\!\sum_{\substack{k \text{ even}\\2\ldots N_i-2}}\!\! f(a+kh_i)\right],
    \qquad
    T_i = \frac{2}{3}\!\!\sum_{\substack{k \text{ odd}\\1\ldots N_i-1}}\!\! f(a+kh_i).
    $$

    Then $I_i = h_i(S_i + 2T_i)$, and on the next doubling $S_i = S_{i-1} +
    T_{i-1}$ — the "even" part of the new sum is exactly the full sum from the
    previous step — so once again only $T_i$, the sum over the freshly added
    points, needs to be recomputed. The error estimate is $\delta_i =
    \frac{1}{15}(I_i - I_{i-1})$, which comes from the same kind of Richardson
    extrapolation argument as the trapezoidal case, just using the fact that
    Simpson's error scales as $h^4$ instead of $h^2$.
    """)
    return


@app.cell
def _(np):
    def adaptive_simpson(f, a, b, target_accuracy, N1=2, max_doublings=20):
        """Adaptive Simpson's rule (Eqs. 5.36-5.39). N1 must be even."""
        N = N1
        h = (b - a) / N
        even_k = np.arange(2, N, 2)
        S = (f(a) + f(b) + 2 * np.sum(f(a + even_k * h))) / 3
        odd_k = np.arange(1, N, 2)
        T = (2 / 3) * np.sum(f(a + odd_k * h))
        I_prev = h * (S + 2 * T)
        history = [{"N": N, "I": I_prev, "error": None}]

        for _ in range(max_doublings):
            S = S + T
            N *= 2
            h = (b - a) / N
            odd_k = np.arange(1, N, 2)
            T = (2 / 3) * np.sum(f(a + odd_k * h))
            I_new = h * (S + 2 * T)
            error = (I_new - history[-1]["I"]) / 15
            history.append({"N": N, "I": I_new, "error": error})
            if abs(error) < target_accuracy:
                break
        return history

    return (adaptive_simpson,)


@app.cell
def _(np):
    def integrand_ex57(x):
        # Exercise 5.7: I = integral from 0 to 1 of sin^2(sqrt(100x)) dx.
        return np.sin(np.sqrt(100 * x)) ** 2

    return (integrand_ex57,)


@app.cell(hide_code=True)
def _(mo):
    accuracy_exponent = mo.ui.slider(-8, -2, value=-6, step=1, label="target accuracy = 10 to the power of...")
    return (accuracy_exponent,)


@app.cell(hide_code=True)
def _(
    accuracy_exponent,
    adaptive_simpson,
    adaptive_trapezoidal,
    integrand_ex57,
    mo,
):
    _target = 10.0 ** accuracy_exponent.value
    _trap_hist = adaptive_trapezoidal(integrand_ex57, 0.0, 1.0, _target)
    _simp_hist = adaptive_simpson(integrand_ex57, 0.0, 1.0, _target)
    _ratio = _trap_hist[-1]["N"] / _simp_hist[-1]["N"]

    _md = mo.md(
        rf"""
        Testing this on the Exercise 5.7 integral,
        $I = \int_0^1 \sin^2(\sqrt{{100x}})\,dx$, with target accuracy
        $10^{{{accuracy_exponent.value}}}$:

        | method | N needed | final estimate | error estimate |
        |---|---|---|---|
        | adaptive trapezoidal | {_trap_hist[-1]['N']} | {_trap_hist[-1]['I']:.8f} | {abs(_trap_hist[-1]['error']):.2e} |
        | adaptive Simpson | {_simp_hist[-1]['N']} | {_simp_hist[-1]['I']:.8f} | {abs(_simp_hist[-1]['error']):.2e} |

        Both methods agree on the value of the integral (around 0.4558), but Simpson's rule gets there using
        about {_ratio:.0f}x fewer slices. That factor is exactly what we'd expect
        from the error scaling as $h^4$ instead of $h^2$: each extra doubling buys
        Simpson's rule roughly 16x the accuracy improvement instead of 4x.
        """
    )
    mo.vstack([accuracy_exponent, _md])
    return


@app.cell(hide_code=True)
def _(adaptive_simpson, adaptive_trapezoidal, integrand_ex57, mo, np, plt):
    _targets = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
    _trap_N = [adaptive_trapezoidal(integrand_ex57, 0.0, 1.0, _t)[-1]["N"] for _t in _targets]
    _simp_N = [adaptive_simpson(integrand_ex57, 0.0, 1.0, _t)[-1]["N"] for _t in _targets]

    _fig, _ax = plt.subplots()
    _ax.loglog(_targets, _trap_N, "o-", label="adaptive trapezoidal")
    _ax.loglog(_targets, _simp_N, "s-", label="adaptive Simpson")
    _ax.invert_xaxis()
    _ax.set(xlabel="target accuracy", ylabel="N needed", title="cost of reaching a target accuracy")
    _ax.legend()
    _ax.grid(True, which="both", alpha=0.3)
    mo.vstack(
        [
            mo.md(
                "Plotting this across a whole range of target accuracies shows the "
                "gap growing steadily wider — the trapezoidal rule's cost keeps "
                "roughly quadrupling for the same accuracy gain that Simpson's rule "
                "gets almost for free:"
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Choosing an integration method (5.7)

    ---

    We now have four methods available: the trapezoidal rule, Simpson's rule,
    Romberg integration, and Gaussian quadrature. Section 5.7 is the
    answer to the obvious question — which one should you actually go for?

    The underlying principle is that **higher-order methods (Romberg,
    Gaussian quadrature) are very efficient on smooth, well-behaved functions,
    but that efficiency comes from an assumption**: that the function doesn't
    do anything unexpected in between the sample points. Romberg integration
    assumes the error can be written as a power series in $h$ (so it can
    extrapolate to $h\to0$); Gaussian quadrature effectively fits a
    high-degree polynomial through very few points. If the integrand has a
    kink, a discontinuity, rapid oscillation, or noise, that assumption breaks
    down, and the "smart" extrapolation can actually make the result *worse*
    than just brute-force sampling with the trapezoidal rule.

    | Method | Sample points | Best suited to | Main weakness |
    |---|---|---|---|
    | Trapezoidal rule | equally spaced | quick estimates; noisy, rough, or singular integrands; uniformly-sampled lab data | converges slowly, only $\mathcal{O}(h^2)$ error |
    | Simpson's rule | equally spaced | smooth integrands, cheap to implement, converges as $\mathcal{O}(h^4)$ | can misbehave if the integrand is noisy or has a kink |
    | Romberg integration | equally spaced (built on the trapezoidal rule) | smooth integrands, very high accuracy for very little extra work, gives a built-in error estimate | fails badly on integrands that aren't smooth |
    | Gaussian quadrature | unevenly spaced, fixed once N is chosen | smooth integrands, extremely high accuracy from very few points, exact for polynomials up to degree $2N-1$ | points aren't nested, so increasing N means starting from scratch |
    """)
    return


@app.cell
def _(np):
    def simpsons_rule(f, a, b, N):
        """Extended Simpson's rule with N slices, N even (Eq. 5.9)."""
        x = np.linspace(a, b, N + 1)
        y = f(x)
        h = (b - a) / N
        return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))

    return (simpsons_rule,)


@app.cell
def _(trapezoidal_rule):
    def romberg_integration(f, a, b, n_levels):
        """Romberg integration (5.4): builds a triangular table of increasingly
        accurate estimates from the trapezoidal rule, using Richardson
        extrapolation (Eq. 5.51), and returns the most accurate (bottom-right)
        entry after n_levels rows. This version recomputes the trapezoidal
        estimate at each row from scratch rather than reusing evaluations as in
        5.3 — less efficient, but easier to follow, and fine for the small
        n_levels used here."""
        R_prev = [trapezoidal_rule(f, a, b, 1)]
        for i in range(1, n_levels):
            N = 2 ** i
            row = [trapezoidal_rule(f, a, b, N)]
            for m in range(1, i + 1):
                value = row[m - 1] + (row[m - 1] - R_prev[m - 1]) / (4 ** m - 1)
                row.append(value)
            R_prev = row
        return R_prev[-1]

    return (romberg_integration,)


@app.cell
def _(np):
    def gauss_quadrature(f, a, b, N):
        """Gaussian quadrature on N points, mapped from [-1, 1] onto [a, b]
        (Eqs. 5.61-5.63). Uses numpy's leggauss to get the Legendre roots and
        weights instead of writing a separate root-finder."""
        x, w = np.polynomial.legendre.leggauss(N)
        x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
        w_mapped = 0.5 * (b - a) * w
        return np.sum(w_mapped * f(x_mapped))

    return (gauss_quadrature,)


@app.cell(hide_code=True)
def _(mo):
    function_choice = mo.ui.radio(
        options=["smooth: exp(x)*cos(3x)", "rough: sqrt(|x - 0.3|)"],
        value="smooth: exp(x)*cos(3x)",
        label="pick a function to integrate on [0, 1]",
    )
    return (function_choice,)


@app.cell
def _(function_choice, np):
    if function_choice.value.startswith("smooth"):
        def bench_f(x):
            return np.exp(x) * np.cos(3 * x)
    else:
        def bench_f(x):
            # has a kink (not differentiable) at x = 0.3
            return np.sqrt(np.abs(x - 0.3))
    return (bench_f,)


@app.cell
def _(bench_f, quad):
    # scipy's quad, used only as an accurate reference value to measure error against
    true_value, _true_err = quad(bench_f, 0.0, 1.0, limit=200)
    return (true_value,)


@app.cell
def _(
    bench_f,
    gauss_quadrature,
    np,
    romberg_integration,
    simpsons_rule,
    trapezoidal_rule,
    true_value,
):
    benchmark_Ns = np.array([2, 4, 8, 16, 32, 64, 128])
    trap_err, simp_err, romb_err, gauss_err = [], [], [], []
    for _N in benchmark_Ns:
        trap_err.append(abs(trapezoidal_rule(bench_f, 0, 1, int(_N)) - true_value))
        simp_err.append(abs(simpsons_rule(bench_f, 0, 1, int(_N)) - true_value))
        _n_levels = int(np.log2(_N)) + 1
        romb_err.append(abs(romberg_integration(bench_f, 0, 1, _n_levels) - true_value))
        gauss_err.append(abs(gauss_quadrature(bench_f, 0, 1, int(_N)) - true_value))
    return benchmark_Ns, gauss_err, romb_err, simp_err, trap_err


@app.cell(hide_code=True)
def _(
    benchmark_Ns,
    function_choice,
    gauss_err,
    mo,
    np,
    plt,
    romb_err,
    simp_err,
    trap_err,
):
    _tiny = 1e-16  # so the log-log plot doesn't choke on exactly-zero error
    _fig, _ax = plt.subplots()
    _ax.loglog(benchmark_Ns, np.maximum(trap_err, _tiny), "o-", label="trapezoidal")
    _ax.loglog(benchmark_Ns, np.maximum(simp_err, _tiny), "s-", label="Simpson")
    _ax.loglog(benchmark_Ns, np.maximum(romb_err, _tiny), "^-", label="Romberg")
    _ax.loglog(benchmark_Ns, np.maximum(gauss_err, _tiny), "d-", label="Gaussian quadrature")
    _ax.set(xlabel="N (sample points)", ylabel="absolute error", title=function_choice.value)
    _ax.legend()
    _ax.grid(True, which="both", alpha=0.3)
    mo.vstack([function_choice, mo.center(_fig)])
    return


@app.cell(hide_code=True)
def _(benchmark_Ns, gauss_err, mo, romb_err, simp_err, trap_err):
    _idx = int(list(benchmark_Ns).index(32))
    mo.md(
        rf"""
        Taking N=32 as a concrete snapshot of the plot above:

        | method | error at N=32 |
        |---|---|
        | trapezoidal | {trap_err[_idx]:.2e} |
        | Simpson | {simp_err[_idx]:.2e} |
        | Romberg | {romb_err[_idx]:.2e} |
        | Gaussian quadrature | {gauss_err[_idx]:.2e} |

        For the **smooth** function, Romberg and Gaussian quadrature both reach
        machine precision (around $10^{{-15}}$ to $10^{{-16}}$) by N=32, while the
        trapezoidal rule is still many orders of magnitude away — this is the
        exactness result from previous chapters showing up directly: Gaussian
        quadrature with N points is exact for polynomials up to degree $2N-1$, and
        both $\exp(x)$ and $\cos(3x)$ are well approximated by fairly low-degree
        polynomials on a small interval like $[0,1]$.

        Switch the radio button above to the **rough** function (the one with the
        kink at $x=0.3$) and this advantage mostly disappears: none of the methods
        reach anywhere near machine precision, and the difference between them
        becomes much smaller — sometimes the trapezoidal rule is even competitive
        with Romberg. That matches the following point exactly: the "smarter" methods
        aren't smarter in general, they're smarter *for smooth functions specifically*.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Multiple integrals (5.9)

    ---

    Everything so far has been for a single integration variable. In physics,
    double (and higher) integrals come up constantly — areas, gravitational and
    electric fields from extended objects, and so on. Consider

    $$
    I = \int_0^1\int_0^1 f(x,y)\,dx\,dy.
    $$

    The trick we use is to treat this as two nested one-dimensional
    integrals. Define

    $$
    F(y) = \int_0^1 f(x,y)\,dx, \qquad \text{so that} \qquad I = \int_0^1 F(y)\,dy.
    $$

    In other words: for each fixed value of $y$, do a normal 1D integral over
    $x$ to get $F(y)$; then integrate $F(y)$ over $y$. If we approximate *both*
    integrals with $N$-point Gaussian quadrature and substitute one into the
    other, the two sums combine into a double sum — the **Gauss-Legendre
    product formula**:

    $$
    I \approx \sum_{i=1}^N \sum_{j=1}^N w_i w_j\, f(x_i, y_j). \qquad (5.114)
    $$

    So instead of N sample points we now need an $N\times N$ grid of them, and
    each point's weight is just the product of its 1D $x$- and $y$-weights.
    Since 1D Gaussian quadrature with N points is exact up to degree $2N-1$,
    this product rule is exact for any polynomial of degree up to $2N-1$ in
    *each* variable separately. Nothing about this derivation is specific to
    Gaussian quadrature, either — the same nesting argument works with the
    trapezoidal or Simpson rule, Gaussian quadrature is just the most
    efficient choice when the integrand is smooth, for the same reasons as
    in 5.7.
    """)
    return


@app.cell
def _(np):
    def gauss_double(f, ax, bx, ay, by, N):
        """2D Gauss-Legendre product rule over a rectangle (Eq. 5.114). f must
        accept and return numpy arrays elementwise (i.e. be vectorized), since
        this evaluates it on the whole N-by-N grid at once."""
        x, wx = np.polynomial.legendre.leggauss(N)
        y, wy = np.polynomial.legendre.leggauss(N)
        x_mapped = 0.5 * (bx - ax) * x + 0.5 * (bx + ax)
        y_mapped = 0.5 * (by - ay) * y + 0.5 * (by + ay)
        wx_mapped = 0.5 * (bx - ax) * wx
        wy_mapped = 0.5 * (by - ay) * wy
        X, Y = np.meshgrid(x_mapped, y_mapped, indexing="ij")
        W = np.outer(wx_mapped, wy_mapped)
        return np.sum(W * f(X, Y))

    return (gauss_double,)


@app.cell(hide_code=True)
def _(mo):
    grid_N = mo.ui.slider(3, 20, value=10, step=1, label="points per axis (N)")
    return (grid_N,)


@app.cell(hide_code=True)
def _(grid_N, mo, np, plt):
    # sample-point grid and weights, same idea as Fig. 5.5
    _x, _wx = np.polynomial.legendre.leggauss(grid_N.value)
    _X, _Y = np.meshgrid(_x, _x, indexing="ij")
    _W = np.outer(_wx, _wx)

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(_X, _Y, s=1500 * np.abs(_W), color="tab:blue", alpha=0.6, edgecolor="k", linewidth=0.5)
    _ax.set(xlabel="x", ylabel="y", title=f"2D Gaussian quadrature grid, N = {grid_N.value}\n(dot size shows the weight w_i*w_j)")
    mo.vstack([grid_N, mo.center(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a direct consequence of the 1D case: since Gaussian quadrature
    clusters more points near the edges of $[-1,1]$ (with smaller weights
    there) and fewer, more heavily weighted points near the centre, the 2D
    product grid inherits the same pattern along both axes — points bunch up
    near the boundary of the square, and the largest weights sit near the
    middle.

    ### Exercise 5.14: Gravitational pull of a uniform sheet

    A flat, uniform square metal sheet — side length $L$, total mass $M$, areal
    density $\sigma = M/L^2$ — sits in space. A 1 kg point mass is held a
    distance $z$ above the centre of the sheet, along the axis perpendicular to
    it. Integrating Newton's law of gravitation over every point of the sheet,
    the the force along that axis is given as

    $$
    F_z(z) = G\sigma z \iint_{-L/2}^{L/2}\frac{dx\,dy}{(x^2+y^2+z^2)^{3/2}}.
    $$

    Using the numbers from the exercise ($L=10$ m, $M=10{,}000$ kg), I evaluated
    this double integral with the product rule above for a range of $z$ values.
    As a sanity check, in the limit $z\to0$ an *infinite* sheet would give a
    constant field $2\pi G\sigma$ (a standard result from Gauss's law for
    gravity), so that is plotted as a reference line too.
    """)
    return


@app.cell
def _(gauss_double, np):
    G = 6.674e-11
    L = 10.0
    mass = 10_000.0
    sigma = mass / L ** 2
    infinite_sheet_field = 2 * np.pi * G * sigma

    def force_z(z, N=60):
        def integrand(x, y):
            return 1.0 / (x ** 2 + y ** 2 + z ** 2) ** 1.5
        return G * sigma * z * gauss_double(integrand, -L / 2, L / 2, -L / 2, L / 2, N)

    return force_z, infinite_sheet_field


@app.cell(hide_code=True)
def _(force_z, infinite_sheet_field, mo, np, plt):
    _z = np.linspace(0.02, 10, 250)
    _F = np.array([force_z(_zi) for _zi in _z])

    _fig, _ax = plt.subplots()
    _ax.plot(_z, _F, label="computed F_z(z)")
    _ax.axhline(infinite_sheet_field, color="crimson", ls="--", label="infinite-sheet limit, 2*pi*G*sigma")
    _ax.set(xlabel="z (m)", ylabel="F_z (N)", title="force from the sheet vs distance z")
    _ax.legend()

    mo.vstack(
        [
            mo.md(
                r"""
                Away from $z=0$, the curve behaves sensibly and even approaches the
                infinite-sheet reference line as $z$ gets smaller — but right near
                $z=0$ it dips back down instead of levelling off, which is not
                physical. This is a **sampling artifact**, not a mistake in the
                physics: as $z\to0$ the integrand becomes an extremely sharp,
                narrow spike centred at $x=y=0$, and with a *fixed* grid of only
                N=60 points per axis, that spike eventually becomes narrower than
                the spacing between sample points, so the numerical integral misses
                most of it and comes out too small. This is exactly what part (c)
                of Exercise 5.14 asks us to notice and explain — it's
                a direct illustration of the 5.7 point that Gaussian quadrature
                only "sees" the integrand at its sample points. (A fix would be to
                increase N as $z$ shrinks, or switch to an adaptive method near the
                singular point.)
                """
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Non-rectangular domains

    The integral above was over a fixed rectangle, but sometimes the limits of
    one variable depend on the other — for example,

    $$
    I = \int_0^1 dy \int_0^y dx\, f(x,y),
    $$

    which is an integral over the triangular region $0\le x\le y\le 1$. The
    same nested-integral idea still works: define $F(y)=\int_0^y f(x,y)\,dx$
    and integrate that over $y$ as before. The only difference is that the
    *inner* integral's upper limit is now $y$ itself, so for each value of $y$
    we have to remap the Gaussian nodes onto $[0,y]$ instead of a fixed
    interval — which means the sample points end up squeezed closer together
    wherever the domain is narrow (small $y$), and more spread out where it's
    wide (large $y$).
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    _N = 8
    _y_nodes, _ = np.polynomial.legendre.leggauss(_N)
    _y_mapped = 0.5 * (_y_nodes + 1)
    _x_nodes, _ = np.polynomial.legendre.leggauss(_N)

    _fig, _ax = plt.subplots(figsize=(5, 5))
    for _y in _y_mapped:
        _x_mapped = 0.5 * (_x_nodes + 1) * _y
        _ax.scatter(_x_mapped, np.full_like(_x_mapped, _y), s=30, color="tab:blue", zorder=3)
    _ax.plot([0, 1, 0, 0], [0, 1, 1, 0], "k-", lw=1)
    _ax.set(xlabel="x", ylabel="y", title="sample points over the triangle 0 <= x <= y <= 1",
            xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    mo.center(_fig)
    return


@app.cell
def _(np):
    def gauss_triangle(f, N=20):
        """Integrate f(x, y) over the triangle 0 <= x <= y <= 1 by nested
        Gaussian quadrature, where the inner (x) limits depend on the outer (y)
        variable, so the x-nodes and x-weights have to be rescaled separately
        for every value of y."""
        y_nodes, wy = np.polynomial.legendre.leggauss(N)
        y_mapped = 0.5 * (y_nodes + 1)
        wy_mapped = 0.5 * wy
        x_nodes, wx = np.polynomial.legendre.leggauss(N)

        total = 0.0
        for yi, wyi in zip(y_mapped, wy_mapped):
            x_mapped = 0.5 * (x_nodes + 1) * yi   # rescale x-nodes onto [0, yi]
            wx_mapped = 0.5 * yi * wx
            inner = np.sum(wx_mapped * f(x_mapped, yi))
            total += wyi * inner
        return total

    return (gauss_triangle,)


@app.cell(hide_code=True)
def _(gauss_triangle, mo):
    _f = lambda x, y: x * y
    _result = gauss_triangle(_f, N=20)
    mo.md(
        rf"""
        As a check, $f(x,y)=xy$ is used, which can be worked out by hand:

        $$
        \int_0^1\int_0^y xy\,dx\,dy = \int_0^1 \frac{{1}}{{2}}y^3\,dy = \frac{{1}}{{8}} = 0.125.
        $$

        The nested Gaussian-quadrature code above gives **{_result:.10f}**, matching
        to machine precision — which makes sense, since $xy$ is a low-degree
        polynomial and this is exactly the situation Gaussian quadrature handles
        exactly.

        This approach does generalize, but it starts to break down for genuinely
        irregular domains — regions with holes, disconnected pieces, or curved
        boundaries that can't be written as "$x$ between two functions of $y$". For
        those, two options exist: extend the integrand by zero outside
        the true domain and integrate over a simple rectangle that encloses it, or
        switch to Monte Carlo integration, which is covered in Chapter 10.
        """
    )
    return


if __name__ == "__main__":
    app.run()