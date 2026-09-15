import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Integrals Over Infinite Ranges



    Every method we have built so far — the trapezoidal rule, Simpson's
    rule, Romberg integration, Gaussian quadrature — needs a *finite*
    number of sample points spread over a *finite* interval $[a,b]$.
    But physics is full of integrals like

    $$
    \int_0^\infty f(x)\,dx \qquad\text{or}\qquad \int_{-\infty}^{\infty} f(x)\,dx,
    $$

    where the domain itself is unbounded. We obviously cannot sample an
    infinite range with a finite number of points directly.

    The fix is a **change of variables**: squash the infinite range down
    onto a finite one, then apply any of our usual finite-range methods
    (we'll use Gaussian quadrature, since it pairs so well with a smooth
    transformed integrand). This notebook builds that idea up from the
    basic substitution, checks it against known integrals, visualizes
    what the change of variables actually does to the sample points, and
    finally looks at where the trick starts to struggle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The standard substitution
    ---

    For an integral over $[0,\infty)$, the standard change of variables is

    $$
    z = \frac{x}{1+x} \qquad\Longleftrightarrow\qquad x = \frac{z}{1-z}.
    $$

    As $x$ runs from $0$ to $\infty$, $z$ runs from $0$ to $1$ — a finite
    interval. Differentiating $x=z/(1-z)$ gives $dx = dz/(1-z)^2$, so

    $$
    \int_0^\infty f(x)\,dx \;=\; \int_0^1 \frac{1}{(1-z)^2}\, f\!\left(\frac{z}{1-z}\right) dz.
    $$

    Nothing here is special about the constant $1$ in $z=x/(1+x)$: any
    substitution of the form $z = x/(c+x)$ works for any $c>0$, mapping
    $x=c$ to $z=\tfrac12$. Larger $c$ spreads the sample points out
    further along $x$; smaller $c$ packs them in closer to the origin.
    We'll come back to how much that choice matters in Section 6.

    For a range starting at some finite $a$ instead of $0$, shift first
    ($y=x-a$) and then apply the same substitution:

    $$
    \int_a^\infty f(x)\,dx \;=\; \int_0^1 \frac{1}{(1-z)^2}\, f\!\left(\frac{z}{1-z}+a\right) dz.
    $$

    Because Gauss–Legendre sample points lie strictly *inside* $(-1,1)$
    (never at the endpoints), the mapped points $z$ never actually reach
    $z=1$, so we never divide by zero — a convenient bonus of pairing
    this substitution with Gaussian quadrature specifically.
    """)
    return


@app.cell
def _(np):
    def gauss_quad(f, a, b, N):
        """Gaussian quadrature of f over [a, b] using N points, built from
        NumPy's Gauss-Legendre nodes/weights on the standard interval
        [-1, 1] (Eqs. 5.61-5.63)."""
        x, w = np.polynomial.legendre.leggauss(N)
        xp = 0.5 * (b - a) * x + 0.5 * (b + a)
        wp = 0.5 * (b - a) * w
        return np.sum(wp * f(xp))

    def semi_infinite_integral(f, N, a=0.0, c=1.0):
        """Integral of f(x) from x=a to infinity, via z = (x-a)/(c+x-a),
        evaluated with N-point Gaussian quadrature on z in [0, 1]."""

        def g(z):
            return f(a + c * z / (1 - z)) * c / (1 - z) ** 2

        return gauss_quad(g, 0.0, 1.0, N)

    return gauss_quad, semi_infinite_integral


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Three test integrals
    ---

    To check the method we need integrands whose exact answer we know.
    We'll use three, chosen to behave quite differently as $x\to\infty$:

    * $\displaystyle\int_0^\infty e^{-x^2}\,dx = \tfrac12\sqrt{\pi}$ —
      decays like a Gaussian, extremely smooth.
    * $\displaystyle\int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}$ —
      the Stefan–Boltzmann integral from Exercise 5.12, decays
      exponentially.
    * $\displaystyle\int_0^\infty \frac{1}{(1+x)^{1.5}}\,dx = 2$ — decays
      only algebraically (like a power law with a non-integer exponent),
      which turns out to matter a lot.
    """)
    return


@app.cell
def _(np):
    def f_gauss(x):
        return np.exp(-x**2)

    def f_planck(x):
        # x^3 / (e^x - 1); guard against overflow for very large x, where
        # the true value underflows to 0 long before exp(x) overflows.
        safe_x = np.minimum(x, 700.0)
        val = safe_x**3 / (np.exp(safe_x) - 1)
        return np.where(x > 700.0, 0.0, val)

    def f_powerlaw(x):
        return 1.0 / (1.0 + x) ** 1.5

    infinite_range_functions = {
        "Gaussian tail:  e^(−x²)": dict(f=f_gauss, true=0.8862269254527579),
        "Planck / Stefan–Boltzmann:  x³/(eˣ − 1)": dict(f=f_planck, true=6.493939402266828),
        "Power-law decay:  1/(1+x)^1.5": dict(f=f_powerlaw, true=2.0),
    }
    return f_gauss, f_planck, f_powerlaw, infinite_range_functions


@app.cell
def _(infinite_range_functions, mo):
    infinite_function_choice = mo.ui.dropdown(
        options=list(infinite_range_functions.keys()),
        value=list(infinite_range_functions.keys())[0],
        label="Integrand (0 to ∞)",
    )
    n_points_slider = mo.ui.slider(3, 60, value=20, step=1, label="Number of Gaussian quadrature points, N")
    return infinite_function_choice, n_points_slider


@app.cell
def _(
    infinite_function_choice,
    infinite_range_functions,
    mo,
    n_points_slider,
    semi_infinite_integral,
):
    _problem = infinite_range_functions[infinite_function_choice.value]
    _f, _true = _problem["f"], _problem["true"]
    _N = n_points_slider.value

    _estimate = semi_infinite_integral(_f, _N)
    _error = abs(_estimate - _true)

    mo.vstack(
        [
            mo.hstack([infinite_function_choice, n_points_slider]),
            mo.hstack(
                [
                    mo.stat(label="Estimate", value=f"{_estimate:.12f}"),
                    mo.stat(label="True value", value=f"{_true:.12f}"),
                    mo.stat(label="|error|", value=f"{_error:.3e}"),
                ],
                justify="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try the Gaussian and Planck integrands first: with only $N\approx15$–20
    points you should already be near machine precision. Now switch to
    the power-law integrand and watch the error shrink far more slowly
    as $N$ grows — we'll see exactly why in Section 4.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. What the substitution actually does to the sample points
    ---

    It's easy to lose track of what $z=x/(1+x)$ is doing geometrically.
    Gaussian quadrature places its $N$ nodes at roughly evenly-spread
    positions in $z\in(0,1)$ (more precisely, at the roots of the
    $N$th Legendre polynomial). Mapping those back to $x=z/(1-z)$ shows
    how those evenly-spread $z$ points get stretched into a set of $x$
    points that reach out towards infinity, with the spacing between
    consecutive points growing rapidly for large $x$.
    """)
    return


@app.cell
def _(infinite_function_choice, mo, n_points_slider, np, plt):
    _N = n_points_slider.value
    _z_nodes, _weights = np.polynomial.legendre.leggauss(_N)
    _z_nodes = 0.5 * _z_nodes + 0.5  # map [-1,1] -> [0,1]
    _x_nodes = _z_nodes / (1 - _z_nodes)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9, 3.2))

    _ax1.plot(_z_nodes, np.zeros_like(_z_nodes), "o", ms=4)
    _ax1.set(title="Sample points in z ∈ (0, 1)", xlabel="z", yticks=[])
    _ax1.set_xlim(0, 1)

    _finite_x = _x_nodes[_x_nodes < np.percentile(_x_nodes, 90) * 3]
    _ax2.plot(_finite_x, np.zeros_like(_finite_x), "o", ms=4, color="tab:orange")
    _ax2.set(title="Same points, mapped to x = z/(1−z)", xlabel="x", yticks=[])

    plt.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"With N = {_N} points, evenly-ish spread over z ∈ (0, 1), "
                f"the largest mapped sample point reaches out to "
                f"x ≈ {_x_nodes.max():.3g}."
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Why convergence speed depends on the integrand's tail
    ---

    Gaussian quadrature converges extremely fast — roughly a factor of
    $N^2$ per extra point — *provided the transformed integrand
    $g(z) = f(z/(1-z))/(1-z)^2$ is smooth on $[0,1]$*, including at the
    endpoint $z=1$.

    For a fast-decaying $f$ (exponential or faster), $g(z)$ and all its
    derivatives quietly go to zero as $z\to1$, and the method converges
    almost as if the integrand were a nice polynomial: a handful of
    points buys full machine precision.

    For a merely *algebraically* decaying $f(x)\sim x^{-p}$, though,
    $g(z)$ picks up a term behaving like $(1-z)^{p-2}$ near $z=1$ — not
    smooth to all orders unless $p$ is a very specific kind of integer.
    The power-series argument behind Gaussian quadrature's spectacular
    convergence breaks down, and we're back to a much more pedestrian,
    polynomial rate of convergence in $N$.
    """)
    return


@app.cell
def _(
    f_gauss,
    f_planck,
    f_powerlaw,
    infinite_range_functions,
    mo,
    np,
    plt,
    semi_infinite_integral,
):
    _N_values = np.arange(3, 41)
    _tiny = 1e-17

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    for _label, _marker in [
        ("Gaussian tail:  e^(−x²)", "o-"),
        ("Planck / Stefan–Boltzmann:  x³/(eˣ − 1)", "s-"),
        ("Power-law decay:  1/(1+x)^1.5", "^-"),
    ]:
        _problem = infinite_range_functions[_label]
        _f, _true = _problem["f"], _problem["true"]
        _errors = np.array(
            [abs(semi_infinite_integral(_f, int(_n)) - _true) for _n in _N_values]
        )
        _ax.semilogy(_N_values, np.maximum(_errors, _tiny), _marker, label=_label, ms=4)

    _ax.set(
        xlabel="Number of Gaussian quadrature points, N",
        ylabel="Absolute error",
        title="Convergence depends on how the integrand decays",
    )
    _ax.grid(alpha=0.3, which="both")
    _ax.legend(fontsize=8)

    mo.vstack(
        [
            mo.md(
                "The two smooth, fast-decaying integrands hit machine "
                "precision within a couple dozen points; the power-law "
                "integrand's error decreases far more gradually."
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Does the constant c matter?
    ---

    Recall the more general substitution $z=x/(c+x)$, which maps
    $x=c$ to $z=\tfrac12$. Changing $c$ doesn't affect the exact
    integral — it's just a reparametrization — but for a *fixed, finite*
    $N$ it does change where the sample points land relative to the
    "interesting" part of the integrand, and hence the accuracy we get.
    If $c$ is chosen far too large or too small relative to the natural
    width of $f(x)$, the points may be badly matched to where $f$
    actually varies.
    """)
    return


@app.cell
def _(f_gauss, mo, np, semi_infinite_integral):
    c_slider = mo.ui.slider(0.1, 8.0, value=1.0, step=0.1, label="c in z = x/(c + x)")
    n_fixed_slider = mo.ui.slider(3, 20, value=6, step=1, label="N (kept small on purpose)")
    return c_slider, n_fixed_slider


@app.cell
def _(c_slider, f_gauss, mo, n_fixed_slider, semi_infinite_integral):
    _true = 0.8862269254527579
    _c = c_slider.value
    _N = n_fixed_slider.value
    _estimate = semi_infinite_integral(f_gauss, _N, a=0.0, c=_c)
    _error = abs(_estimate - _true)

    mo.vstack(
        [
            mo.hstack([c_slider, n_fixed_slider]),
            mo.md(
                f"Integrating $e^{{-x^2}}$ from 0 to ∞ with N = {_N} points and "
                f"c = {_c:.1f}: estimate = `{_estimate:.10f}`, "
                f"error = `{_error:.3e}`."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With few enough points you should be able to see the error move
    around noticeably as $c$ changes — smaller $c$ crowds points near
    the origin (good if $f$ is concentrated there), while larger $c$
    spreads them further out. In practice $c=1$ is a perfectly
    reasonable default, and once $N$ is reasonably large the sensitivity
    to $c$ mostly disappears.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The full real line
    ---

    For $\int_{-\infty}^{\infty} f(x)\,dx$ we have (at least) two options.

    **Split and reuse.** Break the integral at $x=0$ and apply the
    semi-infinite substitution to each half separately:

    $$
    \int_{-\infty}^{\infty} f(x)\,dx = \int_0^\infty f(x)\,dx + \int_0^\infty f(-x)\,dx.
    $$

    **A single substitution.** Use $x=\tan z$, so $dx = dz/\cos^2 z$ and

    $$
    \int_{-\infty}^{\infty} f(x)\,dx = \int_{-\pi/2}^{\pi/2} \frac{f(\tan z)}{\cos^2 z}\,dz.
    $$

    The second is often more convenient — one substitution, one call to
    Gaussian quadrature — but both are equally valid; let's check they
    agree.
    """)
    return


@app.cell
def _(np, semi_infinite_integral):
    def full_line_tan(f, N):
        """Integral of f(x) from -infinity to infinity via x = tan(z)."""

        def g(z):
            return f(np.tan(z)) / np.cos(z) ** 2

        x, w = np.polynomial.legendre.leggauss(N)
        a, b = -np.pi / 2, np.pi / 2
        xp = 0.5 * (b - a) * x + 0.5 * (b + a)
        wp = 0.5 * (b - a) * w
        return np.sum(wp * g(xp))

    def full_line_split(f, N):
        """Integral of f(x) from -infinity to infinity, splitting at 0 and
        reusing the semi-infinite substitution on each half."""
        return semi_infinite_integral(f, N) + semi_infinite_integral(lambda x: f(-x), N)

    return full_line_split, full_line_tan


@app.cell
def _(np):
    def f_normal(x):
        return np.exp(-x**2)

    def f_lorentzian(x):
        return 1.0 / (1.0 + x**2)

    full_line_functions = {
        "Gaussian:  e^(−x²)": dict(f=f_normal, true=1.7724538509055159),
        "Lorentzian:  1/(1+x²)": dict(f=f_lorentzian, true=3.141592653589793),
    }
    return f_lorentzian, f_normal, full_line_functions


@app.cell
def _(full_line_functions, mo):
    full_line_choice = mo.ui.dropdown(
        options=list(full_line_functions.keys()),
        value=list(full_line_functions.keys())[0],
        label="Integrand (−∞ to ∞)",
    )
    full_line_n_slider = mo.ui.slider(3, 40, value=16, step=1, label="N")
    return full_line_choice, full_line_n_slider


@app.cell
def _(
    full_line_choice,
    full_line_functions,
    full_line_n_slider,
    full_line_split,
    full_line_tan,
    mo,
):
    _problem = full_line_functions[full_line_choice.value]
    _f, _true = _problem["f"], _problem["true"]
    _N = full_line_n_slider.value

    _tan_result = full_line_tan(_f, _N)
    _split_result = full_line_split(_f, _N)

    mo.vstack(
        [
            mo.hstack([full_line_choice, full_line_n_slider]),
            mo.hstack(
                [
                    mo.stat(
                        label="x = tan(z), one substitution",
                        value=f"{_tan_result:.10f}  (err {abs(_tan_result - _true):.2e})",
                    ),
                    mo.stat(
                        label="Split at 0, two substitutions",
                        value=f"{_split_result:.10f}  (err {abs(_split_result - _true):.2e})",
                    ),
                ],
                justify="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both approaches should land close to the true value, usually to
    similar accuracy for the same total number of function evaluations.
    Which one is more convenient is mostly a matter of taste and of
    whether you already have working code for the semi-infinite case.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. When this trick struggles
    ---

    The substitution $z=x/(1+x)$ (or any of its variants) is not
    magic — it works well when it turns $f$ into something smooth on
    $[0,1]$, and it can work poorly otherwise. Two situations worth
    watching for:

    * **Slow, power-law decay.** As Section 4 showed, if $f(x)$ dies off
      like $x^{-p}$ rather than exponentially, the transformed
      integrand is not smooth at $z=1$ and Gaussian quadrature loses
      its spectacular convergence rate. More points still get you
      there, just not nearly as fast.
    * **The wrong substitution for the problem.** Equation (5.67) is a
      good first guess, but it is not the only choice, and for some
      integrands a different mapping (different power $\gamma$ in
      $z=x^\gamma/(1+x^\gamma)$, say, or a different splitting point)
      converges far better. This is exactly the situation alluded to in
      Exercise 5.17 of the text: sometimes you have to experiment with
      the substitution itself, not just increase $N$.

    As a rule of thumb: if increasing $N$ substantially is not fixing
    accuracy the way it does for a well-behaved integrand, that's a sign
    the substitution — not just the sample count — is the thing to
    reconsider.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Summary

    * An infinite range of integration can be mapped onto a finite one
      with a change of variables such as $z=x/(1+x)$ (for $[0,\infty)$)
      or $x=\tan z$ (for $(-\infty,\infty)$), after which any standard
      finite-range method applies.
    * Gaussian quadrature pairs especially well with these
      substitutions, since its sample points never touch the endpoints
      of the finite interval, so the (potentially singular) endpoint of
      the mapped integrand is never actually evaluated.
    * The rate of convergence depends heavily on how the *original*
      integrand behaves at infinity: exponentially decaying integrands
      converge almost as fast as for a finite, smooth integral, while
      power-law-decaying integrands converge much more slowly.
    * The generalized substitution $z=x/(c+x)$ gives a free parameter
      $c$ that can be tuned to better match the natural scale of the
      integrand, though its effect mostly washes out once $N$ is large.
    * For the full real line, splitting at a point and reusing the
      semi-infinite substitution, or using a single substitution like
      $x=\tan z$, are both valid and usually give comparable accuracy.
    * No single substitution is best for every integrand — when
      convergence is disappointingly slow, it is often the change of
      variables itself, not just the number of sample points, that is
      worth reconsidering.
    """)
    return


if __name__ == "__main__":
    app.run()
