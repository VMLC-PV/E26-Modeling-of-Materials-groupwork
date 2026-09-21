import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.special import kn
    import pandas as pd
    try:
        from utils import plot_settings_screen
    except ImportError:
        pass
    import math
    from scipy.interpolate import interp1d
    from scipy.integrate import quad

    return go, interp1d, mo, np, pd, plt, quad


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Numerical integration

    Numerical integration is one of the most useful tools in computational
    materials science and physics.

    We often encounter quantities of the form

    \[
    I = \int_a^b f(x)\,dx
    \]

    for which

    - no convenient analytical solution exists,
    - the function is expensive to evaluate,
    - or the function is known only at discrete points.

    This notebook builds the classic quadrature rules from first principles:

    1. **Rectangle** (midpoint) and **trapezoidal** rules
    2. **Simpson's rule**, obtained by combining the two above
    3. **Convergence** behaviour of each rule
    4. **Adaptive** integration, controlling the error by step doubling
    5. **Romberg** integration, a systematic extrapolation to higher order
    6. **Difficult integrands**: discontinuities and improper integrals
    7. A physics capstone: the density of a relativistic quantum gas

    Throughout, we track a single reference example so that the methods can be
    compared directly against each other and against the known exact answer.
    """)
    return


@app.cell
def _(mo):
    md_ref_title = mo.md(
        """
        ## 1. A reference example
        ---
        """
    )
    return (md_ref_title,)


@app.cell
def _(mo):
    from functools import partial
    from wigglystuff import TangleLatex

    def ref_func(x, a, b, c):
        return a * x**4 + b * x + c

    formula_widget = mo.ui.anywidget(
        TangleLatex(
            latex=(r"I = \int_0^2 \left(\tangle{a}x^4 + \tangle{b}x + \tangle{c}\right) dx"),
            parameters={
                "a": {
                    "value": 1.0,
                    "min_value": -5.0,
                    "max_value": 5.0,
                    "step": 0.1,
                    "digits": 2,
                    "display": "number",
                    "symbol": "a",
                    "label": "Quartic coefficient",
                },
                "b": {
                    "value": -2.0,
                    "min_value": -5.0,
                    "max_value": 5.0,
                    "step": 0.1,
                    "digits": 2,
                    "display": "number",
                    "symbol": "b",
                    "label": "Linear coefficient",
                },
                "c": {
                    "value": 1.0,
                    "min_value": -5.0,
                    "max_value": 5.0,
                    "step": 0.1,
                    "digits": 2,
                    "display": "number",
                    "symbol": "c",
                    "label": "Constant term",
                },
            },
            editor="inline",
            reveal_all_on_drag=True,
        )
    )

    def f(x):
        return ref_func(
            x,
            float(formula_widget.values["a"]),
            float(formula_widget.values["b"]),
            float(formula_widget.values["c"]),
        )





    return f, formula_widget


@app.cell
def _(f, formula_widget, np):
    interval = [0.0, 2.0]
    a_ref = interval[0]
    b_ref = interval[1]

    a_val = float(formula_widget.values["a"])
    b_val = float(formula_widget.values["b"])
    c_val = float(formula_widget.values["c"])

    I_exact = (
        (a_val / 5.0) * (b_ref**5 - a_ref**5)
        + (b_val / 2.0) * (b_ref**2 - a_ref**2)
        + c_val * (b_ref - a_ref)
    )

    _x = np.linspace(a_ref, b_ref, 400)
    _y = f(_x)
    return I_exact, a_ref, a_val, b_ref, b_val, c_val


@app.cell
def _(a_val, b_val, c_val, mo):
    md_ref_text = mo.md(
        rf"""
        Consider $f(x) = {a_val:g}x^4 + {b_val:g}x + {c_val:g}$ on $[0,2]$.
        Its antiderivative is elementary, so we know the exact answer and can
        measure the error of every method below against it:
        """
    )
    return (md_ref_text,)


@app.cell
def _(
    I_exact,
    a_ref,
    a_val,
    b_ref,
    b_val,
    c_val,
    f,
    formula_widget,
    md_ref_text,
    md_ref_title,
    mo,
    np,
    plt,
):
    _xplot = np.linspace(a_ref, b_ref, 400)
    _yplot = f(_xplot)

    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, _yplot, color="crimson", lw=2, label=r"$f(x) = ax^4 + bx + c$")
    _ax.fill_between(_xplot, _yplot, alpha=0.2, color="crimson")
    _ax.axhline(0.0, color="black", lw=0.8, ls="--")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend()

    mo.vstack(
        [
            md_ref_title,
            md_ref_text,
            mo.hstack([formula_widget,mo.md(rf"""
                $$
                \huge
                = \left[\frac{{{a_val:g}}}{{5}}x^5 + \frac{{{b_val:g}}}{{2}}x^2 + {c_val:g}x\right]_0^2 = {I_exact:g}
                $$
                """)],align="center",justify="center"),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    md_rect_title = mo.md(
        """
        ## 2. Rectangle (midpoint) rule
        ---
        """
    )
    md_rect_text = mo.md(
        r"""
        Approximate the integral over a slice by the area of a rectangle whose
        height is the value of $f$ at the slice's midpoint:
        $$
        \int_a^b f(x)\,dx \approx (b-a)\, f\!\left(\frac{a+b}{2}\right).
        $$
        Splitting $[a,b]$ into $N$ slices of width $h=(b-a)/N$ and applying
        this to each one gives the **composite rectangle rule**
        $$
        \int_a^b f(x)\,dx \approx h \sum_{k=1}^{N} f(x_k), \qquad
        x_k = a + \left(k - \tfrac12\right) h.
        $$
        A Taylor expansion around the midpoint shows that a single slice has
        error $\frac{h^3}{24}f''(\xi)$, so the composite rule converges as
        $\mathcal{O}(h^2)$. The rule is exact for any linear function, since
        $f''=0$ then.
        """
    )
    return md_rect_text, md_rect_title


@app.cell
def _(np):
    def rectangle_rule(f, a, b, n):
        h = (b - a) / n
        xk = a + h / 2.0 + h * np.arange(n)
        return h * np.sum(f(xk))

    return (rectangle_rule,)


@app.cell
def _(mo):
    rect_n = mo.ui.slider(1, 40, value=5, step=1, label="Number of slices $N$")
    return (rect_n,)


@app.cell
def _(
    I_exact,
    a_ref,
    b_ref,
    f,
    md_rect_text,
    md_rect_title,
    mo,
    np,
    plt,
    rect_n,
    rectangle_rule,
):
    _n = rect_n.value
    _h = (b_ref - a_ref) / _n
    _edges = a_ref + _h * np.arange(_n + 1)
    _midpoints = a_ref + _h / 2.0 + _h * np.arange(_n)
    _heights = f(_midpoints)

    _xplot = np.linspace(a_ref, b_ref, 400)
    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, f(_xplot), color="crimson", lw=2, label="$f(x)$")
    _ax.bar(
        _edges[:-1],
        _heights,
        width=_h,
        align="edge",
        color="steelblue",
        alpha=0.4,
        edgecolor="steelblue",
        label="Rectangles",
    )
    _ax.scatter(_midpoints, _heights, color="steelblue", zorder=3, s=25)
    _ax.axhline(0.0, color="black", lw=0.8, ls="--")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend()

    _estimate = rectangle_rule(f, a_ref, b_ref, _n)
    mo.vstack(
        [
            md_rect_title,
            md_rect_text,
            mo.hstack([rect_n,mo.stat(label="Estimate", value=f"{_estimate:.6f}"),
                    mo.stat(
                        label="Absolute error",
                        value=f"{abs(I_exact - _estimate):.3e}",
                    )]),
            _fig,
        ],
        justify="center"
    )
    mo.vstack(
        [
            md_rect_title,
            mo.hstack(
                [
                    mo.vstack([md_rect_text,mo.hstack([mo.stat(label="Estimate", value=f"{_estimate:.6f}"),
                    mo.stat(label="Absolute error",value=f"{abs(I_exact - _estimate):.3e}")])]),
                    mo.vstack([rect_n,_fig],align="center"),

                ],
                widths=[0.5, 0.5],
                align="start"
            )
        ], 

    )
    return


@app.cell
def _(mo):
    md_trap_title = mo.md(
        """
        ## 3. Trapezoidal rule
        ---
        """
    )
    md_trap_text = mo.md(
        r"""
        Instead of a rectangle, approximate $f$ on each slice by the straight
        line through its two endpoints. The area under that line is a trapezoid:
        $$
        \int_a^b f(x)\,dx \approx (b-a)\,\frac{f(a)+f(b)}{2}.
        $$
        The composite rule over $N$ slices of width $h=(b-a)/N$ reads
        $$
        \int_a^b f(x)\,dx \approx h \left[\frac{f(x_0)+f(x_N)}{2} +
        \sum_{k=1}^{N-1} f(x_k)\right], \qquad x_k = a + kh.
        $$
        Its leading error term is $-\frac{h^3}{12}f''(\xi)$ per slice — same
        order as the rectangle rule, $\mathcal{O}(h^2)$ overall, but with the
        opposite sign and twice the magnitude. It is likewise exact for
        linear functions.
        """
    )
    return md_trap_text, md_trap_title


@app.cell
def _(mo):
    trap_n = mo.ui.slider(1, 40, value=5, step=1, label="Number of slices $N$")
    return (trap_n,)


@app.cell
def _(
    I_exact,
    a_ref,
    b_ref,
    f,
    md_trap_text,
    md_trap_title,
    mo,
    np,
    plt,
    trap_n,
):
    _n = trap_n.value
    _x = np.linspace(a_ref, b_ref, _n + 1)
    _y = f(_x)

    _xplot = np.linspace(a_ref, b_ref, 400)
    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, f(_xplot), color="crimson", lw=3, label="$f(x)$")
    _ax.fill_between(
        _x, _y, color="darkorange", alpha=0.4, label="Trapezoids", step=None
    )
    _ax.plot(_x, _y, color="darkorange", marker="o", ms=4)
    # add vertical lines at the trapezoid edges that stops at the function curve
    for xi in _x:
        _ax.vlines(xi, 0, f(xi), color="darkorange", lw=1, ls="-", alpha=0.9)
        # add linear segments connecting the function values at the trapezoid edges
        if xi != _x[-1]:
            _ax.plot([xi, _x[np.where(_x == xi)[0][0] + 1]], [f(xi), f(_x[np.where(_x == xi)[0][0] + 1])], color="darkorange", lw=1, ls="-", alpha=0.9)
    _ax.axhline(0.0, color="black", lw=0.8, ls="--")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend()

    def trapezoidal_rule(f, a, b, n):
        x = np.linspace(a, b, n + 1)
        y = f(x)
        h = (b - a) / n
        return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))

    _estimate = trapezoidal_rule(f, a_ref, b_ref, _n)
    mo.vstack(
        [
            md_trap_title,
            mo.hstack(
                [
                    mo.vstack([md_trap_text,mo.hstack([mo.stat(label="Estimate", value=f"{_estimate:.6f}"),
                    mo.stat(label="Absolute error",value=f"{abs(I_exact - _estimate):.3e}")])]),
                    mo.vstack([trap_n,_fig],align="center"),

                ],
                widths=[0.5, 0.5],
                align="start"
            )
        ]
    )
    return (trapezoidal_rule,)


@app.cell
def _(mo):
    md_simpson_title = mo.md(
        """
        ## 4. Simpson's rule
        ---
        """
    )
    md_simpson_text = mo.md(
        r"""
        The rectangle and trapezoidal errors have the same order but opposite
        sign and different weight:
        $$
        I - I_{\rm rect} = \frac{h^3}{24}f''(\xi) + \mathcal{O}(h^4), \qquad
        I - I_{\rm trap} = -\frac{h^3}{12}f''(\xi) + \mathcal{O}(h^4).
        $$
        Combining them in the ratio that cancels the $h^2$ term,
        $$
        I_S = \frac{2 I_{\rm rect} + I_{\rm trap}}{3},
        $$
        removes the leading error and leaves an $\mathcal{O}(h^4)$ method:
        **Simpson's rule**. Equivalently, it fits a parabola through the two
        endpoints and the midpoint of each slice. Over $N$ (even) slices with
        $h=(b-a)/N$:
        $$
        \int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(x_0) + f(x_N) +
        4\sum_{k~{\rm odd}} f(x_k) + 2\sum_{k~{\rm even}, \,k\neq 0,N} f(x_k)\right].
        $$
        Because the local parabola matches $f$, $f'$ and $f''$ at the
        midpoint, Simpson's rule integrates any cubic **exactly** — one order
        higher than either of its ingredients.
        """
    )
    return md_simpson_text, md_simpson_title


@app.cell
def _(mo):
    simpson_n = mo.ui.slider(2, 40, value=4, step=2, label="Number of slices $N$ (even)")
    return (simpson_n,)


@app.cell
def _(
    I_exact,
    a_ref,
    b_ref,
    f,
    md_simpson_text,
    md_simpson_title,
    mo,
    np,
    plt,
    simpson_n,
):
    _n = simpson_n.value
    _h = (b_ref - a_ref) / _n
    _x = a_ref + _h * np.arange(_n + 1)
    _y = f(_x)

    _xplot = np.linspace(a_ref, b_ref, 400)
    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, f(_xplot), color="crimson", lw=2, label="$f(x)$")

    for _k in range(0, _n, 2):
        _x3 = _x[_k : _k + 3]
        _y3 = _y[_k : _k + 3]
        _coeffs = np.polyfit(_x3, _y3, 2)
        _xfine = np.linspace(_x3[0], _x3[-1], 30)
        _ax.plot(
            _xfine,
            np.polyval(_coeffs, _xfine),
            color="seagreen",
            lw=2,
            label="Local parabola" if _k == 0 else None,
        )
    _ax.plot(_x, _y, "o", color="seagreen", ms=4)
    _ax.axhline(0.0, color="black", lw=0.8, ls="--")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend()

    def simpson_rule(f, a, b, n):
        if n % 2 != 0:
            raise ValueError("Simpson's rule requires an even number of slices.")
        x = np.linspace(a, b, n + 1)
        y = f(x)
        h = (b - a) / n
        return (h / 3.0) * (
            y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
        )

    _estimate = simpson_rule(f, a_ref, b_ref, _n)
    mo.vstack(
        [
            md_simpson_title,
            md_simpson_text,
            simpson_n,
            _fig,
            mo.hstack(
                [
                    mo.stat(label="Estimate", value=f"{_estimate:.6f}"),
                    mo.stat(
                        label="Absolute error",
                        value=f"{abs(I_exact - _estimate):.3e}",
                    ),
                ]
            ),
        ]
    )
    return (simpson_rule,)


@app.cell
def _(mo):
    md_exact_title = mo.md(
        """
        ### Exactness demo
        ---
        """
    )
    md_exact_text = mo.md(
        r"""
        Pick the degree of a random polynomial and compare how each rule
        performs on $[-1,2]$ using only $N=6$ slices. Simpson's rule should
        show (near) zero error up to degree 3, while the rectangle and
        trapezoidal rules only do so up to degree 1.
        """
    )
    return md_exact_text, md_exact_title


@app.cell
def _(mo):
    exactness_degree = mo.ui.dropdown(
        options={
            "Constant (degree 0)": 0,
            "Linear (degree 1)": 1,
            "Quadratic (degree 2)": 2,
            "Cubic (degree 3)": 3,
            "Quartic (degree 4)": 4,
        },
        value="Cubic (degree 3)",
        label="Polynomial degree:",
    )
    return (exactness_degree,)


@app.cell
def _(
    exactness_degree,
    md_exact_text,
    md_exact_title,
    mo,
    np,
    plt,
    rectangle_rule,
    simpson_rule,
    trapezoidal_rule,
):
    _rng = np.random.default_rng(0)
    _coeffs = _rng.uniform(-3.0, 3.0, size=exactness_degree.value + 1)

    def _poly(x):
        return np.polyval(_coeffs, x)

    _a, _b, _n = -1.0, 2.0, 8
    _antideriv = np.polyint(_coeffs) # Antiderivative of the polynomial
    _I_poly_exact = np.polyval(_antideriv, _b) - np.polyval(_antideriv, _a)

    _I_rect = rectangle_rule(_poly, _a, _b, _n)
    _I_trap = trapezoidal_rule(_poly, _a, _b, _n)
    _I_simp = simpson_rule(_poly, _a, _b, _n)

    # add a figure of the polynomial
    _xplot = np.linspace(_a, _b, 400)
    _fig, _ax = plt.subplots()
    _ax.plot(_xplot, _poly(_xplot), color="crimson", lw=2, label="$f(x)$")
    _ax.axhline(0.0, color="black", lw=0.8, ls="--")
    _ax.set(xlabel="$x$", ylabel="$f(x)$")
    _ax.legend()
    mo.vstack(
        [
            md_exact_title,
            md_exact_text,
            exactness_degree,
            mo.hstack(
                [
                    mo.stat(
                        label="Rectangle error",
                        value=f"{abs(_I_poly_exact - _I_rect):.2e}",
                    ),
                    mo.stat(
                        label="Trapezoidal error",
                        value=f"{abs(_I_poly_exact - _I_trap):.2e}",
                    ),
                    mo.stat(
                        label="Simpson error",
                        value=f"{abs(_I_poly_exact - _I_simp):.2e}",
                    ),
                ]
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    md_conv_title = mo.md(
        """
        ## 5. Convergence study
        ---
        """
    )
    md_conv_text = mo.md(
        r"""
        The error scaling predicted above, $\mathcal{O}(h^2)$ for rectangle
        and trapezoidal, $\mathcal{O}(h^4)$ for Simpson, can be verified
        empirically: plotting $\log(\text{error})$ against $\log(h)$ should
        give straight lines whose slopes match those orders.
        """
    )
    return md_conv_text, md_conv_title


@app.cell
def _(mo):
    conv_nmax = mo.ui.slider(6, 60, value=30, step=2, label="Maximum $N$ (even)")
    return (conv_nmax,)


@app.cell
def _(
    I_exact,
    a_ref,
    b_ref,
    conv_nmax,
    f,
    md_conv_text,
    md_conv_title,
    mo,
    np,
    plt,
    rectangle_rule,
    simpson_rule,
    trapezoidal_rule,
):
    _N_values = np.arange(2, conv_nmax.value + 1, 2)
    _h_values = (b_ref - a_ref) / _N_values
    _err_rect = np.array(
        [abs(I_exact - rectangle_rule(f, a_ref, b_ref, n)) for n in _N_values]
    )
    _err_trap = np.array(
        [abs(I_exact - trapezoidal_rule(f, a_ref, b_ref, n)) for n in _N_values]
    )
    _err_simp = np.array(
        [abs(I_exact - simpson_rule(f, a_ref, b_ref, n)) for n in _N_values]
    )

    _order_rect = np.polyfit(np.log(_h_values), np.log(_err_rect), 1)[0]
    _order_trap = np.polyfit(np.log(_h_values), np.log(_err_trap), 1)[0]
    _order_simp = np.polyfit(np.log(_h_values), np.log(_err_simp), 1)[0]

    _fig, _ax = plt.subplots()
    _ax.loglog(_h_values, _err_rect, "o-", label=f"Rectangle (order {_order_rect:.2f})")
    _ax.loglog(_h_values, _err_trap, "s-", label=f"Trapezoidal (order {_order_trap:.2f})")
    _ax.loglog(_h_values, _err_simp, "^-", label=f"Simpson (order {_order_simp:.2f})")
    _ax.set(xlabel="$h$", ylabel="Absolute error")
    _ax.legend()

    mo.vstack(
        [
            md_conv_title,
            md_conv_text,
            conv_nmax,
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    md_adapt_title = mo.md(
        """
        ## 6. Adaptive integration by step doubling
        ---
        """
    )
    md_adapt_text = mo.md(
        r"""
        In practice we rarely know $f''$, so we cannot predict the error
        directly — but we can *estimate* it by comparing two successive
        refinements. Doubling $N$ halves $h$; since the error scales as
        $\varepsilon = c\,h^p$,
        $$
        \varepsilon_2 = I - I_2 = c\left(\frac{h_1}{2}\right)^p, \qquad
        \varepsilon_1 = I - I_1 = c\,h_1^p = 2^p\,\varepsilon_2,
        $$
        so that
        $$
        \varepsilon_2 \approx \frac{I_2 - I_1}{2^p - 1}.
        $$
        For rectangle/trapezoidal ($p=2$) the divisor is $3$; for Simpson
        ($p=4$) it is $15$. We keep doubling $N$ until this error estimate
        drops below a target tolerance.
        """
    )
    return md_adapt_text, md_adapt_title


@app.cell
def _(np):
    def adaptive_integrate(rule, error_divisor, f, a, b, tol=1e-8, n_start=1, max_iter=24):
        n = n_start
        I_prev = rule(f, a, b, n)
        history = [(n, I_prev, np.nan)]
        for _ in range(max_iter):
            n *= 2
            I_new = rule(f, a, b, n)
            err_est = (I_new - I_prev) / error_divisor
            history.append((n, I_new, err_est))
            if abs(err_est) < tol:
                return I_new, history
            I_prev = I_new
        return I_new, history

    return (adaptive_integrate,)


@app.cell
def _(mo):
    adaptive_rule_choice = mo.ui.dropdown(
        options={
            "Rectangle (order 2)": "rectangle",
            "Trapezoidal (order 2)": "trapezoidal",
            "Simpson (order 4)": "simpson",
        },
        value="Rectangle (order 2)",
        label="Rule:",
    )
    adaptive_tol_exp = mo.ui.slider(
        -12, -2, value=-8, step=1, label="Target tolerance, $\\log_{10}(\\rm tol)$"
    )
    return adaptive_rule_choice, adaptive_tol_exp


@app.cell
def _(
    I_exact,
    a_ref,
    adaptive_integrate,
    adaptive_rule_choice,
    adaptive_tol_exp,
    b_ref,
    f,
    md_adapt_text,
    md_adapt_title,
    mo,
    np,
    pd,
    plt,
    rectangle_rule,
    simpson_rule,
    trapezoidal_rule,
):
    _rule_map = {
        "rectangle": (rectangle_rule, 3, 1),
        "trapezoidal": (trapezoidal_rule, 3, 1),
        "simpson": (simpson_rule, 15, 2),
    }
    _rule_fn, _err_div, _n_start = _rule_map[adaptive_rule_choice.value]
    _tol = 10.0 ** adaptive_tol_exp.value

    _I_final, _history = adaptive_integrate(
        _rule_fn, _err_div, f, a_ref, b_ref, tol=_tol, n_start=_n_start
    )
    _df = pd.DataFrame(_history, columns=["N", "Estimate", "Error estimate"])

    _fig, _ax = plt.subplots()
    _iterations = np.arange(1, len(_history))
    _errs = np.abs([h[2] for h in _history[1:]])
    _ax.semilogy(_iterations, _errs, "o-", color="steelblue")
    _ax.axhline(_tol, color="black", ls="--", label="Target tolerance")
    _ax.set(xlabel="Doubling iteration", ylabel="Error estimate")
    _ax.legend()

    mo.vstack(
        [
            md_adapt_title,
            md_adapt_text,
            mo.hstack([adaptive_rule_choice, adaptive_tol_exp]),
            mo.ui.table(_df),
            _fig,
            mo.hstack(
                [
                    mo.stat(label="Final $N$", value=str(_history[-1][0])),
                    mo.stat(
                        label="Absolute error",
                        value=f"{abs(I_exact - _I_final):.3e}",
                    ),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.1 - Fundamental methods for evaluating integrals
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fundamental Methods for Evaluating Integrals

    In an integral the domain of a function is divided into small units. At a point, $x$, in each unit,
    the function, $f(x)$, is multiplied by the measurement of the unit. All the products are then summed.

    This chapter will cover the numerical evaluation of an integral of a function of a single variable
    over a finite range.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.1.1 - Trapezoidal Rule
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calculating the integral of a function is equivalent to calculating the area underneath
    the function's curve within a domain.

    For instance, function $f(x)$ with domain $x=a$ to $x=b$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Trapezodial Rule Formula:**

    $$\int_{a}^{b} f(x) \, dx $$

    $$\approx h \left[ \frac{1}{2} f(a) + \frac{1}{2} f(b) + \sum_{k=1}^{n-1} f(a+kh) \right]$$

    $ n $ is number of slices

    $ h $ is slice width
    $$ h = (b-a)/n$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code
    """)
    return


@app.cell
def _(mo):
    # User controls integer input
    number_input = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=1,
        # label="Select an interval $h$ to approximate the integral (1 - 100):",
    )
    num_md = mo.md(f"""### Select an interval $n$ to approximate the integral (1 - 100):""")
    num_in = mo.vstack([num_md, number_input], align="center")
    return num_in, number_input


@app.cell
def _(mo, np, number_input, plt):
    def function(x):
        return x**2 -2*x+1

    n = int(number_input.value)
    a = 2
    b = 10

    # width of interval
    h = (b-a)/n

    # Values for the approximation 
    x = np.linspace(a, b, n+1)
    y = function(x)

    # Values for the function 
    x_function = np.linspace(0, 15, 1000)
    y_function = function(x_function)

    plt.plot(x_function, y_function, color='red', label='f(x)=$x^2 -2x+1$')
    plt.vlines(x, ymin=0, ymax=y) # draws the verticle lines for the approx
    plt.axhline(y=0, xmin=0, xmax=1, color='black') # draws the horizontal lines for the approx
    plt.plot(x, y, marker='.', color='black', markersize=10, label='approximation')

    plt.title('Trapezoidal Rule')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    # plt.show()
    graph = mo.mpl.interactive(plt.gcf())
    return graph, h, n, y


@app.cell
def _(h, mo, n, y):
    area = (h/2) * (y[0] + (2*sum(y[1:n])) + y[n])
    trap_area = mo.md(f"""### Approximated area: {area}""")
    return area, trap_area


@app.cell
def _(mo):
    trap_title = mo.md(
        r"""
        ## Trapezodial Approximation of $I = \int_{2}^{10} \ (x^2-2x+1) \, dx$"""
    )
    return (trap_title,)


@app.cell
def _(area, mo, trap_area):
    # actual integral value from x=2 to x=10
    integral = 728/3

    actual_val = mo.md(f"""### Actual integral value: {integral}""")

    results = mo.hstack(
                [trap_area, actual_val],
                justify="center", gap=5
            )


    # absolute error
    abs_error = abs(integral - area)

    # relative error
    rel_error = abs_error/integral

    absolute_error = mo.md(f"""### Absolute error: {abs_error}""")

    relative_error = mo.md(f"""### Relative error: {rel_error}""")

    error = mo.hstack(
                [absolute_error, relative_error],
                justify="center", gap=5
            )
    return error, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visuals
    """)
    return


@app.cell
def _(error, graph, mo, num_in, results, trap_title):
    mo.vstack([trap_title, num_in, results, error, graph])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.1.2 - Simpson’s rule
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **Trapezoidal Rule** estimates the area under a curve by approximating the curve with linear segments.

    The **Simpson's Rule** approximates the function with quadratic curves.

    To do define a quadratic curve we need 3 points. These 3 points can then be used to fit a quadratic within the "slices" used to integrate.

    Such that a slice with width $h$ could have the points $x = +h$,  $0$,  $-h$.

    Then the Simpson's rule will give the area underneath two adjacent slices. To do so the number of slices must therefore be even.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Simpson's Rule Formula:**

    $$\int_{a}^{b} f(x) \, dx $$

    $$\approx \frac{h}{3} \left[f(a) + f(b) + 4 * \sum_{k odd} f(a+kh) + 2* \sum_{k even} f(a+kh)\right]$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code
    """)
    return


@app.cell
def _(mo):
    # get number of slices 
    number_slices = mo.ui.number(
        start=0,
        stop=100,
        step=2,
        value=2,
        # label="Select an interval $h$ to approximate the integral (1 - 100):",
    )
    num_slice_md = mo.md(f"""### Select a number of slices $n$ to approximate the integral (must be even integer):""")
    num_slice_in = mo.vstack([num_slice_md, number_slices], align="center")

    # title     x1 * np.exp(-0.5 * x1) 
    simpson_title = mo.md(
        r"""
        ## Simpson's Rule Approximation of $I = \int_{0}^{12} \ (xe^{-0.5x}) \, dx$"""
    )
    return num_slice_in, number_slices, simpson_title


@app.cell
def _(np):
    def g(x_x):
        return x_x * np.exp(-0.5 * x_x) 

    # data for function
    x1 = np.linspace(0, 15, 1000)
    y1 = g(x1)
    return g, x1, y1


@app.cell
def _(g, mo, np, number_slices):
    # Approximate the integral with Simpson's rule 
    def approx_integral(a1, b1, n1):
        h1 = (b1 - a1)/n1 

        g_sum = (g(a1) + g(b1))
        sum_odd = 0
        sum_even = 0

        # calculate for odd terms
        for k in range(1, n1, 2): 
            sum_odd += ( g(a1+k*h1)) 
        sum_odd *= 4

        # calculate for even terms 
        for k in range (2, n1, 2):
            sum_even += ( g(a1+k*h1))
        sum_even *= 2 

        total_sum = h1/3 * (g_sum + sum_odd + sum_even)

        return total_sum

    a1 = 0
    b1 = 12
    n1 = int(number_slices.value)

    # Integral Values 

    simpson = approx_integral(a1, b1, n1)
    simpson_value = mo.md(f"""### Simpson approximation: {simpson}""")

    actual_simpson = 4 - 28 * np.exp(-6)
    actual_simpson_val = mo.md(f"""### Actual integral value: {actual_simpson}""")

    simpson_results = mo.hstack(
                [simpson_value, actual_simpson_val],
                justify="center", gap=5
            )
    return a1, b1, n1, simpson_results


@app.cell
def _(a1, b1, n1):
    # Finding the x values : 

    x_values = []

    def calculate_x (a1, h1, n1):
        a_x = a1
        for i in range (0, n1):
            a_xx = a_x+i*h1
            x_values.append(a_xx)
        print (x_values)

        return x_values


    h1 = (b1 - a1)/n1 
    calculate_x (a1, h1, n1)
    return


@app.cell
def _(a1, b1, g, interp1d, mo, n1, np, plt, x1, y1):
    # For simpson's rule plot 
    # https://www.youtube.com/watch?v=WM3GXyHGGUQ 
    x_simpson = np.linspace(a1, b1, n1+1)
    y_simpson = g(x_simpson)

    j = 0
    while j in range(n1):
        x_quad = np.linspace(x_simpson[j], x_simpson[j+2], 25)
        quadratic_func = interp1d(x_simpson, y_simpson, kind='quadratic')
        plt.plot(x_quad, quadratic_func(x_quad), linestyle='dashed', label='Simpson approx.')
        plt.fill_between(x_quad, quadratic_func(x_quad), alpha=0.15)
        j = j + 2


    # plot of the funtion
    plt.plot (x1, y1, label = '$g(x)=xe^{-0.5x}$')

    plt.title("Simpson's Rule")
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.legend()

    graph_simpson = mo.mpl.interactive(plt.gcf())
    return (graph_simpson,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visuals
    """)
    return


@app.cell
def _(graph_simpson, mo, num_slice_in, simpson_results, simpson_title):
    mo.vstack([simpson_title, num_slice_in, simpson_results, graph_simpson])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trapezodial vs Simpson
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code
    """)
    return


@app.cell
def _(mo):
    # get number of slices 
    num_n_com = mo.ui.number(
        start=0,
        stop=100,
        step=2,
        value=2,
        # label="Select an interval $h$ to approximate the integral (1 - 100):",
    )
    num_txt = mo.md(f"""### Select a number of slices $n$ to approximate the integral (must be even integer):""")
    num_txt_com = mo.vstack([num_txt, num_n_com], align="center")
    return num_n_com, num_txt_com


@app.cell
def _(np, num_n_com):
    def f_com(t):
        return 2*t**3 + 2*t**2 - 3*t + 3

    a_com = -1
    b_com = 1

    n_com = int(num_n_com.value)

    # slice width
    h_com = (b_com - a_com)/n_com

    # variables for approximation
    x_com = np.linspace(a_com, b_com, n_com+1)
    y_com = f_com(x_com)

    # Trapezodial Rule 
    trapezodial_com = (h_com/2) * (y_com[0] + (2*sum(y_com[1:n_com])) + y_com[n_com])

    # Simpson's Rule 
    simpsons_com = (h_com/3) * (y_com[0] + 4*sum(y_com[1:n_com:2]) + 2*sum(y_com[2:n_com-1:2]) + y_com[n_com])
    return f_com, n_com, simpsons_com, trapezodial_com, x_com, y_com


@app.cell
def _(mo, simpsons_com, trapezodial_com):
    # Integral Values 

    int_trapezodial_com = mo.md(f"""### Trapezodial approximation: {trapezodial_com}""")

    int_simpson_com = mo.md(f"""### Simpson approximation: {simpsons_com}""")

    actual_int_com = 22/3
    int_com = mo.md(f"""### Actual integral value: {actual_int_com}""")

    com_results = mo.hstack(
                [int_trapezodial_com, int_simpson_com, int_com],
                justify="center", gap=5
            )
    return (com_results,)


@app.cell
def _(f_com, mo, np, plt, x_com, y_com):
    x_func_com = np.linspace(-1.5, 1.5, 100)
    y_func_com = f_com(x_func_com)

    # Trapezodial Rule Plot 

    plt.plot(x_func_com, y_func_com, color='red', label='h(x)=$2x^3 + 2x^2 - 3x + 3$')
    plt.vlines(x_com, ymin=0, ymax=y_com) # draws the verticle lines for the approx
    plt.axhline(y=0, xmin=0, xmax=1, color='black') # draws the horizontal lines for the approx
    plt.plot(x_com, y_com, marker='.', color='black', markersize=10, label='approximation')

    plt.title('Trapezoidal Rule')
    plt.xlabel('x')
    plt.ylabel('h(x)')
    plt.legend()

    graph_trapezodial_com= mo.mpl.interactive(plt.gcf())
    return graph_trapezodial_com, x_func_com, y_func_com


@app.cell
def _(interp1d, mo, n_com, np, plt, x_com, x_func_com, y_com, y_func_com):
    q = 0
    while q in range(n_com):
        x_quad_com = np.linspace(x_com[q], x_com[q+2], 25)
        quadratic_func_com = interp1d(x_com, y_com, kind='quadratic')
        plt.plot(x_quad_com, quadratic_func_com(x_quad_com), linestyle='dashed', label='Simpson approx.')
        plt.fill_between(x_quad_com, quadratic_func_com(x_quad_com), alpha=0.15)
        q = q + 2


    # plot of the funtion
    plt.plot (x_func_com, y_func_com, color='red', label = '$h(x)=2x^3 + 2x^2 - 3x + 3$')

    plt.title("Simpson's Rule")
    plt.xlabel('x')
    plt.ylabel('h(x)')
    plt.legend()

    graph_simpson_com= mo.mpl.interactive(plt.gcf())
    return (graph_simpson_com,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visuals
    """)
    return


@app.cell
def _(com_results, graph_simpson_com, graph_trapezodial_com, mo, num_txt_com):
    compare_title = mo.md(
        r"""
        ## Comparison between Trapezodial and Simpson's rules for $I = \int_{-1}^{1} \ (2x^3 + 2x^2 - 3x + 3) \, dx$"""
    )


    compare_graphs = mo.hstack(
                [graph_trapezodial_com, graph_simpson_com],
                justify="center", gap=5
            )

    mo.vstack([compare_title, num_txt_com, com_results, compare_graphs])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.2 - Errors on integrals
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In numerical calculations there is usually a rounding error. But as these integration rules are only approximations the main source of error is the **approximation error**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trapezodial Rule
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Considering one slice between $x_{k-1}$ and $x_k$, a **Taylor Expansion** of the function can be made.

    $$f(x)=\sum_{n=0}^\infty \left[ \frac{f^n(x_{k-1})}{n!} (x-x_{k-1})^n \right]$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The taylor expression is then integrated form $x_{k-1}$ to $x_k$.

    $$ \int_{x_{k-1}}^{x_k} f(x) \, dx$$

    $$ ⋮ $$

    $$ =  \frac{1}{2} h \left[ f(x_{k-1}) + f(x_k)\right] + \frac{1}{4}h^2 \left[ f'(x_{k-1}) - f'(x_k) \right] + \frac{1}{12}h^3 \left[ f''(x_{k-1}) + f''(x_k)\right] + O(h^4)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This expression can then be summed over all the slices, $k$, to obtain the full integral.

    $$\sum_{k=1}^{N} \int_{x_{k-1}}^{x_k} f(x) \, dx$$

    $$= \frac{1}{2}h \sum_{k=1}^{N} \left[ f(x_{k-1}) + f(x_k) \right] + \frac{1}{4}h^2 \left[ f'(a) - f'(b)\right] + \frac{1}{12}h^3 \sum_{k=1}^{N} \left[ f''(x_{k-1}) + f''(x_k)\right] + O(h^4)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first sum is the trapezodial rule.

    The rest of the series is equal to the approximation error.

    The second term (of $h^2$) is the function evaluated at the first ($a$) and last ($b$) terms. A similar case can be seen for all **even** powers of $h$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The third term (of $h^3$) is the trapezodial rule approximation of the integral of the second derivative of the function.

    $$ \int_{a}^{b} f''(x) \, dx $$

    Multiplying by $\frac{1}{6}h^2$ and substituting $\int f''(x)\, dx = f'(x)$

    $$ ⋮ $$

    The final expression is

    $$\int_{a}^{b} f(x) \, dx $$
    $$= \frac{1}{2}h \sum_{k=1}^{N} \left[ f(x_{k-1}) + f(x_k)\right] + \frac{1}{12}h^2 \left[ f'(a) - f'(b)\right] + O(h^4)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the final expression the terms not constituting the trapezodial rule are the approximation error, $\delta$

    $$ \delta = \frac{1}{12} h^2 \left[ f'(a) - f'(b)\right]$$

    This is the first term of the **Euler–Maclaurin formula**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Rounding Error
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    is the value of the integral multiplied by the machine precision, $\epsilon \approx 10^{-16}$

    Increasing the number of slices reduces the approximation error. However there is no point in increasing the number of slices so that the approximation error becomes smaller than the rounding error as then the roudning error will dominate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Increasing the number of slices helps until the approximation and rounding errors are equal

    $$\frac{1}{12}h^2 \left[ f'(a) - f'(b)\right] \approx \epsilon \int_{a}^{b} f''(x) \, dx $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simpson's Rule
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    approximation error:

    $$\delta = \frac{1}{180} h^4 \left[ f'''(a) - f'''(b)\right]$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    rounding error:

    $$\epsilon \int_{a}^{b} f(x) \, dx$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conclusion

    **Trapezodial rule**
        - 1st order integration rule
        - Increasing n too much will cause rounding error to dominate

    **Simpson's rule**
        - 3rd order integration rule
        - Increasing n too much ($\approx$ a few thousand) slices causes the calculation to reach the limits of precision of the computer
    """)
    return


@app.cell
def _(mo, np):
    def func_e(x_e):
        return x_e**4 -2*x_e+1

    n_e1 = 10
    n_e2 = 20
    a_e = 0
    b_e = 2

    # width of interval
    h_e1 = (b_e-a_e)/n_e1
    h_e2 = (b_e-a_e)/n_e2

    # Values for the approximation 
    x_e1 = np.linspace(a_e, b_e, n_e1+1)
    y_e1 = func_e(x_e1)

    area_e1 = (h_e1/2) * (y_e1[0] + (2*sum(y_e1[1:n_e1])) + y_e1[n_e1])

    x_e2 = np.linspace(a_e, b_e, n_e2+1)
    y_e2 = func_e(x_e2)

    area_e2 = (h_e2/2) * (y_e2[0] + (2*sum(y_e2[1:n_e2])) + y_e2[n_e2])

    # Values for the function 

    # x_func_e = np.linspace(0, 15, 1000)
    # y_func_e = func_e(x_func_e)

    delta = 1/3 * (area_e2 - area_e1)

    delta_error = mo.md(f"""### The approximation error for the Trapezodial rule is: {delta}""")
    return (delta_error,)


@app.cell
def _(mo):
    integral_e = mo.md(f"""### Approximation error for: $\int_{0}^{2} x^{4}-2x+1 \, dx$ with number of slices $n=10$ and $n=20$""")

    error_formula = mo.md(f"""### $\delta=\frac{1}{3}(I_2-I_1)$ """)
    return (integral_e,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$\delta=\frac{1}{3}(I_2-I_1)$$
    """)
    return


@app.cell
def _(delta_error, integral_e, mo):
    mo.vstack([integral_e, delta_error])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.3 - Choosing the number of steps

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


@app.cell
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
    def trapezoidal_rule2(f, a, b, N):
        """Extended trapezoidal rule with N slices (Eq. 5.3)."""
        x = np.linspace(a, b, N + 1)
        y = f(x)
        h = (b - a) / N
        return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))

    return (trapezoidal_rule2,)


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


@app.cell
def _(mo):
    accuracy_exponent = mo.ui.slider(-8, -2, value=-6, step=1, label="target accuracy = 10 to the power of...")
    return (accuracy_exponent,)


@app.cell
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


@app.cell
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
    # Chapter 5.4 - Romberg Integration



    The trapezoidal rule is easy to code but converges slowly: halving the
    step size only quarters the error. Simpson's rule does better, but it
    is a separate formula built from fitting quadratics.

    Romberg integration takes a different route. It reuses the plain
    trapezoidal-rule estimates you already computed while doubling the
    number of slices, and combines them algebraically to **cancel out**
    the leading error term. Do this once and you get Simpson's rule.
    Do it again and to increase the accuracy further

    This notebook rebuilds that idea from scratch: first the trapezoidal
    doubling trick, then the single extrapolation step, then the full
    recursive scheme, and finally an adaptive version that stops as soon
    as a target accuracy is reached.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Recap: doubling the trapezoidal rule
    ---

    For an integral $I(a,b)=\int_a^b f(x)\,dx$ split into $N$ slices of
    width $h=(b-a)/N$, the trapezoidal rule reads

    $$
    I_N \;=\; h\left[\tfrac12 f(a) + \tfrac12 f(b) + \sum_{k=1}^{N-1} f(a+kh)\right].
    $$

    Suppose we have already evaluated $I_{N}$ and now double the number
    of slices to $2N$. The new set of sample points contains every old
    point plus a fresh point exactly halfway between each pair of old
    points. Nothing needs to be thrown away: the new estimate can be
    written purely in terms of the old one plus the *new* function
    evaluations,

    $$
    I_{2N} \;=\; \tfrac12 I_{N} \;+\; h_{2N}\!\!\sum_{k\ \mathrm{odd}}^{1\ldots 2N-1} f(a+k\,h_{2N}),
    $$

    where $h_{2N}=(b-a)/2N$. This is the "nesting" trick: each doubling
    costs only $N$ new function evaluations, not $2N$.

    The approximation error of the plain trapezoidal rule is, to leading
    order, proportional to $h^2$. That single fact is the seed of
    everything that follows.
    """)
    return


@app.cell
def _(np):
    def trapezoidal_sequence(f, a, b, n_levels):
        """Return I_1, I_2, ..., I_{n_levels}, the trapezoidal-rule estimate
        after 1, 2, 4, 8, ... slices, computed with the doubling/nesting
        trick of Eq. (5.34) so that every function value is evaluated once.
        """
        h = b - a
        N = 1
        I = 0.5 * h * (f(a) + f(b))
        estimates = [I]
        for _ in range(1, n_levels):
            h_new = h / 2
            odd_sum = sum(f(a + (2 * k - 1) * h_new) for k in range(1, N + 1))
            I = 0.5 * I + h_new * odd_sum
            estimates.append(I)
            N *= 2
            h = h_new
        return np.array(estimates)

    return (trapezoidal_sequence,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A test problem

    Following the book's own example, let's track
    $\displaystyle\int_0^2 (x^4 - 2x + 1)\,dx = 4.4$ (exact), together
    with two harder integrands used later in the chapter's exercises.
    Pick one and watch how the trapezoidal estimates behave as the
    number of slices doubles.
    """)
    return


@app.cell
def _(np):
    def f_poly(x):
        return x**4 - 2 * x + 1

    def f_gauss(x):
        return np.exp(-x**2)

    def f_oscillatory(x):
        return np.sin(np.sqrt(100 * x)) ** 2

    test_functions = {
        "Smooth polynomial:  x⁴ − 2x + 1  on [0, 2]": dict(
            f=f_poly, a=0.0, b=2.0, true=4.4
        ),
        "Smooth Gaussian:  e^(−x²)  on [0, 1]": dict(
            f=f_gauss, a=0.0, b=1.0, true=0.7468241328124271
        ),
        "Rough / infinite-slope:  sin²(√(100x))  on [0, 1]": dict(
            f=f_oscillatory, a=0.0, b=1.0, true=0.4558325
        ),
    }
    return (test_functions,)


@app.cell
def _(mo, test_functions):
    function_choice = mo.ui.dropdown(
        options=list(test_functions.keys()),
        value=list(test_functions.keys())[0],
        label="Integrand",
    )
    n_levels_slider = mo.ui.slider(2, 12, value=8, step=1, label="Number of doublings to compute")
    return function_choice, n_levels_slider


@app.cell
def _(
    function_choice,
    mo,
    n_levels_slider,
    np,
    test_functions,
    trapezoidal_sequence,
):
    _problem = test_functions[function_choice.value]
    _f, _a, _b, _true = _problem["f"], _problem["a"], _problem["b"], _problem["true"]

    trap_estimates = trapezoidal_sequence(_f, _a, _b, n_levels_slider.value)
    trap_N = 2 ** np.arange(len(trap_estimates))
    trap_error = np.abs(trap_estimates - _true)

    _rows = []
    for _i in range(len(trap_estimates)):
        _rows.append(
            f"N={trap_N[_i]:<6d}  I = {trap_estimates[_i]:.10f}   |error| = {trap_error[_i]:.3e}"
        )
    trap_table_text = "\n".join(_rows)

    mo.vstack(
        [
            mo.hstack([function_choice, n_levels_slider]),
            mo.md(f"```\n{trap_table_text}\n```"),
        ]
    )
    return (trap_estimates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the error roughly divides by four every time $N$ doubles —
    exactly the $O(h^2)$ behaviour we expect from the trapezoidal rule.
    To reach, say, six correct digits this way can require thousands of
    slices for an awkward integrand. Romberg's idea is to get those
    digits from the *first handful* of rows instead.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. One extrapolation step: Richardson's trick
    ---

    Write the true value of the integral as the trapezoidal estimate
    plus its error, and expand that error as a power series in $h$
    (only even powers survive):

    $$
    I = I_N + c_2 h^2 + c_4 h^4 + c_6 h^6 + \cdots
    $$

    Do this for both $I_N$ (spacing $h$) and $I_{2N}$ (spacing $h/2$):

    $$
    I = I_N + c_2 h^2 + O(h^4), \qquad
    I = I_{2N} + c_2\!\left(\tfrac{h}{2}\right)^{2} + O(h^4).
    $$

    These are two equations for the two unknowns $I$ and $c_2$.
    Eliminating $c_2$ gives an estimate of $I$ that is accurate to
    $O(h^4)$ instead of $O(h^2)$:

    $$
    I \;\approx\; I_{2N} + \frac{1}{3}\bigl(I_{2N}-I_N\bigr) \;\equiv\; R_2 .
    $$

    This costs nothing beyond arithmetic on numbers we already have. And
    remarkably, $R_2$ is **exactly Simpson's rule** evaluated on the same
    $2N$ points — we have derived Simpson's rule as a byproduct of
    cleverly combining two trapezoidal estimates.
    """)
    return


@app.cell
def _(np):
    def simpsons_rule(f, a, b, N):
        """Standard Simpson's rule with N (even) slices, for comparison."""
        if N % 2 != 0:
            raise ValueError("Simpson's rule needs an even number of slices")
        h = (b - a) / N
        x = a + h * np.arange(N + 1)
        y = f(x)
        s = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])
        return h * s / 3

    return (simpsons_rule,)


@app.cell
def _(function_choice, mo, simpsons_rule, test_functions, trap_estimates):
    _problem = test_functions[function_choice.value]
    _f, _a, _b = _problem["f"], _problem["a"], _problem["b"]

    # One Richardson step using the last two trapezoidal levels available
    _I_N, _I_2N = trap_estimates[-2], trap_estimates[-1]
    _N_2N = 2 ** (len(trap_estimates) - 1)
    R2 = _I_2N + (_I_2N - _I_N) / 3
    simpson_direct = simpsons_rule(_f, _a, _b, _N_2N)

    mo.md(
        f"""
        For the current integrand, combining the last two trapezoidal rows
        (N = {_N_2N // 2} and N = {_N_2N}):

        * Richardson extrapolation &nbsp; $R_2$ = `{R2:.12f}`
        * Direct Simpson's rule on N = {_N_2N} slices = `{simpson_direct:.12f}`
        * Difference = `{abs(R2 - simpson_direct):.2e}`  (should be at the level of rounding error)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Going further: the full Romberg recursion
    ---

    Nothing stops us from repeating the trick on the *extrapolated*
    values themselves. Define $R_{i,1}\equiv I_i$, the plain trapezoidal
    estimate at doubling level $i$. Combining two neighbouring values
    $R_{i,m}$ and $R_{i-1,m}$ — which agree to order $h^{2m-1}$ with an
    error of order $h^{2m}$ — cancels that error and produces a value
    accurate to order $h^{2m+2}$:

    $$
    \boxed{\,R_{i,m+1} \;=\; R_{i,m} + \dfrac{R_{i,m}-R_{i-1,m}}{4^{m}-1}\,}
    $$

    Arranged in a triangle, where each row is one more doubling of $N$
    and each column is one more level of extrapolation, this looks like

    $$
    \begin{array}{cccccc}
    R_{1,1} \\
    R_{2,1} & R_{2,2} \\
    R_{3,1} & R_{3,2} & R_{3,3} \\
    R_{4,1} & R_{4,2} & R_{4,3} & R_{4,4} \\
    \vdots  &         &         &         & \ddots
    \end{array}
    $$

    Column 1 is the plain trapezoidal rule. Column 2 is (equivalent to)
    Simpson's rule. Column 3 and beyond have no simple classical name —
    they are simply more accurate. The bottom-right entry of a
    triangle with $n$ rows, $R_{n,n}$, is by far the best estimate in
    the table, accurate to roughly $O(h_n^{2n})$.
    """)
    return


@app.cell
def _():
    def romberg_table(I_list):
        """Build the full Romberg triangle from a sequence of trapezoidal
        estimates I_1, I_2, ..., I_n obtained by repeated doubling.
        R[i][m] corresponds to R_{i+1, m+1} in 1-indexed textbook notation.
        """
        n = len(I_list)
        R = [[0.0] * (i + 1) for i in range(n)]
        for i in range(n):
            R[i][0] = I_list[i]
        for m in range(1, n):
            for i in range(m, n):
                R[i][m] = R[i][m - 1] + (R[i][m - 1] - R[i - 1][m - 1]) / (4**m - 1)
        return R

    def format_romberg_table(R):
        lines = []
        for i, row in enumerate(R):
            entries = "  ".join(f"{val:14.9f}" for val in row)
            lines.append(f"N={2**i:<5d}| {entries}")
        return "\n".join(lines)

    return format_romberg_table, romberg_table


@app.cell
def _(format_romberg_table, mo, romberg_table, trap_estimates):
    _R = romberg_table(list(trap_estimates))
    romberg_table_text = format_romberg_table(_R)
    romberg_best = _R[-1][-1]

    mo.vstack(
        [
            mo.md("Romberg triangle built from the trapezoidal rows above (best estimate is the bottom-right entry):"),
            mo.md(f"```\n{romberg_table_text}\n```"),
        ]
    )
    return (romberg_best,)


@app.cell
def _(function_choice, mo, romberg_best, test_functions, trap_estimates):
    _true = test_functions[function_choice.value]["true"]
    mo.hstack(
        [
            mo.stat(label="Best Romberg estimate", value=f"{romberg_best:.12f}"),
            mo.stat(label="Error vs. true value", value=f"{abs(romberg_best - _true):.3e}"),
            mo.stat(
                label="Plain trapezoidal error (same # evals)",
                value=f"{abs(trap_estimates[-1] - _true):.3e}",
            ),
        ],
        justify="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the smooth test integrands, a handful of trapezoidal rows turn,
    after extrapolation, into an answer correct to machine precision —
    something that would take an enormous number of slices with the
    plain trapezoidal rule alone.
    """)
    return


@app.cell
def _(mo):
    md_convergence_title = mo.md("## 5. Visualizing the speed-up\n---")
    md_convergence = mo.md(
        r"""
        The plot below tracks the absolute error, on a log scale, against
        the number of trapezoidal function evaluations already spent —
        for three different estimates built from *exactly the same* raw
        trapezoidal data:

        * the plain trapezoidal rule (column 1 of the triangle),
        * the single Richardson step / Simpson-equivalent estimate (column 2),
        * the best full Romberg estimate available at that row ($R_{i,i}$).

        Because Romberg integration reuses old evaluations, all three
        curves are "free" byproducts of the same sequence of trapezoidal
        calculations.
        """
    )
    return md_convergence, md_convergence_title


@app.cell
def _(
    function_choice,
    md_convergence,
    md_convergence_title,
    mo,
    n_levels_slider,
    np,
    plt,
    romberg_table,
    test_functions,
    trapezoidal_sequence,
):
    _problem = test_functions[function_choice.value]
    _f, _a, _b, _true = _problem["f"], _problem["a"], _problem["b"], _problem["true"]

    _n = n_levels_slider.value
    _I = trapezoidal_sequence(_f, _a, _b, _n)
    _R = romberg_table(list(_I))

    _N = 2 ** np.arange(_n)
    _err_trap = np.abs(np.array([_R[i][0] for i in range(_n)]) - _true)
    _err_simpson_equiv = np.abs(
        np.array([_R[i][1] if i >= 1 else np.nan for i in range(_n)]) - _true
    )
    _err_romberg_best = np.abs(np.array([_R[i][i] for i in range(_n)]) - _true)

    _tiny = 1e-17
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.semilogy(_N, np.maximum(_err_trap, _tiny), "o-", label="Trapezoidal rule (column 1)")
    _ax.semilogy(_N[1:], np.maximum(_err_simpson_equiv[1:], _tiny), "s-", label="1 extrapolation (≈ Simpson's rule)")
    _ax.semilogy(_N, np.maximum(_err_romberg_best, _tiny), "^-", label="Full Romberg, $R_{i,i}$")
    _ax.set(
        xlabel="Number of trapezoidal slices, N (= work already spent)",
        ylabel="Absolute error",
        title="Same trapezoidal data, three different estimates",
    )
    _ax.set_xscale("log", base=2)
    _ax.grid(alpha=0.3, which="both")
    _ax.legend()

    mo.vstack([md_convergence_title, md_convergence, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. An adaptive stopping rule
    ---

    In practice we rarely fix the number of levels in advance. Instead
    we pick a target accuracy $\delta$ and keep adding rows until we
    reach it. Two error estimates are available at each row $i$:

    $$
    \text{trapezoidal-only:}\quad \delta_i \approx \tfrac13\bigl(R_{i,1}-R_{i-1,1}\bigr),
    \qquad
    \text{Romberg, level } m:\quad \delta_i \approx \dfrac{R_{i,m}-R_{i-1,m}}{4^{m}-1}.
    $$

    Because each new row of the Romberg triangle also extends the
    diagonal by one entry, we simply watch the difference between
    successive *best* (diagonal) estimates and stop as soon as it drops
    below $\delta$. This typically needs far fewer rows — and hence far
    fewer function evaluations — than doing the same thing with the
    plain trapezoidal rule.
    """)
    return


@app.cell
def _(mo):
    tol_slider = mo.ui.slider(
        2, 12, value=8, step=1, label="Target accuracy: 10^(−value)"
    )
    return (tol_slider,)


@app.cell
def _(function_choice, mo, romberg_table, test_functions, tol_slider):
    _problem = test_functions[function_choice.value]
    _f, _a, _b, _true = _problem["f"], _problem["a"], _problem["b"], _problem["true"]
    _tol = 10.0 ** (-tol_slider.value)

    # --- adaptive plain trapezoidal (Eq. 5.28/5.30) ---
    _h = _b - _a
    _N = 1
    _I_prev = 0.5 * _h * (_f(_a) + _f(_b))
    _trap_evals = 2
    _trap_levels = 1
    for _ in range(60):
        _h_new = _h / 2
        _odd_sum = sum(_f(_a + (2 * k - 1) * _h_new) for k in range(1, _N + 1))
        _I_new = 0.5 * _I_prev + _h_new * _odd_sum
        _trap_evals += _N
        _trap_levels += 1
        _delta = abs(_I_new - _I_prev) / 3
        _I_prev = _I_new
        _N *= 2
        _h = _h_new
        if _delta < _tol:
            break
    trap_adaptive_result, trap_adaptive_evals, trap_adaptive_levels = _I_prev, _trap_evals, _trap_levels

    # --- adaptive Romberg: grow the triangle one row at a time ---
    _estimates = []
    _h = _b - _a
    _N = 1
    _I = 0.5 * _h * (_f(_a) + _f(_b))
    _estimates.append(_I)
    _rom_evals = 2
    for _level in range(1, 60):
        _h_new = _h / 2
        _odd_sum = sum(_f(_a + (2 * k - 1) * _h_new) for k in range(1, _N + 1))
        _I = 0.5 * _I + _h_new * _odd_sum
        _estimates.append(_I)
        _rom_evals += _N
        _N *= 2
        _h = _h_new
        _R = romberg_table(_estimates)
        _best_now = _R[-1][-1]
        _best_prev = _R[-2][-1] if len(_R) > 1 else _R[0][0]
        if abs(_best_now - _best_prev) < _tol:
            break
    romberg_adaptive_result = _R[-1][-1]
    romberg_adaptive_evals = _rom_evals
    romberg_adaptive_levels = len(_estimates)

    mo.vstack(
        [
            tol_slider,
            mo.hstack(
                [
                    mo.stat(
                        label="Adaptive trapezoidal",
                        value=f"{trap_adaptive_evals} evals, error {abs(trap_adaptive_result - _true):.2e}",
                    ),
                    mo.stat(
                        label="Adaptive Romberg",
                        value=f"{romberg_adaptive_evals} evals, error {abs(romberg_adaptive_result - _true):.2e}",
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
    For a smooth integrand this comparison is usually dramatic:
    Romberg reaches the same accuracy with a handful of evaluations
    where plain adaptive trapezoidal needs many more doublings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. When Romberg integration is *not* the right tool
    ---

    The whole scheme rests on one assumption: the error really can be
    written as a smooth power series in $h$. That requires the
    integrand to be well-behaved — enough continuous derivatives, no
    singularities, no noise.

    Switch the dropdown above to **"Rough / infinite-slope: sin²(√(100x))"**
    and look again at the convergence plot in Section 5. Near $x=0$ this
    function has an unbounded derivative, which breaks the power-series
    assumption. You should see the Romberg curves lose their dramatic
    advantage over plain Simpson's rule or even the trapezoidal rule,
    and sometimes extrapolation can briefly make things *worse* before
    it improves, because it is confidently amplifying an error trend
    that no longer holds.

    **Rule of thumb:** use Romberg integration (or Gaussian quadrature,
    covered later in the chapter) for smooth, well-behaved integrands
    where a handful of samples already capture the shape of the
    function. Fall back to the plain trapezoidal rule, or an adaptive
    method with many small steps, for integrands that are noisy,
    singular, or vary rapidly — there, extrapolating from just a few
    points is not trustworthy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Summary

    * The trapezoidal rule's doubling trick lets you reuse every
      function evaluation as you refine the grid.
    * Combining two neighbouring trapezoidal estimates cancels their
      leading $O(h^2)$ error term — one such combination reproduces
      Simpson's rule exactly.
    * Repeating the cancellation across a triangle of estimates
      (Richardson extrapolation) produces the Romberg method: each
      extra row buys two extra orders of accuracy for the cost of one
      more trapezoidal doubling.
    * An adaptive version stops as soon as neighbouring diagonal
      entries agree to within the desired tolerance, typically needing
      far fewer evaluations than the plain adaptive trapezoidal rule.
    * The method assumes a smooth integrand. For rough, singular, or
      noisy functions it can lose its advantage or even mislead —
      simpler, lower-order methods are then the safer choice.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.5 - Polynomial fitting
    While the trapezoidal rule fits straight-line segments (degree 1) and Simpson's rule fits quadratic curves (degree 2), higher-order rules fit higher-degree polynomials—such as **cubics** or **quartics**—to capture the curve more accurately.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## General weighted sum

    Every Newton–Cotes rule works by taking a **weighted sum** of the function evaluated at evenly spaced points:

    $$\int_a^b f(x) \, dx \approx \sum_{k=1}^N w_k f(x_k)$$

    You simply evaluate the function $f(x)$ at sample points $x_k$ and multiply each by a corresponding fixed weight $w_k$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Determining Weights (Newton–Cotes Approach)

    **Goal**: For a given set of sample points $\{x_1, x_2, x_3, x_4\}$, find weights $w_k$ so that the weighted sum exactly integrates polynomials up to degree 3 over $[-1, 1]$:

    $$\int_{-1}^{1} f(x) \, dx \approx w_1 f(x_1) + w_2 f(x_2) + w_3 f(x_3) + w_4 f(x_4)$$

    ### **1. Integrating constant, linear, square, cubic**:
    * $f(x) = 1   \implies \int_{-1}^{1} 1 \, dx = 2$
    * $f(x) = x   \implies \int_{-1}^{1} x \, dx = 0$
    * $f(x) = x^2  \implies \int_{-1}^{1} x^2 \, dx = \frac{2}{3}$
    * $f(x) = x^3  \implies \int_{-1}^{1} x^3 \, dx = 0$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **2. Linear System for Weights**:
    $$ \begin{bmatrix}
    1 & 1 & 1 & 1 \\
    x_1 & x_2 & x_3 & x_4 \\
    x_1^2 & x_2^2 & x_3^2 & x_4^2 \\
    x_1^3 & x_2^3 & x_3^3 & x_4^3
    \end{bmatrix}
    \begin{bmatrix}
    w_1 \\
    w_2 \\
    w_3 \\
    w_4
    \end{bmatrix}
    =
    \begin{bmatrix}
    2 \\
    0 \\
    \frac{2}{3} \\
    0
    \end{bmatrix} $$

    ### **3. Limitation of Fixed/Uniform Points**:
    * **Weight Instability**: For large $N$ with equally spaced points, weights swing wildly between large positive and negative values, magnifying numerical errors.
    """)
    return


@app.cell
def _(mo):
    n_slider_nc = mo.ui.slider(2, 100, value=8, step=1, label="Number of equally spaced points N")
    return (n_slider_nc,)


@app.cell
def _(mo, n_slider_nc, np, plt):
    _n = int(n_slider_nc.value)
    _nodes = np.linspace(-1.0, 1.0, _n)
    _vandermonde = np.array([_nodes**degree for degree in range(_n)])
    _moments = np.array([(1.0 - (-1.0) ** (degree + 1)) / (degree + 1) for degree in range(_n)])
    _weights = np.linalg.solve(_vandermonde, _moments)

    _fig, _ax_weights = plt.subplots(figsize=(8, 4.5))
    _ax_weights.axhline(0, color="black", linewidth=0.8)
    _ax_weights.plot(np.arange(1, _n + 1), _weights, color="#2878b5", marker="o", linewidth=2, label="weights")
    _ax_weights.scatter(np.arange(1, _n + 1), _weights, c=np.where(_weights >= 0, "#2878b5", "#d1495b"), edgecolor="white", zorder=3)
    _ax_weights.set_title(f"Newton–Cotes weights, N={_n}")
    _ax_weights.set_xlabel("sample point index")
    _ax_weights.set_ylabel("weight")
    _ax_weights.grid(axis="y", alpha=0.25)
    _fig.tight_layout()

    mo.vstack([n_slider_nc, _fig, mo.md(
      f"The largest absolute weight is **{np.max(np.abs(_weights)):.3g}**. "
      "Large alternating weights amplify round-off and data noise."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.6 - Gaussian Quadrature
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Legendre Polynomials $P_N(x)$

    **Definition**: An $N$-th degree polynomial.
    *($\{1, x, x^2, x^3, x^4\}$)*

    $$P_0(x) = 1$$
    $$P_1(x) = x$$
    $$P_2(x) = \frac{1}{2}(3x^2 - 1)$$
    $$P_3(x) = \frac{1}{2}(5x^3 - 3x)$$
    $$P_4(x) = \frac{1}{8}(35x^4 - 30x^2 + 3)$$

    **Normalization Condition**: $P_N(1) = 1$

    ### **Key Properties**:
    1. **Parity**: Alternates between even and odd functions ($P_N(-x) = (-1)^N P_N(x)$).
    2. **Orthogonality**: $P_N(x)$ is orthogonal to every polynomial of lower degree over $[-1, 1]$.
    3. **Real Roots**: $P_N(x)$ has exactly $N$ real roots, all lying in the interval $[-1, 1]$.
    """)
    return


@app.cell
def _(mo):
    degree_slider = mo.ui.slider(1, 8, value=4, step=1, label="Highest degree")
    return (degree_slider,)


@app.cell
def _(degree_slider, mo, np, plt):
    _max_degree = int(degree_slider.value)
    _x = np.linspace(-1.0, 1.0, 600)
    _fig, _ax = plt.subplots(figsize=(9, 4.5))
    _colors = plt.cm.viridis(np.linspace(0.1, 0.9, _max_degree + 1))
    for _degree in range(_max_degree + 1):
      _coefficients = np.zeros(_degree + 1)
      _coefficients[-1] = 1.0
      _values = np.polynomial.legendre.legval(_x, _coefficients)
      _ax.plot(_x, _values, color=_colors[_degree], label=f"P$_{{{_degree}}}$(x)")
      if _degree == _max_degree and _degree > 0:
        _roots = np.polynomial.legendre.legroots(_coefficients)
        _ax.scatter(_roots, np.zeros_like(_roots), color="#d1495b", edgecolor="white", zorder=4, label="roots")
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.axvline(0, color="0.75", linewidth=0.8)
    _ax.set_title("Legendre polynomials and the roots used by Gaussian quadrature")
    _ax.set_xlabel("x")
    _ax.set_ylabel("P$_n$(x)")
    _ax.set_xlim(-1, 1)
    _ax.grid(alpha=0.2)
    _ax.legend(ncol=3, frameon=False, fontsize=9)
    _fig.tight_layout()

    mo.vstack([degree_slider, _fig, mo.md(
      f"The highlighted polynomial has **{_max_degree}** roots in $[-1, 1]$, "
      "which become the Gaussian sample points."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing the Sample Points (Gaussian Quadrature)


    ### **1. Polynomial Division**:
    Dividing $f(x)$ (degree $\le 2N-1$) by the $N$-th Legendre polynomial $P_N(x)$ gives:
    $$f(x) = q(x) P_N(x) + r(x)$$
    *(where $q(x)$ and $r(x)$ are both polynomials of degree $\le N-1$)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **2. Integrate Over $[-1, 1]$**:
    $$\int_{-1}^{1} f(x) \, dx = \int_{-1}^{1} q(x) P_N(x) \, dx + \int_{-1}^{1} r(x) \, dx$$
    * By **orthogonality**, $\int_{-1}^{1} q(x) P_N(x) \, dx = 0$.
    * Therefore, $\int_{-1}^{1} f(x) \, dx = \int_{-1}^{1} r(x) \, dx$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **3. Choose $x_k$ as the Roots of $P_N(x)$**:
    * When evaluating $f(x)$ at each sample point $x_k$, the weighted sum is:
    $$\sum_{k=1}^N w_k f(x_k) = \sum_{k=1}^N w_k q(x_k) P_N(x_k) + \sum_{k=1}^N w_k r(x_k) \quad$$

    * By choosing $x_k$ to be the roots of $P_N(x)$, $P_N(x_k) = 0$ for every $k$, leaving only:
    $$\sum_{k=1}^N w_k r(x_k)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Conclusion**:
    $$\int_{-1}^{1} f(x) \, dx = \sum_{k=1}^N w_k f(x_k) \quad \text{(Exact for degree } \le 2N-1 \text{)}$$
    * **No wild weight oscillations**: Sample points cluster near interval boundaries, producing smooth, well-behaved weights.
    """)
    return


@app.cell
def _(mo):
    gaussian_points_slider = mo.ui.slider(2, 100, value=6, step=1, label="Number of Gaussian points N")
    return (gaussian_points_slider,)


@app.cell
def _(gaussian_points_slider, mo, plt):
    from scipy.special import roots_legendre

    _n = int(gaussian_points_slider.value)
    _nodes, _weights = roots_legendre(_n)
    _fig, _ax = plt.subplots(figsize=(9, 4.2))
    _markerline, _stemlines, _baseline = _ax.stem(_nodes, _weights, linefmt="#2878b5", markerfmt="o", basefmt="k-")
    plt.setp(_stemlines, linewidth=1.8)
    plt.setp(_markerline, markersize=6, markerfacecolor="#d1495b", markeredgecolor="white")
    _ax.set_title(f"Gaussian quadrature sample points and weights, N={_n}")
    _ax.set_xlabel("sample point $x_k$ (roots of $P_N$)")
    _ax.set_ylabel("weight $w_k$")
    _ax.set_xlim(-1.05, 1.05)
    _ax.grid(axis="y", alpha=0.25)
    _fig.tight_layout()

    mo.vstack([gaussian_points_slider, _fig, mo.md(
      f"The weights are positive and sum to **{_weights.sum():.6f}**, "
      "which equals the exact integral of 1 over $[-1, 1]$."
    )])
    return (roots_legendre,)


@app.cell
def _(go, mo, np, roots_legendre):
    def _newton_cotes_weights(_nodes):
      _n = len(_nodes)
      _vandermonde = np.array([_nodes**degree for degree in range(_n)])
      _moments = np.array([(1.0 - (-1.0) ** (degree + 1)) / (degree + 1) for degree in range(_n)])
      return np.linalg.solve(_vandermonde, _moments)

    _frame_ns = list(range(2, 11))
    _initial_gaussian_nodes, _initial_gaussian_weights = roots_legendre(_frame_ns[0])
    _initial_newton_nodes = np.linspace(-1.0, 1.0, _frame_ns[0])
    _initial_newton_weights = _newton_cotes_weights(_initial_newton_nodes)
    _fig = go.Figure(
      data=[
        go.Scatter(x=_initial_gaussian_nodes, y=_initial_gaussian_weights, mode="lines+markers", name="Gaussian weights", line={"color": "#2878b5", "width": 2}),
        go.Scatter(x=_initial_newton_nodes, y=_initial_newton_weights, mode="lines+markers", name="Newton–Cotes weights", line={"color": "#d1495b", "width": 2}),
      ],
      layout=go.Layout(
        title=f"Weight comparison for N={_frame_ns[0]}",
        xaxis_title="sample point x",
        yaxis_title="weight",
        yaxis_zeroline=True,
        template="plotly_white",
        updatemenus=[{
          "type": "buttons",
          "showactive": False,
          "x": 1.0,
          "xanchor": "right",
          "y": 1.18,
          "yanchor": "top",
          "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 650, "redraw": True}, "fromcurrent": True}]},
            {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
          ],
        }],
        sliders=[{
          "active": 0,
          "currentvalue": {"prefix": "N = "},
          "steps": [{"label": str(_n), "method": "animate", "args": [[str(_n)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]} for _n in _frame_ns],
        }],
      ),
    )
    _fig.frames = [
      go.Frame(
        name=str(_n),
        data=[
          go.Scatter(x=roots_legendre(_n)[0], y=roots_legendre(_n)[1], mode="lines+markers", line={"color": "#2878b5", "width": 2}),
          go.Scatter(x=np.linspace(-1.0, 1.0, _n), y=_newton_cotes_weights(np.linspace(-1.0, 1.0, _n)), mode="lines+markers", line={"color": "#d1495b", "width": 2}),
        ],
        layout=go.Layout(title=f"Weight comparison for N={_n}"),
      )
      for _n in _frame_ns
    ]

    mo.vstack([
    _fig,
      mo.md(" ")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unevenly spaced samples can give very accurate answers with only a small number of points

    When the sample points $x_k$ are fixed at equal intervals (like in Newton–Cotes formulas), your only freedom is choosing the $N$ weights $w_k$. With $N$ adjustable weights ($N$ degrees of freedom), you can make the rule exact for polynomials up to degree $N - 1$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 5.2: Gaussian integral of a simple function

    Consider the integral

    $$
    \int_0^2 (x^4 - 2x + 1)\,dx.
    $$

    Its exact value is $4.4$. We will evaluate it with Gaussian quadrature using only $N=3$ sample points.

    Gaussian integration with $N$ points is exact for polynomials up to degree $2N-1$. For $N=3$, this means every polynomial up to degree five is integrated exactly. Since $x^4-2x+1$ has degree four, the numerical answer should be exact up to floating-point round-off.
    """)
    return


@app.cell
def _(mo, np, plt, roots_legendre):
    def _gaussxw(_n, _a=-1.0, _b=1.0):
      """Return Gaussian nodes and weights mapped to [_a, _b]."""
      _nodes, _weights = roots_legendre(_n)
      _mapped_nodes = 0.5 * (_b - _a) * _nodes + 0.5 * (_a + _b)
      _mapped_weights = 0.5 * (_b - _a) * _weights
      return _mapped_nodes, _mapped_weights


    def _f_example(_x):
      return _x**4 - 2.0 * _x + 1.0

    _N_example = 3
    _a_example = 0.0
    _b_example = 2.0
    _x_example, _w_example = _gaussxw(_N_example, _a_example, _b_example)
    _example_terms = _w_example * _f_example(_x_example)
    _gaussian_result = _example_terms.sum()
    _exact_result = 4.4
    _absolute_error = abs(_gaussian_result - _exact_result)

    _x_plot_example = np.linspace(_a_example, _b_example, 600)
    _y_plot_example = _f_example(_x_plot_example)
    _example_fig, _example_ax = plt.subplots(figsize=(8, 4.2))
    _example_ax.plot(_x_plot_example, _y_plot_example, color="#222222", linewidth=2.5, label=r"$f(x)=x^4-2x+1$")
    _example_ax.scatter(
      _x_example,
      _f_example(_x_example),
      color="#d1495b",
      edgecolor="white",
      s=75,
      zorder=3,
      label="Gaussian sample points",
    )
    _example_ax.vlines(_x_example, 0, _f_example(_x_example), color="#2878b5", linewidth=1.5, alpha=0.7)
    _example_ax.axhline(0, color="black", linewidth=0.8)
    _example_ax.set_title("Example 5.2: Gaussian quadrature with N=3")
    _example_ax.set_xlabel("x")
    _example_ax.set_ylabel("f(x)")
    _example_ax.grid(alpha=0.25)
    _example_ax.legend(frameon=False)
    _example_fig.tight_layout()

    _example_rows = "\n".join(
      f"| {_k + 1} | {_x_example[_k]:.6f} | {_w_example[_k]:.6f} | {_f_example(_x_example[_k]):.6f} | {_example_terms[_k]:.6f} |"
      for _k in range(_N_example)
    )

    mo.vstack([
      _example_fig,
      mo.md(
        f"""**Legendre polynomial and division**

    For the interval $[0, 2]$, use $x=t+1$, so $dx=dt$ and $t\in[-1,1]$. The third Legendre polynomial is

    $$P_3(t)=\\frac{{1}}{{2}}(5t^3-3t).$$

    The transformed integrand and its polynomial division are

    $$f(t+1)=t^4+4t^3+6t^2+2t
    =\\left(\\frac{{2}}{{5}}t+\\frac{{8}}{{5}}\\right)P_3(t)
    +\\left(\\frac{{33}}{{5}}t^2+\\frac{{34}}{{5}}t\\right).$$

    Thus, the quotient is $q(t)=\\frac{{2}}{{5}}t+\\frac{{8}}{{5}}$ and the remainder is $r(t)=\\frac{{33}}{{5}}t^2+\\frac{{34}}{{5}}t$.

    **Numerical Gaussian rule**

    | $k$ | point $x_k$ | weight $w_k$ | $f(x_k)$ | $w_k f(x_k)$ |
    |---:|---:|---:|---:|---:|
    {_example_rows}

    The weighted sum is $\\sum_k w_k f(x_k)={_gaussian_result:.16f}$, while integrating the remainder gives

    $$\\int_{{-1}}^1 r(t)\\,dt=\\frac{{33}}{{5}}\\frac{{2}}{{3}}=\\frac{{22}}{{5}}=4.4.$$

    The absolute error is ${_absolute_error:.2e}$, which is floating-point round-off."""
      ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Factor of improvement
    Adding one sample point, $N \to N+1$, can reduce the error by approximately

    $$\text{improvement factor} \approx \frac{c}{N^2}$$

    For example, moving from $N=10$ to $N=11$ improves the estimate by roughly a **factor of 100**.

    Doubling points ($N \to 2N$) reduces error by a factor of roughly $N^{-2N}$, making $N \le 100$ sufficient to reach machine precision on smooth functions.
    """)
    return


@app.cell
def _(np, plt):
    _N_values = np.arange(2, 31)
    _c = 10000.0
    _improvement = _c / _N_values**2

    _fig, _ax = plt.subplots(figsize=(8, 4.2))
    _ax.plot(_N_values, _improvement, color="#2878b5", linewidth=2.5, label=r"$c/N^2$")
    _ax.set_yscale("log")
    _ax.set_title("One more Gaussian point can greatly reduce the error")
    _ax.set_xlabel("number of points, N")
    _ax.set_ylabel("illustrative improvement factor")
    _ax.grid(alpha=0.25, which="both")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How accurate is the answer?

    ### A practical error check

    Run the same integral twice: once with $N$ points, then with $2N$ points.

    $$I_N = \text{estimate with } N \text{ points}$$
    $$I_{2N} = \text{refined estimate with } 2N \text{ points}$$

    Because the refined estimate is usually much more accurate,

    $$\boxed{\delta_N \approx I_{2N} - I_N}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example: integrating the Runge function

    The classic Runge function is

    $$f(x) = \frac{1}{1 + 25x^2}, \qquad x \in [-1, 1]$$

    It is smooth, but it changes rapidly near $x=0$ and is famous for causing large errors in **equally spaced polynomial interpolation**. Gaussian quadrature uses carefully chosen nodes instead, so we can test how quickly the integral converges:

    $$I = \int_{-1}^{1} \frac{1}{1 + 25x^2}\,dx = \frac{2}{5}\arctan(5)$$
    """)
    return


@app.cell
def _(mo):
    runge_points_slider = mo.ui.slider(1, 100, value=3, step=1, label="Number of Gaussian points N")
    return (runge_points_slider,)


@app.cell
def _(mo, np, plt, roots_legendre, runge_points_slider):
    def _runge_function(_x):
        return 1.0 / (1.0 + 25.0 * _x**2)


    def _runge_integral(_n):
        _nodes, _weights = roots_legendre(_n)
        return np.sum(_weights * _runge_function(_nodes))

    _runge_n = int(runge_points_slider.value)
    _runge_i_n = _runge_integral(_runge_n)
    _runge_i_2n = _runge_integral(2 * _runge_n)
    _runge_exact = 0.4 * np.arctan(5.0)
    _runge_estimated_error = abs(_runge_i_2n - _runge_i_n)
    _runge_true_error_n = abs(_runge_i_n - _runge_exact)
    _runge_true_error_2n = abs(_runge_i_2n - _runge_exact)
    _runge_estimate_error = abs(_runge_estimated_error - _runge_true_error_n)

    _runge_nodes_n, _runge_weights_n = roots_legendre(_runge_n)
    _runge_nodes_2n, _runge_weights_2n = roots_legendre(2 * _runge_n)
    _runge_x_plot = np.linspace(-1.0, 1.0, 600)
    _runge_y_plot = _runge_function(_runge_x_plot)

    _runge_fig, _runge_ax = plt.subplots(figsize=(8, 4.5))
    _runge_ax.plot(_runge_x_plot, _runge_y_plot, color="#222222", linewidth=2.5, label=r"Runge function $f(x)$")
    _runge_ax.scatter(
        _runge_nodes_n,
        _runge_function(_runge_nodes_n),
        color="#2878b5",
        edgecolor="white",
        s=55,
        zorder=3,
        label=fr"Gaussian nodes ($N={_runge_n}$)",
    )
    _runge_ax.scatter(
        _runge_nodes_2n,
        _runge_function(_runge_nodes_2n),
        color="#d1495b",
        edgecolor="white",
        s=42,
        zorder=3,
        label=fr"Gaussian nodes ($2N={2 * _runge_n}$)",
    )
    _runge_ax.set_title("Runge function and Gaussian sample points")
    _runge_ax.set_xlabel("x")
    _runge_ax.set_ylabel(r"$f(x)=1/(1+25x^2)$")
    _runge_ax.set_xlim(-1.0, 1.0)
    _runge_ax.grid(alpha=0.25)
    _runge_ax.legend(frameon=False)
    _runge_fig.tight_layout()

    mo.vstack([
        runge_points_slider,
        _runge_fig,
        mo.md(
            fr"""

    Exact value: **{_runge_exact:.12f}**

    | Calculation | Result | Error |
    | --- | ---: | ---: |
    | $I_N$ with $N={_runge_n}$ | **{_runge_i_n:.12f}** | **{_runge_true_error_n:.3e}** |
    | $I_{{2N}}$ with $2N={2 * _runge_n}$ | **{_runge_i_2n:.12f}** | **{_runge_true_error_2n:.3e}** |
    | Difference $\lvert I_{{2N}}-I_N\rvert$ | **{_runge_estimated_error:.3e}** | **{_runge_estimate_error:.3e}** |

    """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.7 - Choosing an integration method

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
    def simpsons_rule2(f, a, b, N):
        """Extended Simpson's rule with N slices, N even (Eq. 5.9)."""
        x = np.linspace(a, b, N + 1)
        y = f(x)
        h = (b - a) / N
        return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))

    return (simpsons_rule2,)


@app.cell
def _(trapezoidal_rule2):
    def romberg_integration(f, a, b, n_levels):
        """Romberg integration (5.4): builds a triangular table of increasingly
        accurate estimates from the trapezoidal rule, using Richardson
        extrapolation (Eq. 5.51), and returns the most accurate (bottom-right)
        entry after n_levels rows. This version recomputes the trapezoidal
        estimate at each row from scratch rather than reusing evaluations as in
        5.3 — less efficient, but easier to follow, and fine for the small
        n_levels used here."""
        R_prev = [trapezoidal_rule2(f, a, b, 1)]
        for i in range(1, n_levels):
            N = 2 ** i
            row = [trapezoidal_rule2(f, a, b, N)]
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


@app.cell
def _(mo):
    function_choice2 = mo.ui.radio(
        options=["smooth: exp(x)*cos(3x)", "rough: sqrt(|x - 0.3|)"],
        value="smooth: exp(x)*cos(3x)",
        label="pick a function to integrate on [0, 1]",
    )
    return (function_choice2,)


@app.cell
def _(function_choice2, np):
    if function_choice2.value.startswith("smooth"):
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
    simpsons_rule2,
    trapezoidal_rule2,
    true_value,
):
    benchmark_Ns = np.array([2, 4, 8, 16, 32, 64, 128])
    trap_err, simp_err, romb_err, gauss_err = [], [], [], []
    for _N in benchmark_Ns:
        trap_err.append(abs(trapezoidal_rule2(bench_f, 0, 1, int(_N)) - true_value))
        simp_err.append(abs(simpsons_rule2(bench_f, 0, 1, int(_N)) - true_value))
        _n_levels = int(np.log2(_N)) + 1
        romb_err.append(abs(romberg_integration(bench_f, 0, 1, _n_levels) - true_value))
        gauss_err.append(abs(gauss_quadrature(bench_f, 0, 1, int(_N)) - true_value))
    return benchmark_Ns, gauss_err, romb_err, simp_err, trap_err


@app.cell
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


@app.cell
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
    # Chapter 5.8 - Integrals Over Infinite Ranges



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

    return (semi_infinite_integral,)


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
    def f_gauss2(x):
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
        "Gaussian tail:  e^(−x²)": dict(f=f_gauss2, true=0.8862269254527579),
        "Planck / Stefan–Boltzmann:  x³/(eˣ − 1)": dict(f=f_planck, true=6.493939402266828),
        "Power-law decay:  1/(1+x)^1.5": dict(f=f_powerlaw, true=2.0),
    }
    return f_gauss2, infinite_range_functions


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
def _(mo, n_points_slider, np, plt):
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
def _(infinite_range_functions, mo, np, plt, semi_infinite_integral):
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
def _(mo):
    c_slider = mo.ui.slider(0.1, 8.0, value=1.0, step=0.1, label="c in z = x/(c + x)")
    n_fixed_slider = mo.ui.slider(3, 20, value=6, step=1, label="N (kept small on purpose)")
    return c_slider, n_fixed_slider


@app.cell
def _(c_slider, f_gauss2, mo, n_fixed_slider, semi_infinite_integral):
    _true = 0.8862269254527579
    _c = c_slider.value
    _N = n_fixed_slider.value
    _estimate = semi_infinite_integral(f_gauss2, _N, a=0.0, c=_c)
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
    return (full_line_functions,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.9 - Multiple integrals

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


@app.cell
def _(mo):
    grid_N = mo.ui.slider(3, 20, value=10, step=1, label="points per axis (N)")
    return (grid_N,)


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Useful links
    [MathTheBeautiful videos on Gaussian quadrature](https://www.youtube.com/watch?v=65zwMgGZnUs&list=PLlXfTHzgMRULZfrNCrrJ7xDcTjGr633mm&index=18)
    """)
    return


if __name__ == "__main__":
    app.run()
