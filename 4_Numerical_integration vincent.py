import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


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
    return mo, np, pd, plt


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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


if __name__ == "__main__":
    app.run()
