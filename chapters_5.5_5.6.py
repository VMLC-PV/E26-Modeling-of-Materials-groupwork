import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Polynomial fitting
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
def _():
    import marimo as _mo

    n_slider_nc = _mo.ui.slider(2, 100, value=8, step=1, label="Number of equally spaced points N")
    return (n_slider_nc,)


@app.cell
def _(n_slider_nc):
    import marimo as _mo
    import matplotlib.pyplot as _plt
    import numpy as _np

    _n = int(n_slider_nc.value)
    _nodes = _np.linspace(-1.0, 1.0, _n)
    _vandermonde = _np.array([_nodes**degree for degree in range(_n)])
    _moments = _np.array([(1.0 - (-1.0) ** (degree + 1)) / (degree + 1) for degree in range(_n)])
    _weights = _np.linalg.solve(_vandermonde, _moments)

    _fig, _ax_weights = _plt.subplots(figsize=(8, 4.5))
    _ax_weights.axhline(0, color="black", linewidth=0.8)
    _ax_weights.plot(_np.arange(1, _n + 1), _weights, color="#2878b5", marker="o", linewidth=2, label="weights")
    _ax_weights.scatter(_np.arange(1, _n + 1), _weights, c=_np.where(_weights >= 0, "#2878b5", "#d1495b"), edgecolor="white", zorder=3)
    _ax_weights.set_title(f"Newton–Cotes weights, N={_n}")
    _ax_weights.set_xlabel("sample point index")
    _ax_weights.set_ylabel("weight")
    _ax_weights.grid(axis="y", alpha=0.25)
    _fig.tight_layout()

    _mo.vstack([n_slider_nc, _fig, _mo.md(
      f"The largest absolute weight is **{_np.max(_np.abs(_weights)):.3g}**. "
      "Large alternating weights amplify round-off and data noise."
    )])
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
def _():
    import marimo as _mo

    degree_slider = _mo.ui.slider(1, 8, value=4, step=1, label="Highest degree")
    return (degree_slider,)


@app.cell
def _(degree_slider):
    import marimo as _mo
    import matplotlib.pyplot as _plt
    import numpy as _np

    _max_degree = int(degree_slider.value)
    _x = _np.linspace(-1.0, 1.0, 600)
    _fig, _ax = _plt.subplots(figsize=(9, 4.5))
    _colors = _plt.cm.viridis(_np.linspace(0.1, 0.9, _max_degree + 1))
    for _degree in range(_max_degree + 1):
      _coefficients = _np.zeros(_degree + 1)
      _coefficients[-1] = 1.0
      _values = _np.polynomial.legendre.legval(_x, _coefficients)
      _ax.plot(_x, _values, color=_colors[_degree], label=f"P$_{{{_degree}}}$(x)")
      if _degree == _max_degree and _degree > 0:
        _roots = _np.polynomial.legendre.legroots(_coefficients)
        _ax.scatter(_roots, _np.zeros_like(_roots), color="#d1495b", edgecolor="white", zorder=4, label="roots")
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.axvline(0, color="0.75", linewidth=0.8)
    _ax.set_title("Legendre polynomials and the roots used by Gaussian quadrature")
    _ax.set_xlabel("x")
    _ax.set_ylabel("P$_n$(x)")
    _ax.set_xlim(-1, 1)
    _ax.grid(alpha=0.2)
    _ax.legend(ncol=3, frameon=False, fontsize=9)
    _fig.tight_layout()

    _mo.vstack([degree_slider, _fig, _mo.md(
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
def _():
    import marimo as _mo

    gaussian_points_slider = _mo.ui.slider(2, 100, value=6, step=1, label="Number of Gaussian points N")
    return (gaussian_points_slider,)


@app.cell
def _(gaussian_points_slider):
    import marimo as _mo
    import matplotlib.pyplot as _plt
    from scipy.special import roots_legendre as _roots_legendre

    _n = int(gaussian_points_slider.value)
    _nodes, _weights = _roots_legendre(_n)
    _fig, _ax = _plt.subplots(figsize=(9, 4.2))
    _markerline, _stemlines, _baseline = _ax.stem(_nodes, _weights, linefmt="#2878b5", markerfmt="o", basefmt="k-")
    _plt.setp(_stemlines, linewidth=1.8)
    _plt.setp(_markerline, markersize=6, markerfacecolor="#d1495b", markeredgecolor="white")
    _ax.set_title(f"Gaussian quadrature sample points and weights, N={_n}")
    _ax.set_xlabel("sample point $x_k$ (roots of $P_N$)")
    _ax.set_ylabel("weight $w_k$")
    _ax.set_xlim(-1.05, 1.05)
    _ax.grid(axis="y", alpha=0.25)
    _fig.tight_layout()

    _mo.vstack([gaussian_points_slider, _fig, _mo.md(
      f"The weights are positive and sum to **{_weights.sum():.6f}**, "
      "which equals the exact integral of 1 over $[-1, 1]$."
    )])
    return


@app.cell
def _():
    import marimo as _mo
    import numpy as _np
    import plotly.graph_objects as _go
    from scipy.special import roots_legendre as _roots_legendre

    def _newton_cotes_weights(_nodes):
      _n = len(_nodes)
      _vandermonde = _np.array([_nodes**degree for degree in range(_n)])
      _moments = _np.array([(1.0 - (-1.0) ** (degree + 1)) / (degree + 1) for degree in range(_n)])
      return _np.linalg.solve(_vandermonde, _moments)

    _frame_ns = list(range(2, 11))
    _initial_gaussian_nodes, _initial_gaussian_weights = _roots_legendre(_frame_ns[0])
    _initial_newton_nodes = _np.linspace(-1.0, 1.0, _frame_ns[0])
    _initial_newton_weights = _newton_cotes_weights(_initial_newton_nodes)
    _fig = _go.Figure(
      data=[
        _go.Scatter(x=_initial_gaussian_nodes, y=_initial_gaussian_weights, mode="lines+markers", name="Gaussian weights", line={"color": "#2878b5", "width": 2}),
        _go.Scatter(x=_initial_newton_nodes, y=_initial_newton_weights, mode="lines+markers", name="Newton–Cotes weights", line={"color": "#d1495b", "width": 2}),
      ],
      layout=_go.Layout(
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
      _go.Frame(
        name=str(_n),
        data=[
          _go.Scatter(x=_roots_legendre(_n)[0], y=_roots_legendre(_n)[1], mode="lines+markers", line={"color": "#2878b5", "width": 2}),
          _go.Scatter(x=_np.linspace(-1.0, 1.0, _n), y=_newton_cotes_weights(_np.linspace(-1.0, 1.0, _n)), mode="lines+markers", line={"color": "#d1495b", "width": 2}),
        ],
        layout=_go.Layout(title=f"Weight comparison for N={_n}"),
      )
      for _n in _frame_ns
    ]

    _mo.vstack([
      _fig,
      _mo.md(" ")
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
def _():
    import marimo as _mo
    import matplotlib.pyplot as _plt
    import numpy as _np
    from scipy.special import roots_legendre as _roots_legendre_example


    def _gaussxw(_n, _a=-1.0, _b=1.0):
      """Return Gaussian nodes and weights mapped to [_a, _b]."""
      _nodes, _weights = _roots_legendre_example(_n)
      _mapped_nodes = 0.5 * (_b - _a) * _nodes + 0.5 * (_a + _b)
      _mapped_weights = 0.5 * (_b - _a) * _weights
      return _mapped_nodes, _mapped_weights


    def _f_example(_x):
      return _x**4 - 2.0 * _x + 1.0

    _N_example = 3
    _a_example = 0.0
    _b_example = 2.0
    _x_example, _w_example = _gaussxw(_N_example, _a_example, _b_example)
    _gaussian_result = sum(_w_example[_k] * _f_example(_x_example[_k]) for _k in range(_N_example))
    _exact_result = 4.4
    _absolute_error = abs(_gaussian_result - _exact_result)

    _x_plot_example = _np.linspace(_a_example, _b_example, 600)
    _y_plot_example = _f_example(_x_plot_example)
    _example_fig, _example_ax = _plt.subplots(figsize=(8, 4.2))
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

    _mo.vstack([
      _example_fig,
      _mo.md(
        f"**Answer:** With $N={_N_example}$, the Gaussian sample points are "
        f"${_np.round(_x_example, 6).tolist()}$ and the weights are "
        f"${_np.round(_w_example, 6).tolist()}$."
        f" The weighted sum gives **{_gaussian_result:.16f}**, compared with the exact "
        f"value **{_exact_result:.1f}**. The absolute error is "
        f"${_absolute_error:.2e}$, which is floating-point round-off."
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
def _():
    import matplotlib.pyplot as _plt
    import numpy as _np

    _N_values = _np.arange(2, 31)
    _c = 10000.0
    _improvement = _c / _N_values**2

    _fig, _ax = _plt.subplots(figsize=(8, 4.2))
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
    # How accurate is the answer?

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
def _():
    import marimo as _mo

    runge_points_slider = _mo.ui.slider(1, 100, value=3, step=1, label="Number of Gaussian points N")
    return (runge_points_slider,)


@app.cell
def _(runge_points_slider):
    import marimo as _mo
    import matplotlib.pyplot as _plt
    import numpy as _np
    from scipy.special import roots_legendre as _roots_legendre


    def _runge_function(_x):
        return 1.0 / (1.0 + 25.0 * _x**2)


    def _runge_integral(_n):
        _nodes, _weights = _roots_legendre(_n)
        return _np.sum(_weights * _runge_function(_nodes))

    _runge_n = int(runge_points_slider.value)
    _runge_i_n = _runge_integral(_runge_n)
    _runge_i_2n = _runge_integral(2 * _runge_n)
    _runge_exact = 0.4 * _np.arctan(5.0)
    _runge_estimated_error = abs(_runge_i_2n - _runge_i_n)
    _runge_true_error_n = abs(_runge_i_n - _runge_exact)
    _runge_true_error_2n = abs(_runge_i_2n - _runge_exact)
    _runge_estimate_error = abs(_runge_estimated_error - _runge_true_error_n)

    _runge_nodes_n, _runge_weights_n = _roots_legendre(_runge_n)
    _runge_nodes_2n, _runge_weights_2n = _roots_legendre(2 * _runge_n)
    _runge_x_plot = _np.linspace(-1.0, 1.0, 600)
    _runge_y_plot = _runge_function(_runge_x_plot)

    _runge_fig, _runge_ax = _plt.subplots(figsize=(8, 4.5))
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

    _mo.vstack([
        runge_points_slider,
        _runge_fig,
        _mo.md(
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
    # Useful link
    [MathTheBeautiful videos on Gaussian quadrature](https://www.youtube.com/watch?v=65zwMgGZnUs&list=PLlXfTHzgMRULZfrNCrrJ7xDcTjGr633mm&index=18)
    """)
    return


if __name__ == "__main__":
    app.run()
