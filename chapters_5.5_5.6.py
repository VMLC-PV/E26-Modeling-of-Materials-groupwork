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
    n_slider_nc
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

    _mo.vstack([_fig, _mo.md(
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
    degree_slider
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

    _mo.vstack([_fig, _mo.md(
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
    * Setting $P_N(x_k) = 0$ eliminates $q(x_k) P_N(x_k)$ in the sum:
      $$\sum_{k=1}^N w_k f(x_k) = \sum_{k=1}^N w_k r(x_k)$$
    * Because $r(x)$ has degree $\le N-1$, $\sum_{k=1}^N w_k r(x_k) = \int_{-1}^{1} r(x) \, dx$ **exactly**.
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
    gaussian_points_slider

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

    _mo.vstack([_fig, _mo.md(
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
      _mo.md("Press **Play** or move the slider. Gaussian weights remain positive and well behaved while equally spaced Newton–Cotes weights begin to oscillate.")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unevenly spaced samples can give very accurate answers with only a small number of points

    **Equally Spaced Points ($N - 1$ Degree Exactness)**: When the sample points $x_k$ are fixed at equal intervals (like in Newton–Cotes formulas), your only freedom is choosing the $N$ weights $w_k$. With $N$ adjustable weights ($N$ degrees of freedom), you can make the rule exact for polynomials up to degree $N - 1$.

    **Adding $N$ Degrees of Freedom**: If you allow the positions of the sample points $x_k$ to move anywhere instead of keeping them fixed, you gain another $N$ adjustable parameters.

    **Doubling the Accuracy ($2N - 1$ Degree Exactness)**: Combining $N$ weights and $N$ sample positions gives you $2N$ total degrees of freedom. By choosing both the weights and positions optimally, you can double the polynomial degree that the method can integrate perfectly—making it exact for polynomials up to degree $2N - 1$ using just $N$ points.
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

    Its exact value is $4.4$. We will evaluate it with Gaussian quadrature using
    only $N=3$ sample points.

    Gaussian integration with $N$ points is exact for polynomials up to degree
    $2N-1$. For $N=3$, this means every polynomial up to degree five is integrated
    exactly. Since $x^4-2x+1$ has degree four, the numerical answer should be
    exact up to floating-point round-off.

    The textbook implementation uses `gaussxw(N, a, b)` from `gaussxw.py`. The
    cell below performs the same interval transformation directly with SciPy's
    Legendre nodes and weights.
    """)
    return


@app.cell
def _():
    import marimo as _mo
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

    _mo.md(
      f"**Answer:** With $N={_N_example}$, the Gaussian sample points are "
      f"`${_np.round(_x_example, 6).tolist()}`$ and the weights are "
      f"`${_np.round(_w_example, 6).tolist()}`$.\\n\\n"
      f"The weighted sum gives **{_gaussian_result:.16f}**, compared with the exact "
      f"value **{_exact_result:.1f}**. The absolute error is "
      f"${_absolute_error:.2e}$, which is floating-point round-off."
    )
    return


if __name__ == "__main__":
    app.run()
