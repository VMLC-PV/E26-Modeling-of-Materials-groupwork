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
    # Romberg Integration



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


@app.cell(hide_code=True)
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


if __name__ == "__main__":
    app.run()
