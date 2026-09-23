import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    return Patch, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercises 6.1 & 6.4 — A circuit of resistors

    From chapter 6 of Newman's *Computational Physics*, following last week's
    lectures on linear algebra (Gaussian elimination)

    **Exercise 6.1** gives a circuit of resistors connecting four unknown
    node voltages $V_1$–$V_4$ to a $V_+ = 5\,$V supply and to ground, and asks
    us to (a) derive the linear equations for all four nodes from Kirchhoff's
    current law, and (b) solve them with Gaussian elimination.

    **Exercise 6.4** asks us to solve the same system again, this time using
    `numpy.linalg.solve`, and check that both approaches agree.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up the circuit

    ---

    Looking at the figure closely, the four junctions $V_1$–$V_4$ are **not**
    all connected to each other. $V_1$ connects to $V_2$, $V_3$, $V_4$, and the
    $V_+ = 5\,$V rail (four resistors). $V_4$ is symmetric to $V_1$: it
    connects to $V_1$, $V_2$, $V_3$, and the ground (four resistors). But $V_2$
    and $V_3$ are each connected to only *three* things: $V_2$ connects to
    $V_1$, $V_4$, and the ground; $V_3$ connects to $V_1$, $V_4$, and the $V_+$ rail. $V_2$
    and $V_3$ are **not** directly connected to each other — the only
    "diagonal" resistor in the picture runs from $V_1$ to $V_4$. That's 9 resistors in total. The circuit looks like this:
    """)
    return


@app.cell(hide_code=True)
def _(plt):
    _circuit_positions = {
        "V+": (0.5, 1.05),
        "V1": (0.12, 0.62),
        "V3": (0.88, 0.62),
        "V2": (0.12, 0.18),
        "V4": (0.88, 0.18),
        "GND": (0.5, -0.22),
    }
    _circuit_edges = [
        ("V+", "V1"), ("V+", "V3"),
        ("V1", "V2"), ("V1", "V3"), ("V1", "V4"),
        ("V2", "V4"),
        ("V3", "V4"),
        ("V2", "GND"), ("V4", "GND"),
    ]

    def draw_resistor_network(voltages=None, title="The resistor network"):
        """Draws the circuit from Exercise 6.1. Pass a dict mapping each node
        name to its voltage to label the solved values; leave it as None to
        just show the layout before solving anything."""
        fig, ax = plt.subplots(figsize=(6, 6.5))
        for n1, n2 in _circuit_edges:
            x1, y1 = _circuit_positions[n1]
            x2, y2 = _circuit_positions[n2]
            ax.plot([x1, x2], [y1, y2], color="0.4", lw=1.3, zorder=1)

        for name, (x, y) in _circuit_positions.items():
            color = "#f6c453" if name in ("V+", "GND") else "#cfe3fb"
            ax.scatter([x], [y], s=900, color=color, edgecolor="black", zorder=3)
            ax.annotate(name, (x, y), ha="center", va="center", fontsize=11, zorder=4)
            if voltages is not None:
                ax.annotate(f"{voltages[name]:.1f} V", (x, y - 0.12), ha="center", va="center",
                            fontsize=10, color="crimson", zorder=4)

        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.4, 1.25)
        ax.axis("off")
        ax.set_title(title)
        return fig

    return (draw_resistor_network,)


@app.cell(hide_code=True)
def _(draw_resistor_network, mo):
    mo.center(draw_resistor_network(title="The resistor network from Exercise 6.1 (before solving)"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Kirchhoff's current law says the net current flowing out of (or into) any junction
    must be zero, and Ohm's law says the current out of junction $i$ through a
    resistor to junction $j$ is $(V_i-V_j)/R$. The book works this out for
    $V_1$ already:

    $$
    \frac{V_1-V_2}{R} + \frac{V_1-V_3}{R} + \frac{V_1-V_4}{R} + \frac{V_1-V_+}{R} = 0
    \quad\Longleftrightarrow\quad
    4V_1 - V_2 - V_3 - V_4 = V_+.
    $$

    Part (a) asks for the same thing at the other three junctions, using
    the logic identified above. The equations for $V_2$, $V_3$, and $V_4$ are:

    $$
    \begin{aligned}
    \frac{V_2-V_1}{R} + \frac{V_2-V_4}{R} + \frac{V_2-0}{R} &= 0
    &&\Longrightarrow&& 3V_2 - V_1 - V_4 = 0, \\[4pt]
    \frac{V_3-V_+}{R} + \frac{V_3-V_1}{R} + \frac{V_3-V_4}{R} &= 0
    &&\Longrightarrow&& 3V_3 - V_1 - V_4 = V_+, \\[4pt]
    \frac{V_4-V_1}{R} + \frac{V_4-V_2}{R} + \frac{V_4-V_3}{R} + \frac{V_4-0}{R} &= 0
    &&\Longrightarrow&& 4V_4 - V_1 - V_2 - V_3 = 0.
    \end{aligned}
    $$

    The resistance $R$ cancels out of all four equations, leaving a system that only depends on $V_+$, which is given as 5 V in the exercise.
    The book also states that all resistors are equal (have the same resistance), which is why the $R$ cancels out. If they were different, the equations would have different coefficients for each term.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before this becomes a matrix, it helps to rewrite each equation so every
    term appears in the same order ($V_1,V_2,V_3,V_4$), filling in an
    explicit coefficient of 0 for any variable that doesn't actually show up
    in a given equation ($V_2$ and $V_3$ are missing from each other's
    equations, since they aren't directly connected):

    $$
    \begin{aligned}
    4V_1 &- 1V_2 - 1V_3 - 1V_4 &&= V_+ \\
    -1V_1 &+ 3V_2 + 0V_3 - 1V_4 &&= 0 \\
    -1V_1 &+ 0V_2 + 3V_3 - 1V_4 &&= V_+ \\
    -1V_1 &- 1V_2 - 1V_3 + 4V_4 &&= 0
    \end{aligned}
    $$

    In matrix form, $\mathbf A\mathbf x=\mathbf v$ with
    $\mathbf x=(V_1,V_2,V_3,V_4)$:

    $$
    \begin{pmatrix}4&-1&-1&-1\\-1&3&0&-1\\-1&0&3&-1\\-1&-1&-1&4\end{pmatrix}
    \begin{pmatrix}V_1\\V_2\\V_3\\V_4\end{pmatrix}
    =
    \begin{pmatrix}V_+\\0\\V_+\\0\end{pmatrix}.
    $$
    """)
    return


@app.cell
def _(np):
    Vplus = 5.0  # volts, given in the exercise

    A = np.array(
        [
            [4.0, -1.0, -1.0, -1.0],
            [-1.0, 3.0, 0.0, -1.0],
            [-1.0, 0.0, 3.0, -1.0],
            [-1.0, -1.0, -1.0, 4.0],
        ]
    )
    v = np.array([Vplus, 0.0, Vplus, 0.0])
    return A, Vplus, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part (b): solving with Gaussian elimination

    ---

    This is the elimination procedure from last week's lecture, with partial
    pivoting added: before eliminating column $k$, look at every remaining
    row (row $k$ and below) and swap in whichever one has the largest
    absolute value in column $k$, then use that as the pivot row. That
    guarantees we never divide by a zero (or needlessly small) pivot. It
    isn't actually necessary for this particular matrix — the diagonal
    entries stay comfortably large throughout — but it's the safer general-purpose approach, and
    it's what last week's lecture covered, so I used it here rather than the
    no-pivoting shortcut.
    """)
    return


@app.cell
def _(np):
    def gauss_elim_solve(A0, v0):
        """Gaussian elimination with partial pivoting."""
        A = A0.astype(float).copy()
        v = v0.astype(float).copy()
        N = len(v)
 
        for k in range(N):
            # partial pivoting: swap in the row (at or below k) with the
            # largest entry in column k, so we never divide by a small pivot
            pivot_row = k + np.argmax(np.abs(A[k:, k]))
            if pivot_row != k:
                A[[k, pivot_row]] = A[[pivot_row, k]]
                v[[k, pivot_row]] = v[[pivot_row, k]]
 
            pivot = A[k, k]
            A[k, :] /= pivot
            v[k] /= pivot
            for i in range(k + 1, N):
                mult = A[i, k]
                A[i, :] -= mult * A[k, :]
                v[i] -= mult * v[k]
 
        x = np.zeros(N)
        for k in range(N - 1, -1, -1):
            x[k] = v[k] - A[k, k + 1 :] @ x[k + 1 :]
        return x

    return (gauss_elim_solve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Elimination
    """)
    return


@app.cell(hide_code=True)
def _(A, v):
    def gaussian_trace(A0, v0):
        """Same partial-pivoting elimination as gauss_elim_solve, but records
        a snapshot of the augmented matrix after every individual operation
        (a row swap, a row normalization, or a single row's elimination)
        instead of only the final answer, so the process can be stepped
        through below."""
        Am = A0.astype(float).copy()
        vm = v0.astype(float).copy()
        N = len(vm)
        trace = [dict(A=Am.copy(), v=vm.copy(), step="initial", k=None, i=None, mult=None, div=None, swap=None)]
 
        for k in range(N):
            pivot_row = k + int(abs(Am[k:, k]).argmax())
            if pivot_row != k:
                Am[[k, pivot_row]] = Am[[pivot_row, k]]
                vm[[k, pivot_row]] = vm[[pivot_row, k]]
                trace.append(dict(A=Am.copy(), v=vm.copy(), step="swap", k=k, i=None, mult=None, div=None, swap=(k, pivot_row)))
 
            div = Am[k, k]
            Am[k, :] /= div
            vm[k] /= div
            trace.append(dict(A=Am.copy(), v=vm.copy(), step="normalize", k=k, i=None, mult=None, div=div, swap=None))
 
            for i in range(k + 1, N):
                mult = Am[i, k]
                Am[i, :] -= mult * Am[k, :]
                Am[i, k] = 0.0
                vm[i] -= mult * vm[k]
                trace.append(dict(A=Am.copy(), v=vm.copy(), step="eliminate", k=k, i=i, mult=mult, div=None, swap=None))
 
        return trace
 
    elimination_trace = gaussian_trace(A, v)
    return (elimination_trace,)


@app.cell(hide_code=True)
def _(elimination_trace, mo):
    ge_step_slider = mo.ui.slider(0, len(elimination_trace) - 1, value=0, step=1, label="Elimination step")
    return (ge_step_slider,)


@app.cell(hide_code=True)
def _(Patch, plt):
    def draw_gauss_step(step_idx, trace, N):
        state = trace[step_idx]
        A, v, step_type, k, i_row, swap = state["A"], state["v"], state["step"], state["k"], state["i"], state["swap"]
        cell = 1.0
        fig, ax = plt.subplots(figsize=(6.2, 5.8))
 
        for r in range(N):
            for c in range(N):
                x0, y0 = c * cell, (N - 1 - r) * cell
                facecolor, edgecolor, lw, textcolor = "white", "0.6", 1.0, "black"
                if step_type == "swap" and swap is not None and r in swap:
                    facecolor, edgecolor, lw = "#f6c453", "black", 2.0
                elif step_type == "normalize":
                    if r == k and c == k:
                        facecolor, edgecolor, lw = "#f6c453", "black", 2.2
                    elif r == k:
                        facecolor = "#fdf0d5"
                elif step_type == "eliminate":
                    if r == k:
                        facecolor = "#fdf0d5"
                    if r == i_row and c == k:
                        facecolor, edgecolor, lw = "#cfe3fb", "#1d4ed8", 2.0
                    elif r == i_row:
                        facecolor = "#fdecea"
                if k is not None and c < k and r > c:
                    textcolor = "0.6"
                ax.add_patch(plt.Rectangle((x0, y0), cell, cell, facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=2))
                ax.text(x0 + cell / 2, y0 + cell / 2, f"{A[r, c]:.3f}", ha="center", va="center", fontsize=12, color=textcolor, zorder=3)
            ax.text(-0.5, (N - 1 - r) * cell + cell / 2, f"$R_{{{r + 1}}}$", ha="center", va="center", fontsize=12)
 
        for c in range(N):
            ax.text(c * cell + cell / 2, N * cell + 0.15, f"$V_{{{c + 1}}}$", ha="center", va="bottom", fontsize=12)
        xg = N * cell + 0.35
        ax.plot([N * cell + 0.15, N * cell + 0.15], [0, N * cell], color="black", lw=1.3)
        ax.text(xg + cell / 2, N * cell + 0.15, "$v$", ha="center", va="bottom", fontsize=12)
 
        for r in range(N):
            y0 = (N - 1 - r) * cell
            facecolor = "white"
            if step_type == "swap" and swap is not None and r in swap:
                facecolor = "#f6c453"
            elif step_type == "normalize" and r == k:
                facecolor = "#fdf0d5"
            elif step_type == "eliminate" and r == k:
                facecolor = "#fdf0d5"
            elif step_type == "eliminate" and r == i_row:
                facecolor = "#fdecea"
            ax.add_patch(plt.Rectangle((xg, y0), cell, cell, facecolor=facecolor, edgecolor="0.6", lw=1.0, zorder=2))
            ax.text(xg + cell / 2, y0 + cell / 2, f"{v[r]:.3f}", ha="center", va="center", fontsize=12, zorder=3)
 
        legend_handles = [
            Patch(facecolor="#f6c453", edgecolor="black", label="Pivot element / swapped rows"),
            Patch(facecolor="#fdf0d5", edgecolor="0.6", label="Pivot row"),
            Patch(facecolor="#fdecea", edgecolor="0.6", label="Row being updated"),
            Patch(facecolor="#cfe3fb", edgecolor="#1d4ed8", label="Entry being eliminated"),
        ]
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.45, -0.05), ncol=1, frameon=False, fontsize=9)
        ax.set_xlim(-1.0, xg + cell + 0.2)
        ax.set_ylim(-0.2, N * cell + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig

    return (draw_gauss_step,)


@app.cell(hide_code=True)
def _(draw_gauss_step, elimination_trace, ge_step_slider, mo):
    _state = elimination_trace[ge_step_slider.value]
    _step, _k, _i, _mult, _div, _swap = _state["step"], _state["k"], _state["i"], _state["mult"], _state["div"], _state["swap"]
 
    if _step == "initial":
        _formula = mo.md("**Initial system.** The augmented matrix $[\\mathbf A\\,|\\,\\mathbf v]$ from Exercise 6.1, before any elimination has taken place.")
        _pivot_stat, _row_stat, _label, _value_stat = "—", "—", "Value", "—"
    elif _step == "swap":
        _r1, _r2 = _swap
        _formula = mo.md(
            rf"""
            **Step {ge_step_slider.value} of {len(elimination_trace) - 1}** —
            partial pivoting: row $R_{{{_r2 + 1}}}$ has the largest remaining
            entry in column {_k + 1}, so swap it into the pivot position:
            $$R_{{{_k + 1}}} \leftrightarrow R_{{{_r2 + 1}}}.$$
            """
        )
        _pivot_stat, _row_stat, _label, _value_stat = f"R{_k + 1}", f"R{_r2 + 1}", "Operation", "swap"
    elif _step == "normalize":
        _formula = mo.md(
            rf"""
            **Step {ge_step_slider.value} of {len(elimination_trace) - 1}** —
            normalize pivot row $R_{{{_k + 1}}}$ so its diagonal entry becomes 1:
            $$R_{{{_k + 1}}} \leftarrow \frac{{R_{{{_k + 1}}}}}{{a_{{{_k + 1}{_k + 1}}}}}, \qquad a_{{{_k + 1}{_k + 1}}} = {_div:.3f}$$
            """
        )
        _pivot_stat, _row_stat, _label, _value_stat = f"R{_k + 1}", "—", "Divisor a_kk", f"{_div:.3f}"
    else:
        _formula = mo.md(
            rf"""
            **Step {ge_step_slider.value} of {len(elimination_trace) - 1}** —
            eliminate $V_{{{_k + 1}}}$ from row $R_{{{_i + 1}}}$ using the
            (normalized) pivot row $R_{{{_k + 1}}}$:
            $$R_{{{_i + 1}}} \leftarrow R_{{{_i + 1}}} - a_{{{_i + 1}{_k + 1}}}\,R_{{{_k + 1}}}, \qquad a_{{{_i + 1}{_k + 1}}} = {_mult:.3f}$$
            """
        )
        _pivot_stat, _row_stat, _label, _value_stat = f"R{_k + 1}", f"R{_i + 1}", "Multiplier a_ik", f"{_mult:.3f}"
 
    if ge_step_slider.value == len(elimination_trace) - 1:
        _note = mo.md(
            "**Elimination complete.** The augmented matrix is now upper "
            "triangular with 1's on the diagonal — ready for back-substitution."
        ).callout(kind="success")
    else:
        _note = mo.md("")
 
    mo.vstack(
        [
            ge_step_slider,
            mo.hstack(
                [
                    draw_gauss_step(ge_step_slider.value, elimination_trace, 4),
                    mo.vstack(
                        [
                            _formula,
                            mo.hstack(
                                [
                                    mo.stat(label="Pivot row", value=_pivot_stat),
                                    mo.stat(label="Other row", value=_row_stat),
                                    mo.stat(label=_label, value=_value_stat),
                                ]
                            ),
                            _note,
                        ]
                    ),
                ],
                widths=[0.55, 0.45],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Back-substitution

    The last row already **is** the answer for $V_4$; substituting it into
    the row above gives $V_3$, then $V_2$, then $V_1$ — working from the
    bottom row upward.
    """)
    return


@app.cell(hide_code=True)
def _(elimination_trace):
    def backsub_trace(A_final, v_final):
        N = len(v_final)
        x = v_final.astype(float).copy()
        trace = [dict(k=None, coeffs=None, known=None, value=None, x=x.copy())]
        for k in range(N - 1, -1, -1):
            coeffs = A_final[k, k + 1 :]
            known = x[k + 1 :]
            value = v_final[k] - coeffs.dot(known)
            x[k] = value
            trace.append(dict(k=k, coeffs=coeffs.copy(), known=known.copy(), value=value, x=x.copy()))
        return trace
 
    _final_state = elimination_trace[-1]
    backsub_steps_trace = backsub_trace(_final_state["A"], _final_state["v"])
    return (backsub_steps_trace,)


@app.cell(hide_code=True)
def _(backsub_steps_trace, mo):
    backsub_slider = mo.ui.slider(0, len(backsub_steps_trace) - 1, value=0, step=1, label="Back-substitution step")
    return (backsub_slider,)


@app.cell(hide_code=True)
def _(plt):
    def draw_backsub_vector(step_idx, trace, N):
        state = trace[step_idx]
        k, x = state["k"], state["x"]
        cell = 1.0
        fig, ax = plt.subplots(figsize=(2.2, 4.6))
        for r in range(N):
            y0 = (N - 1 - r) * cell
            solved = (k is not None) and (r >= k)
            if k is not None and r == k:
                facecolor, edgecolor, lw = "#f6c453", "black", 2.2
            elif solved:
                facecolor, edgecolor, lw = "#d9f2df", "0.6", 1.0
            else:
                facecolor, edgecolor, lw = "white", "0.6", 1.0
            ax.add_patch(plt.Rectangle((0, y0), cell, cell, facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=2))
            label = f"{x[r]:.3f}" if solved else "?"
            ax.text(cell / 2, y0 + cell / 2, label, ha="center", va="center", fontsize=12, zorder=3)
            ax.text(-0.4, y0 + cell / 2, f"$V_{{{r + 1}}}$", ha="center", va="center", fontsize=12)
        ax.set_xlim(-1.0, cell + 0.2)
        ax.set_ylim(-0.2, N * cell + 0.2)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig

    return (draw_backsub_vector,)


@app.cell(hide_code=True)
def _(
    A,
    backsub_slider,
    backsub_steps_trace,
    draw_backsub_vector,
    elimination_trace,
    mo,
    np,
    v,
):
    _state = backsub_steps_trace[backsub_slider.value]
    _k, _coeffs, _known, _value = _state["k"], _state["coeffs"], _state["known"], _state["value"]
    _v_tilde = elimination_trace[-1]["v"]
 
    if _k is None:
        _formula = mo.md(
            "**Nothing solved yet.** The last equation of the triangular "
            "system, $V_4=\\tilde v_4$, already gives an unknown directly — "
            "it's solved first."
        )
    elif len(_coeffs) == 0:
        _formula = mo.md(
            rf"""
            **Step {backsub_slider.value} of {len(backsub_steps_trace) - 1}**
            — solve for $V_{{{_k + 1}}}$:
            $$V_{{{_k + 1}}} = \tilde v_{{{_k + 1}}} = {_value:.3f}$$
            """
        )
    else:
        _terms = " + ".join(rf"({c:.3f})({xj:.3f})" for c, xj in zip(_coeffs, _known))
        _formula = mo.md(
            rf"""
            **Step {backsub_slider.value} of {len(backsub_steps_trace) - 1}**
            — solve for $V_{{{_k + 1}}}$ using the already-known values:
            $$V_{{{_k + 1}}} = \tilde v_{{{_k + 1}}} - \sum_{{j={_k + 2}}}^{{4}} \tilde a_{{{_k + 1},j}}V_j
            = {_v_tilde[_k]:.3f} - \left[{_terms}\right] = {_value:.3f}$$
            """
        )
 
    if backsub_slider.value == len(backsub_steps_trace) - 1:
        _x_final = backsub_steps_trace[-1]["x"]
        _residual = np.max(np.abs(A.dot(_x_final) - v))
        _note = mo.md(
            f"**All four voltages recovered.** Checking against the "
            f"*original* (pre-elimination) system: "
            f"$\\max|\\mathbf A\\mathbf x-\\mathbf v| = {_residual:.2e}$."
        ).callout(kind="success")
    else:
        _note = mo.md("")
 
    mo.vstack(
        [
            backsub_slider,
            mo.hstack(
                [
                    draw_backsub_vector(backsub_slider.value, backsub_steps_trace, 4),
                    mo.vstack([_formula, _note]),
                ],
                widths=[0.25, 0.75],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(A, gauss_elim_solve, mo, v):
    x_gauss = gauss_elim_solve(A, v)
    mo.md(
        rf"""
        Running this on our system gives

        $$
        V_1 = {x_gauss[0]:.4f}\ \text{{V}}, \quad
        V_2 = \tfrac{{5}}{{3}} = {x_gauss[1]:.4f}\ \text{{V}}, \quad
        V_3 = \tfrac{{10}}{{3}} = {x_gauss[2]:.4f}\ \text{{V}}, \quad
        V_4 = {x_gauss[3]:.4f}\ \text{{V}}.
        $$

        Note that $V_1\ne V_3$ and $V_2\ne V_4$ here — which makes
        sense given the circuit is asymmetric: $V_1$ and $V_4$ each have four
        resistors attached, while $V_2$ and $V_3$ only have three.
        """
    )
    return


@app.cell(hide_code=True)
def _(A, Vplus, gauss_elim_solve, mo, v):
    _V1, _V2, _V3, _V4 = gauss_elim_solve(A, v)
 
    _eq1_lhs = 4 * _V1 - _V2 - _V3 - _V4
    _eq2_lhs = 3 * _V2 - _V1 - _V4
    _eq3_lhs = 3 * _V3 - _V1 - _V4
    _eq4_lhs = 4 * _V4 - _V1 - _V2 - _V3
 
    mo.md(
        rf"""
        ### Manual check: plugging the solution back into the original equations

        Rather than only checking the matrix form, it's worth substituting the
        solved voltages back into the four Kirchhoff equations from part (a)
        directly:

        | junction | equation | left-hand side, with our computed numbers | right-hand side |
        |---|---|---|---|
        | $V_1$ | $4V_1-V_2-V_3-V_4=V_+$ | ${_eq1_lhs:.6f}$ | ${Vplus:.6f}$ |
        | $V_2$ | $3V_2-V_1-V_4=0$ | ${_eq2_lhs:.6f}$ | $0.000000$ |
        | $V_3$ | $3V_3-V_1-V_4=V_+$ | ${_eq3_lhs:.6f}$ | ${Vplus:.6f}$ |
        | $V_4$ | $4V_4-V_1-V_2-V_3=0$ | ${_eq4_lhs:.6f}$ | $0.000000$ |

        Every left-hand side matches its right-hand side (up to floating-point
        rounding), so the solved voltages satisfy the actual physical equations
        derived from Kirchhoff's current law.
        """
    ).callout(kind="success")
    return


@app.cell(hide_code=True)
def _(A, draw_resistor_network, gauss_elim_solve, mo, v):
    _voltages = dict(zip(["V1", "V2", "V3", "V4"], gauss_elim_solve(A, v)))
    _voltages["V+"] = 5.0
    _voltages["GND"] = 0.0
    mo.center(draw_resistor_network(voltages=_voltages, title="The resistor network, with the solved node voltages"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 6.4: Solving the same system with `numpy.linalg.solve`

    ---

    Exercise 6.4 asks for the same resistor network, but solved with
    `numpy.linalg.solve` instead of a hand-written elimination routine, and to
    check that the answer matches Exercise 6.1.
    """)
    return


@app.cell(hide_code=True)
def _(A, np, v):
    x_numpy = np.linalg.solve(A, v)
    return (x_numpy,)


@app.cell(hide_code=True)
def _(A, gauss_elim_solve, mo, np, v, x_numpy):
    _x_gauss = gauss_elim_solve(A, v)
    _agree = np.allclose(_x_gauss, x_numpy)
 
    mo.md(
        rf"""
        | | $V_1$ | $V_2$ | $V_3$ | $V_4$ |
        |---|---|---|---|---|
        | Gaussian elimination function | {_x_gauss[0]:.6f} | {_x_gauss[1]:.6f} | {_x_gauss[2]:.6f} | {_x_gauss[3]:.6f} |
        | NumPy function | {x_numpy[0]:.6f} | {x_numpy[1]:.6f} | {x_numpy[2]:.6f} | {x_numpy[3]:.6f} |

        Here, I have also used the `numpy.allclose` function
        which is comparing the solution from my own "hand-written" 
        `gauss_elim_solve` against `numpy.linalg.solve`'s answer. 
        Both should mathematically give exactly $V_1=3$, $V_2=\tfrac{{5}}{{3}}$, $V_3=\tfrac{{10}}{{3}}$,
        and $V_4=2$, but they can differ by something like $1e-15$ due to rounding 
        — nowhere near a real disagreement, just floating-point noise. 
        `np.allclose` returning **{_agree}** says "these two independent methods agree". 
        If we used `_x_gauss == x_numpy` instead, we'd risk getting **False** on some entries 
        purely from that last-bit rounding noise, even though the two methods are actually correct and in agreement.
        """
    )
    return


if __name__ == "__main__":
    app.run()
