import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import time
    from numpy.linalg import solve
    from scipy.linalg import solve_banded

    return mo, np, solve, solve_banded, time


@app.cell
def _(mo):
    mo.md(r"""
    # Exercise 6.7 — A chain of resistors

    A long chain of $N$ internal junctions $V_1,\dots,V_N$ sits between a
    rail at $V_+$ and a rail at $0$ V. Every junction is connected by an
    equal resistor $R$ to the junctions two positions away on either side
    ($V_{i-2}, V_{i-1}, V_{i+1}, V_{i+2}$); where a neighbour would fall
    outside the chain it is instead wired to the nearest rail.

    Applying Kirchhoff's current law at each junction gives a
    **pentadiagonal** system $A\mathbf v = \mathbf w$:

    $$
    \begin{aligned}
    3V_1 - V_2 - V_3 &= V_+ \\
    -V_1 + 4V_2 - V_3 - V_4 &= V_+ \\
    -V_{i-2}-V_{i-1}+4V_i-V_{i+1}-V_{i+2} &= 0 \quad (3\le i\le N-2)\\
    -V_{N-3}-V_{N-2}+4V_{N-1}-V_N &= 0 \\
    -V_{N-2}-V_{N-1}+3V_N &= 0
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(np):
    def build_system(N, Vplus=5.0):
        """Dense pentadiagonal matrix A and vector w for the chain of N junctions."""
        A = np.zeros((N, N))
        w = np.zeros(N)

        A[0, 0] = 3
        A[0, 1] = -1
        A[0, 2] = -1
        w[0] = Vplus

        A[1, 0] = -1
        A[1, 1] = 4
        A[1, 2] = -1
        A[1, 3] = -1
        w[1] = Vplus

        for i in range(2, N - 2):
            A[i, i - 2] = -1
            A[i, i - 1] = -1
            A[i, i] = 4
            A[i, i + 1] = -1
            A[i, i + 2] = -1

        A[N - 2, N - 4] = -1
        A[N - 2, N - 3] = -1
        A[N - 2, N - 2] = 4
        A[N - 2, N - 1] = -1

        A[N - 1, N - 3] = -1
        A[N - 1, N - 2] = -1
        A[N - 1, N - 1] = 3

        return A, w

    return (build_system,)


@app.cell
def _(np):
    def build_banded(N, Vplus=5.0):
        """Banded storage (5, N) for the same pentadiagonal system, for use
        with scipy.linalg.solve_banded((2, 2), ab, w)."""
        ab = np.zeros((5, N))
        ab[2, :] = 4.0
        ab[2, 0] = 3.0
        ab[2, -1] = 3.0
        ab[1, 1:] = -1.0     # offset +1
        ab[0, 2:] = -1.0     # offset +2
        ab[3, :-1] = -1.0    # offset -1
        ab[4, :-2] = -1.0    # offset -2

        w = np.zeros(N)
        w[0] = Vplus
        w[1] = Vplus

        return ab, w

    return (build_banded,)


@app.cell
def _(mo):
    mo.md("""
    ## Part (b): direct (dense) solve for small $N$
    """)
    return


@app.cell
def _(mo):
    N_slider = mo.ui.slider(start=4, stop=60, value=6, step=1, label="N (junctions)")
    vplus_input = mo.ui.number(value=5.0, label="V+ (volts)")
    mo.hstack([N_slider, vplus_input])
    return N_slider, vplus_input


@app.cell
def _(N_slider, build_system, mo, solve, vplus_input):
    N = N_slider.value
    Vplus = vplus_input.value
    A, w = build_system(N, Vplus)
    V = solve(A, w)

    table = mo.ui.table(
        data=[
            {"junction": f"V{i + 1}", "voltage (V)": round(float(v), 4)}
            for i, v in enumerate(V)
        ],
        label=f"Junction voltages for N = {N}, V+ = {Vplus} V",
    )
    table
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part (c): banded solve for a long chain ($N = 10{,}000$)
    """)
    return


@app.cell
def _(build_banded, mo, solve_banded, time):
    N_big = 10_000
    ab, w_big = build_banded(N_big)

    t0 = time.time()
    V_big = solve_banded((2, 2), ab, w_big)
    elapsed = time.time() - t0

    mo.md(
        f"""
        Solved the **{N_big:,}-junction** pentadiagonal system in
        **{elapsed * 1000:.2f} ms** using `scipy.linalg.solve_banded`
    

        - $V_1 \\approx$ {V_big[0]:.5f} V
        - $V_N \\approx$ {V_big[-1]:.5f} V
        """
    )
    return (V_big,)


@app.cell
def _(V_big):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(V_big, lw=0.8)
    ax.set_xlabel("Junction index $i$")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage profile along the $N = 10{,}000$ resistor chain")
    fig
    return


if __name__ == "__main__":
    app.run()
