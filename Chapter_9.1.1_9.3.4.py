import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import time
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrowPatch
    from scipy.fft import dst
    from scipy.integrate import solve_ivp
    from scipy.linalg import eigh, eigh_tridiagonal

    return (
        FancyArrowPatch,
        Rectangle,
        dst,
        eigh,
        eigh_tridiagonal,
        mo,
        np,
        plt,
        solve_ivp,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 9 — Partial differential equations

    Some of the most demanding problems in computational physics are partial
    differential equations (PDEs): the wave equation, the diffusion equation,
    Maxwell's equations, the Schrödinger equation. Solving them numerically is
    expensive, so a lot of the work in this chapter is about getting answers
    *efficiently*.

    | Section | Topic | Main idea |
    |---|---|---|
    | 9.1 | Boundary value problems | replace the PDE by a large set of simultaneous equations on a grid |
    | 9.2 | Faster methods | overrelaxation, Gauss–Seidel |
    | 9.3 | Initial value problems | FTCS, implicit and Crank–Nicolson methods, stability |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chapter 9: what is the problem?

    ---

    An *ordinary* differential equation has one independent variable,
    usually time. A *partial* differential equation has two or more, for example
    position $x$ and time $t$, or positions $x, y, z$. The unknown is now a **field**
    such as $\phi(x,y,z)$ or $\phi(x,t)$ rather than a single number or a short vector.

    As with ODEs, PDE problems come in two broad classes:

    - **Boundary value problems.** The PDE describes a variable in a region of space,
      and the variable is given (or constrained) on the *boundary* of the region.
      The goal is to find the values in the interior. Classic example: the electrostatic
      potential inside a box whose walls are held at fixed voltages, which obeys
      Laplace's equation $\nabla^2\phi=0$.
    - **Initial value problems.** The field also changes in time. It is given at
      $t=0$ (the *initial condition*), together with boundary conditions, and the goal
      is to find how it evolves afterwards. Classic example: a plucked or struck
      string, described by the wave equation.

    In contrast with Chapter 8, **boundary value problems for PDEs are usually
    simpler to solve than initial value problems.** So we start with those,
    then move on to initial value problems.
    """)
    return


@app.cell(hide_code=True)
def _(FancyArrowPatch, Rectangle, mo, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Boundary value problem: box with V on top and 0 elsewhere (Fig. 9.1 in spirit)
    _ax1.add_patch(Rectangle((0, 0), 1, 1, fc="#e8eefc", ec="#1d4ed8", lw=0))
    _ax1.plot([0, 1], [1, 1], color="#c2410c", lw=5, solid_capstyle="butt")
    for _xs, _ys in (([0, 1], [0, 0]), ([0, 0], [0, 1]), ([1, 1], [0, 1])):
        _ax1.plot(_xs, _ys, color="#1d4ed8", lw=5, solid_capstyle="butt")
    _ax1.text(0.5, 1.05, r"$\phi = V$", ha="center", color="#c2410c", fontsize=13)
    _ax1.text(0.5, -0.1, r"$\phi = 0$", ha="center", color="#1d4ed8", fontsize=13)
    _ax1.text(-0.07, 0.5, r"$\phi = 0$", ha="right", va="center", color="#1d4ed8", fontsize=13)
    _ax1.text(1.07, 0.5, r"$\phi = 0$", ha="left", va="center", color="#1d4ed8", fontsize=13)
    _ax1.text(0.5, 0.5, r"$\nabla^2\phi = 0$" + "\nunknown inside", ha="center", va="center", fontsize=14)
    _ax1.set_xlim(-0.4, 1.4)
    _ax1.set_ylim(-0.25, 1.25)
    _ax1.set_aspect("equal")
    _ax1.axis("off")
    _ax1.set_title("Boundary value problem: the field is given on the boundary", fontsize=11)

    # Initial value problem: space-time diagram
    _ax2.add_patch(Rectangle((0, 0), 1, 1, fc="#fff3e8", ec="none"))
    _ax2.plot([0, 1], [0, 0], color="#c2410c", lw=5, solid_capstyle="butt")
    _ax2.plot([0, 0], [0, 1], color="#1d4ed8", lw=5, solid_capstyle="butt")
    _ax2.plot([1, 1], [0, 1], color="#1d4ed8", lw=5, solid_capstyle="butt")
    _ax2.add_patch(FancyArrowPatch((0.5, 0.15), (0.5, 0.85), arrowstyle="-|>", mutation_scale=22, lw=2, color="#444444"))
    _ax2.text(0.55, 0.5, "evolve in time", fontsize=12, color="#444444", va="center")
    _ax2.text(0.5, -0.12, r"initial condition at $t=0$", ha="center", color="#c2410c", fontsize=12)
    _ax2.text(-0.07, 0.5, "boundary\ncondition", ha="right", va="center", color="#1d4ed8", fontsize=12)
    _ax2.text(1.07, 0.5, "boundary\ncondition", ha="left", va="center", color="#1d4ed8", fontsize=12)
    _ax2.set_xlim(-0.5, 1.5)
    _ax2.set_ylim(-0.25, 1.25)
    _ax2.set_aspect("equal")
    _ax2.axis("off")
    _ax2.set_title("Initial value problem: space (horizontal) and time (vertical)", fontsize=11)
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.1 Boundary value problems

    ---

    The fundamental technique for solving PDEs numerically is the **method of finite
    differences**: divide space into a discrete grid of points and replace the partial
    derivatives by differences between values at neighbouring grid points.

    Take the electrostatics example: an empty box with conducting walls, all grounded
    except one wall at voltage $V$. With no charge inside, Maxwell's equations and
    $\mathbf E=-\nabla\phi$ give **Laplace's equation**

    $$
    \nabla^2\phi = 0
    \qquad (9.1)
    $$

    which in full reads

    $$
    \frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}+\frac{\partial^2\phi}{\partial z^2}=0
    \qquad (9.2)
    $$

    The challenge is to solve it inside the box, subject to $\phi=V$ on the top wall and
    $\phi=0$ on the other walls. To make things easy to visualise, work in two
    dimensions:

    $$
    \frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}=0.
    \qquad (9.3)
    $$

    Real problems are three-dimensional, but the techniques in two and three dimensions
    are fundamentally the same: once the 2D version works, the 3D one is a
    straightforward generalisation.

    We put a **regular square grid** on the box. Points are placed *on the boundary*
    (where we already know $\phi$) as well as in the interior (where we want to
    compute it).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    grid_M = mo.ui.slider(4, 20, value=10, step=1, label="Grid squares per side, M")
    return (grid_M,)


@app.cell(hide_code=True)
def _(grid_M, mo, np, plt):
    _M = int(grid_M.value)
    _a = 1.0 / _M
    _idx = np.arange(_M + 1)
    _X, _Y = np.meshgrid(_idx * _a, _idx * _a)
    _boundary = (_X == 0) | (_X == 1) | (_Y == 0) | (_Y == 1)
    _boundary = np.isclose(_X, 0) | np.isclose(_X, 1) | np.isclose(_Y, 0) | np.isclose(_Y, 1)
    _top = np.isclose(_Y, 1)

    _fig, _ax = plt.subplots(figsize=(5.4, 5.4))
    _ax.plot(_X[~_boundary], _Y[~_boundary], "o", color="#1d4ed8", ms=5, label="interior: unknown")
    _ax.plot(_X[_boundary & ~_top], _Y[_boundary & ~_top], "s", color="#444444", ms=5, label=r"boundary: $\phi=0$")
    _ax.plot(_X[_top], _Y[_top], "s", color="#c2410c", ms=5, label=r"top wall: $\phi=V$")
    for _k in _idx:
        _ax.plot([0, 1], [_k * _a, _k * _a], color="#cccccc", lw=0.6, zorder=0)
        _ax.plot([_k * _a, _k * _a], [0, 1], color="#cccccc", lw=0.6, zorder=0)
    _ax.annotate("", xy=(_a, -0.05), xytext=(0, -0.05), arrowprops=dict(arrowstyle="<->", color="#444444"))
    _ax.text(_a / 2, -0.11, "a", ha="center", fontsize=11, color="#444444")
    _ax.set_aspect("equal")
    _ax.set_xlim(-0.08, 1.08)
    _ax.set_ylim(-0.15, 1.1)
    _ax.axis("off")
    _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False, fontsize=9)
    _fig.subplots_adjust(bottom=0.2)

    mo.vstack(
        [
            grid_M,
            mo.hstack(
                [
                    mo.stat(label="Grid spacing a", value=f"{_a:.3f}"),
                    mo.stat(label="Boundary points (known)", value=f"{int(_boundary.sum())}"),
                    mo.stat(label="Interior points (unknown)", value=f"{int((~_boundary).sum())}"),
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
    The number of unknowns is the number of interior points, $(M-1)^2$ on an $M\times M$
    grid. Every one of them will get its own equation, so a fine grid means a *large*
    system of simultaneous equations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.1.1 The Laplacian operator

    ---

    Let the distance between adjacent grid points be $a$ and consider a grid point at
    $(x,y)$. Derivatives can be approximated with finite differences.
    Taylor-expand $\phi$ a step $\pm a$ in the $x$ direction:

    $$
    \begin{aligned}
    \phi(x+a,y) &= \phi + a\,\phi_x + \tfrac{1}{2}a^2\phi_{xx} + \tfrac{1}{6}a^3\phi_{xxx} + \tfrac{1}{24}a^4\phi_{xxxx}+\dots\\
    \phi(x-a,y) &= \phi - a\,\phi_x + \tfrac{1}{2}a^2\phi_{xx} - \tfrac{1}{6}a^3\phi_{xxx} + \tfrac{1}{24}a^4\phi_{xxxx}-\dots
    \end{aligned}
    $$

    Adding the two lines cancels the odd terms, and rearranging gives the second
    derivative in the $x$ direction in terms of three adjacent grid values:

    $$
    \frac{\partial^2\phi}{\partial x^2}
    = \frac{\phi(x+a,y)+\phi(x-a,y)-2\phi(x,y)}{a^2}\;\Big[\,-\tfrac{1}{12}a^2\phi_{xxxx}+\dots\Big]
    \qquad (9.4)
    $$

    where the bracket is the leading error: the approximation is **second order** in
    $a$. In the same way the $y$ direction gives

    $$
    \frac{\partial^2\phi}{\partial y^2}=\frac{\phi(x,y+a)+\phi(x,y-a)-2\phi(x,y)}{a^2},
    \qquad (9.5)
    $$

    and adding (9.4) and (9.5) gives the **Laplacian in two dimensions**:

    $$
    \frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}
    =\frac{\phi(x+a,y)+\phi(x-a,y)+\phi(x,y+a)+\phi(x,y-a)-4\phi(x,y)}{a^2}.
    \qquad (9.6)
    $$

    Expressed in words: *add the values of $\phi$ at the four neighbouring grid points and subtract
    four times the value at the point itself, then divide by $a^2$.*

    **Check 1: is the error really second order?** The cell below applies (9.4) to a
    function whose second derivative we know exactly and plots the error against $a$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    d2_func = mo.ui.dropdown(
        options=["sin(x)", "exp(x)", "x^4"], value="sin(x)", label="Function to differentiate"
    )
    return (d2_func,)


@app.cell(hide_code=True)
def _(d2_func, mo, np, plt):
    _x0 = 1.0
    _funcs = {
        "sin(x)": (np.sin, lambda x: -np.sin(x), np.sin),
        "exp(x)": (np.exp, np.exp, np.exp),
        "x^4": (lambda x: x**4, lambda x: 12 * x**2, lambda x: 24.0 + 0 * x),
    }
    _f, _f2, _f4 = _funcs[d2_func.value]
    _a = np.logspace(-6, 0, 60)
    _approx = (_f(_x0 + _a) + _f(_x0 - _a) - 2 * _f(_x0)) / _a**2
    _err = np.abs(_approx - _f2(_x0))
    _lead = _a**2 / 12 * np.abs(_f4(_x0))

    _fig, _ax = plt.subplots(figsize=(7.2, 4.2))
    _ax.loglog(_a, _err, "o", ms=4, color="#1d4ed8", label="measured error of (9.4)")
    _ax.loglog(_a, _lead, "k--", lw=1.2, label=r"leading error $a^2|\phi_{xxxx}|/12$")
    _ax.set(xlabel="grid spacing a", ylabel="|error| in second derivative at x = 1")
    _ax.grid(True, which="both", alpha=0.3)
    _ax.legend()
    _ax.set_title("Error falls as $a^2$, until rounding error takes over at tiny $a$")
    _fig.tight_layout()

    _mid = (_a > 1e-2) & (_a < 0.3)
    _slope = np.polyfit(np.log(_a[_mid]), np.log(_err[_mid]), 1)[0]
    mo.vstack(
        [
            d2_func,
            mo.stat(label="Fitted slope of error vs a (expect 2)", value=f"{_slope:.3f}"),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For $x^4$ the fourth derivative is the constant $24$, so the error is *exactly*
    $2a^2$ for every $a$. For very small $a$ the numerator becomes a
    difference of nearly equal numbers and **rounding error** dominates,
    which is why the curve turns upward on the left.

    ### The stencil

    Equation (9.6) is often drawn as a diagram, the **stencil** of the Laplacian: the
    weights multiplying $\phi$ at the centre point and its four neighbours, all divided
    by $a^2$.

    **Check 2: the stencil applied to a grid.** Choose a function with a known
    Laplacian and compare the stencil's answer with the exact one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lap_func = mo.ui.dropdown(
        options=["x^2 + y^2", "x^2 - y^2", "sin(pi x) sin(pi y)"],
        value="sin(pi x) sin(pi y)",
        label="Test function",
    )
    lap_M = mo.ui.slider(8, 80, value=20, step=4, label="Grid squares per side, M")
    return lap_M, lap_func


@app.cell(hide_code=True)
def _(lap_M, lap_func, mo, np, plt):
    _M = int(lap_M.value)
    _a = 1.0 / _M
    _g = np.arange(_M + 1) * _a
    _X, _Y = np.meshgrid(_g, _g, indexing="ij")
    _name = lap_func.value
    if _name == "x^2 + y^2":
        _phi = _X**2 + _Y**2
        _exact = 4.0 + 0 * _X
    elif _name == "x^2 - y^2":
        _phi = _X**2 - _Y**2
        _exact = 0.0 * _X
    else:
        _phi = np.sin(np.pi * _X) * np.sin(np.pi * _Y)
        _exact = -2 * np.pi**2 * _phi

    _num = np.zeros_like(_phi)
    _num[1:-1, 1:-1] = (
        _phi[2:, 1:-1] + _phi[:-2, 1:-1] + _phi[1:-1, 2:] + _phi[1:-1, :-2] - 4 * _phi[1:-1, 1:-1]
    ) / _a**2
    _err = (_num - _exact)[1:-1, 1:-1]

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for _ax, _data, _title in (
        (_axes[0], _phi, r"$\phi$"),
        (_axes[1], _num[1:-1, 1:-1], "stencil (9.6)"),
        (_axes[2], _err, "stencil minus exact"),
    ):
        _im = _ax.imshow(_data.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
        _ax.set_title(_title)
        _ax.set_xlabel("x")
        _fig.colorbar(_im, ax=_ax, fraction=0.046)
    _axes[0].set_ylabel("y")
    _fig.tight_layout()

    _scale = max(1.0, float(np.abs(_exact).max()))
    mo.vstack(
        [
            mo.hstack([lap_func, lap_M], justify="start"),
            mo.stat(
                label="Largest error in the interior",
                value=f"{np.abs(_err).max():.2e}",
                caption=f"relative to the largest exact value: {np.abs(_err).max() / _scale:.1e}",
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For $x^2\pm y^2$ the error is zero up to rounding (the fourth derivatives vanish, so
    (9.4) is exact for quadratics). For $\sin\pi x\,\sin\pi y$ the error shrinks by a
    factor of four every time $M$ is doubled: second order again.

    ### Matrix form

    There is another way to write (9.6). Number the grid points from $1$ to $N$ **in any
    order we like** and collect the values of $\phi$ in a vector
    $\boldsymbol\phi=(\phi_1,\phi_2,\dots,\phi_N)$. Then

    $$
    \frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}
    =\frac{1}{a^2}\,\mathbf L\boldsymbol\phi,
    \qquad (9.7)
    $$

    where $\mathbf L$ is an $N\times N$ **symmetric** matrix (the *graph Laplacian*) with
    elements

    $$
    L_{ij}=
    \begin{cases}
    -4 & \text{if } i=j,\\
    \;\;1 & \text{if grid points } i \text{ and } j \text{ are neighbours},\\
    \;\;0 & \text{otherwise.}
    \end{cases}
    \qquad (9.8)
    $$

    Row $i$ of $\mathbf L$ is exactly the stencil, placed where point $i$ sits in the
    numbering: $-4$ on the diagonal and $+1$ in the columns of its (up to four)
    neighbours. Here $N$ counts the points whose equation we write down. For the
    electrostatics box those are the interior points; boundary values, which are known,
    contribute a fixed vector on the right-hand side.

    **Check 3: build $\mathbf L$.** The cell below constructs $\mathbf L$ for an
    $n\times n$ block of interior points, numbering the points row by row, and shows the
    numbering next to the matrix.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def laplacian_2d(n):
        """Graph Laplacian (9.8) for an n-by-n block of interior points, numbered row by row."""
        N = n * n
        L = np.zeros((N, N))
        for iy in range(n):
            for ix in range(n):
                i = iy * n + ix
                L[i, i] = -4.0
                if ix > 0:
                    L[i, i - 1] = 1.0
                if ix < n - 1:
                    L[i, i + 1] = 1.0
                if iy > 0:
                    L[i, i - n] = 1.0
                if iy < n - 1:
                    L[i, i + n] = 1.0
        return L

    def laplacian_1d(N):
        """One-dimensional version: -2 on the diagonal, +1 next to it, for N points."""
        return -2.0 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)

    return laplacian_1d, laplacian_2d


@app.cell(hide_code=True)
def _(mo):
    lmat_n = mo.ui.slider(2, 6, value=4, step=1, label="Interior points per side, n")
    return (lmat_n,)


@app.cell(hide_code=True)
def _(laplacian_2d, lmat_n, mo, np, plt):
    _n = int(lmat_n.value)
    _L = laplacian_2d(_n)
    _N = _n * _n

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw=dict(width_ratios=[0.8, 1.2]))
    for _iy in range(_n):
        for _ix in range(_n):
            _ax1.plot(_ix, _iy, "o", ms=26, color="#e8eefc", mec="#1d4ed8")
            _ax1.text(_ix, _iy, str(_iy * _n + _ix + 1), ha="center", va="center", fontsize=11)
    _ax1.set_xlim(-0.7, _n - 0.3)
    _ax1.set_ylim(-0.7, _n - 0.3)
    _ax1.set_aspect("equal")
    _ax1.axis("off")
    _ax1.set_title("Numbering of the interior points", fontsize=11)

    _im = _ax2.imshow(_L, cmap="RdBu_r", vmin=-4, vmax=4)
    if _N <= 25:
        for _i in range(_N):
            for _j in range(_N):
                if _L[_i, _j] != 0:
                    _ax2.text(_j, _i, f"{int(_L[_i, _j])}", ha="center", va="center", fontsize=8,
                              color="white" if _i == _j else "black")
    _ax2.set_title(f"L ({_N} x {_N}): row i is the stencil centred on point i", fontsize=11)
    _ax2.set_xlabel("column j")
    _ax2.set_ylabel("row i")
    _fig.tight_layout()

    _nnz = int(np.count_nonzero(_L))
    mo.vstack(
        [
            lmat_n,
            mo.hstack(
                [
                    mo.stat(label="Matrix size N x N", value=f"{_N} x {_N}"),
                    mo.stat(label="Non-zero entries", value=f"{_nnz}", caption=f"{100 * _nnz / _N**2:.1f}% of all entries"),
                    mo.stat(label="Symmetric?", value=str(bool(np.allclose(_L, _L.T)))),
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
    Three things to notice: $\mathbf L$ is **symmetric**, it is very **sparse** (at most
    five non-zero entries per row, however large $N$ is), and its diagonal blocks and
    off-diagonal stripes follow directly from the numbering. The numbering is a free
    choice; a different numbering permutes rows and columns but describes the same operator.

    **Check 4: the matrix gives the same answer as the stencil.** Apply $\mathbf L/a^2$ to
    the test function $\sin\pi x\sin\pi y$ (which vanishes on the boundary, so no boundary
    terms are needed) and compare with applying (9.6) point by point.
    """)
    return


@app.cell(hide_code=True)
def _(laplacian_2d, mo, np):
    _n = 12
    _a = 1.0 / (_n + 1)
    _g = np.arange(1, _n + 1) * _a
    _X, _Y = np.meshgrid(_g, _g, indexing="xy")  # row index = y, column index = x
    _phi = np.sin(np.pi * _X) * np.sin(np.pi * _Y)
    _via_matrix = (laplacian_2d(_n) @ _phi.ravel()) / _a**2

    _pad = np.zeros((_n + 2, _n + 2))
    _pad[1:-1, 1:-1] = _phi
    _via_stencil = (
        _pad[2:, 1:-1] + _pad[:-2, 1:-1] + _pad[1:-1, 2:] + _pad[1:-1, :-2] - 4 * _phi
    ) / _a**2
    _diff = np.abs(_via_matrix - _via_stencil.ravel()).max()
    _exact = -2 * np.pi**2 * _phi.ravel()
    mo.hstack(
        [
            mo.stat(label="Largest difference, matrix vs stencil", value=f"{_diff:.1e}"),
            mo.stat(label="Largest deviation from exact Laplacian", value=f"{np.abs(_via_matrix - _exact).max():.2f}",
                    caption=f"out of {np.abs(_exact).max():.1f}: finite-difference error at M = {_n + 1}"),
        ],
        justify="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What is $\mathbf L$ good for?

    With the Laplacian as a matrix, Laplace's equation turns into **linear algebra**.
    For the box of Fig. 9.1 the unknowns are the interior values. Every interior point
    next to the top wall has a neighbour at $V$; those *known* neighbours move to the
    right-hand side, leaving

    $$
    \mathbf L\boldsymbol\phi=-\mathbf b,
    \qquad b_i=\begin{cases}V & \text{if point } i \text{ touches the top wall},\\ 0&\text{otherwise,}\end{cases}
    $$

    a system $\mathbf A\mathbf x=\mathbf v$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    box_n = mo.ui.slider(5, 41, value=21, step=4, label="Interior points per side, n")
    return (box_n,)


@app.cell(hide_code=True)
def _(box_n, laplacian_2d, mo, np, plt):
    _n = int(box_n.value)
    _V = 1.0
    _L = laplacian_2d(_n)
    _b = np.zeros(_n * _n)
    _b[(_n - 1) * _n:] = _V  # last row of the numbering touches the top wall
    _phi = np.linalg.solve(_L, -_b).reshape(_n, _n)

    _full = np.zeros((_n + 2, _n + 2))
    _full[1:-1, 1:-1] = _phi
    _full[-1, :] = _V
    _full[-1, 0] = _full[-1, -1] = 0.5 * _V  # corners: purely cosmetic for the plot

    _fig, _ax = plt.subplots(figsize=(5.6, 5.0))
    _im = _ax.imshow(_full, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", vmin=0, vmax=_V)
    _fig.colorbar(_im, ax=_ax, label=r"$\phi / V$", fraction=0.046)
    _ax.set(xlabel="x", ylabel="y", title="Potential in the box, from solving L phi = -b")
    _fig.tight_layout()

    _c = _n // 2
    mo.vstack(
        [
            box_n,
            mo.hstack(
                [
                    mo.stat(label="Centre value", value=f"{_phi[_c, _c]:.6f}"),
                    mo.stat(label="Expected by symmetry (V/4)", value=f"{_V / 4:.6f}"),
                    mo.stat(label="Matrix size", value=f"{_n * _n} x {_n * _n}"),
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
    The value at the centre is a useful sanity check. Rotating the box by $90^\circ$ four
    times moves the raised wall to each side in turn; the four solutions add up to the
    solution with *all four* walls at $V$, which is $\phi=V$ everywhere. By symmetry each
    rotated solution has the same centre value, so the centre value is $V/4$, exactly, on
    any grid with a central point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.3.4 Spectral methods

    ---

    Forward integration is not the only approach. The *finite element method* (not the
    finite *difference* method used in this chapter) solves the equation inside small
    elements and stitches the pieces together; it is widely used but is the subject of
    whole books. The **spectral method** is a different idea: not particularly
    complicated, enormously useful in physics, and, whenever it applies, it can give
    better results than either finite differences or finite elements.

    ### The idea

    Take the wave equation in $d$ dimensions,

    $$
    \nabla^2\phi-\frac{1}{v^2}\frac{\partial^2\phi}{\partial t^2}=0.
    \qquad (9.70)
    $$

    Look for solutions that oscillate with a single frequency $\omega$:

    $$
    \phi(\mathbf r,t)=\rho(\mathbf r)\,e^{i\omega t}.
    \qquad (9.71)
    $$

    Substituting into (9.70) and cancelling $e^{i\omega t}$ leaves a time-independent
    equation for the spatial shape $\rho$:

    $$
    \nabla^2\rho=-k^2\rho,\qquad k=\frac{\omega}{v}.
    \qquad (9.73)
    $$

    This is the **Helmholtz equation**: $\rho(\mathbf r)$ is an *eigenfunction of the
    Laplacian* with eigenvalue $-k^2$. Solve that eigenproblem once, and the
    time dependence is known for every mode: it is just $e^{i\omega t}$.

    **Why only negative eigenvalues?** Suppose $\nabla^2\rho=\lambda\rho$ with $\rho$
    normalised, $\int\rho^2\,d^d\mathbf r=1$. Multiply by $\rho$ and integrate:
    $\int\rho\nabla^2\rho\,d^d\mathbf r=\lambda$ (9.75). The chain rule gives
    $\nabla\cdot(\rho\nabla\rho)=|\nabla\rho|^2+\rho\nabla^2\rho$ (9.76), and Gauss's law
    turns the integral of the left-hand side into a surface term
    $\oint\rho\nabla\rho\cdot d\mathbf S$ that vanishes if either $\rho=0$ on the
    boundary (Dirichlet) or the normal gradient is zero (Neumann). Therefore

    $$
    \lambda=-\int|\nabla\rho|^2\,d^d\mathbf r\;\le\;0.
    \qquad (9.79)
    $$

    All eigenvalues of the Laplacian are zero or negative, so writing the eigenvalue as
    $-k^2$ costs nothing: the Helmholtz equation is simply "$\rho$ is an eigenfunction of
    the Laplacian".

    ### On a grid: the eigenproblem is a matrix eigenproblem

    On a regular grid with spacing $a$ (numbered $1,\dots,N$ in any order) the Laplacian
    is the matrix of Section 9.1.1:

    $$
    \nabla^2\boldsymbol\rho=\frac{1}{a^2}\mathbf L\boldsymbol\rho
    \quad\Longrightarrow\quad
    \frac{1}{a^2}\mathbf L\boldsymbol\rho=-k^2\boldsymbol\rho.
    \qquad (9.80\text{–}9.82)
    $$

    The eigenfunction on the grid points is an **eigenvector of the symmetric matrix
    $\mathbf L/a^2$**. We know how to find those: the QR algorithm, or
    `eigh` in `numpy.linalg`. For $N$ grid points there are $N$ eigenvectors
    $\boldsymbol\rho_n$ with eigenvalues $-k_n^2$, and each one gives a solution

    $$
    \phi(\mathbf r_i,t)=\rho_{ni}\,e^{i\omega_n t},\qquad \omega_n=vk_n.
    \qquad (9.83)
    $$

    ### Building the complete solution

    The wave equation is linear, so **any linear combination of solutions is a solution**:

    $$
    \boldsymbol\phi(t)=\sum_{n=1}^{N}b_n\,\boldsymbol\rho_n\,e^{i\omega_nt},
    \qquad b_n=\alpha_n+i\eta_n,
    \qquad (9.84\text{–}9.85)
    $$

    and the physical solution is the **real part**. The $\boldsymbol\rho_n$ are real because
    $\mathbf L$ is symmetric. Setting $t=0$ and then differentiating in time:

    $$
    \boldsymbol\phi(0)=\sum_n\alpha_n\boldsymbol\rho_n,
    \qquad
    \left(\frac{d\boldsymbol\phi}{dt}\right)_{t=0}=-v\sum_nk_n\eta_n\boldsymbol\rho_n.
    \qquad (9.86\text{–}9.87)
    $$

    Eigenvectors of a symmetric matrix are orthonormal,
    $\boldsymbol\rho_m\cdot\boldsymbol\rho_n=\delta_{mn}$ (9.88), so *dotting* these two
    equations with $\boldsymbol\rho_n$ picks out the coefficients:

    $$
    \boxed{\;\alpha_n=\boldsymbol\rho_n\cdot\boldsymbol\phi(0),
    \qquad
    \eta_n=-\frac{1}{vk_n}\,\boldsymbol\rho_n\cdot\left(\frac{d\boldsymbol\phi}{dt}\right)_{t=0}\;}
    \qquad (9.91)
    $$

    Both initial conditions, the displacement and the velocity, are matched, and by
    uniqueness that **is** the solution for all time. Explicitly,

    $$
    \phi(x_i,t)=\sum_n\Big[\alpha_n\cos(\omega_nt)-\eta_n\sin(\omega_nt)\Big]\rho_{ni}.
    $$

    ### Recipe

    1. Compute eigenvalues $-k_n^2$ and eigenvectors $\boldsymbol\rho_n$ of $\mathbf L/a^2$ (once).
    2. Compute $\alpha_n$ and $\eta_n$ from the initial displacement and velocity (9.91).
    3. Evaluate the sum above at **any** time $t$ you like.

    **Advantage:** there is no stepping through time. To see the solution at
    $t=10$ s you go straight there. **No stability problem** either. **Drawback:** the sum has $N$ terms
    at each of $N$ grid points ($N^2$ terms in total) and it only works for **linear**
    equations. The following demonstrations make these points concrete.
    """)
    return


@app.cell(hide_code=True)
def _(laplacian_1d, np):
    # Piano string of Exercise 9.9: length L, fixed ends, struck at rest.
    str_L = 1.0       # m
    str_v = 100.0     # m/s
    str_N = 100       # slices; N - 1 interior points
    str_a = str_L / str_N
    str_x = np.arange(1, str_N) * str_a
    _d, _C, _sigma = 0.10, 1.0, 0.3
    str_phi0 = np.zeros(str_N - 1)
    str_psi0 = _C * str_x * (str_L - str_x) / str_L**2 * np.exp(-((str_x - _d) ** 2) / (2 * _sigma**2))
    str_Lap = laplacian_1d(str_N - 1)

    def wave_energy(phi, psi):
        """Discrete energy of the string, 1/2 a sum(psi^2) + 1/2 a v^2 sum((dphi/dx)^2)."""
        padded = np.concatenate([[0.0], phi, [0.0]])
        slope = np.diff(padded) / str_a
        return 0.5 * str_a * np.sum(psi**2) + 0.5 * str_a * str_v**2 * np.sum(slope**2)

    return (
        str_L,
        str_Lap,
        str_N,
        str_a,
        str_phi0,
        str_psi0,
        str_v,
        str_x,
        wave_energy,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Demonstration 1: the eigenvectors of $\mathbf L$ are standing waves

    We use the system of Exercise 9.9: a piano string of length $L=1$ m with fixed ends,
    wave speed $v=100$ m/s, divided into $N=100$ slices. The $N-1=99$ interior points
    form a *one-dimensional* grid, and the Laplacian matrix of 9.1.1 becomes the
    tridiagonal matrix with $-2$ on the diagonal and $+1$ beside it.

    Diagonalising it gives the eigenvalues $-k_n^2$ and eigenvectors $\boldsymbol\rho_n$.
    For this matrix the answers are known exactly,

    $$
    \lambda_n=-4\sin^2\!\frac{n\pi}{2N},
    \qquad
    \rho_{n}(x_j)=\sqrt{\frac{2}{N}}\,\sin\frac{n\pi j}{N},
    $$

    i.e. the standing waves of the string, which gives us a direct check on the code.
    """)
    return


@app.cell(hide_code=True)
def _(eigh, np, str_Lap, str_N, str_a):
    _lam, _rho = eigh(str_Lap)
    # eigh returns ascending eigenvalues (most negative first); reorder so n = 1 is the longest wave
    _lam = _lam[::-1]
    _rho = _rho[:, ::-1]
    for _n in range(_rho.shape[1]):
        if _rho[0, _n] < 0:
            _rho[:, _n] = -_rho[:, _n]
    str_lam = _lam
    str_rho = _rho
    str_k = np.sqrt(-_lam) / str_a
    _n_arr = np.arange(1, str_N)
    str_lam_err = np.abs(_lam - (-4 * np.sin(_n_arr * np.pi / (2 * str_N)) ** 2)).max()
    return str_k, str_lam_err, str_rho


@app.cell(hide_code=True)
def _(mo, str_N):
    mode_n = mo.ui.slider(1, str_N - 1, value=3, step=1, label="Mode number n")
    return (mode_n,)


@app.cell(hide_code=True)
def _(
    mo,
    mode_n,
    np,
    plt,
    str_L,
    str_N,
    str_k,
    str_lam_err,
    str_rho,
    str_v,
    str_x,
):
    _n = int(mode_n.value)
    _rho = str_rho[:, _n - 1]
    _exact = np.sqrt(2.0 / str_N) * np.sin(_n * np.pi * str_x / str_L)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.2, 1]))
    _ax1.plot(str_x, _rho, "o", ms=4, color="#1d4ed8", label="eigenvector from eigh")
    _ax1.plot(str_x, _exact, "k-", lw=1, label=r"$\sqrt{2/N}\,\sin(n\pi x/L)$")
    _ax1.set(xlabel="x (m)", ylabel=r"$\rho_n$", title=f"Mode n = {_n}")
    _ax1.legend(loc="upper right", fontsize=8)
    _ax1.grid(alpha=0.3)

    _nn = np.arange(1, str_N)
    _ax2.plot(_nn, str_k * str_v / (2 * np.pi), "o", ms=3, color="#c2410c", label="frequency from the grid")
    _ax2.plot(_nn, _nn * str_v / (2 * str_L), "k--", lw=1, label="continuum  n v / 2L")
    _ax2.plot([_n], [str_k[_n - 1] * str_v / (2 * np.pi)], "o", ms=10, mfc="none", mec="k")
    _ax2.set(xlabel="mode number n", ylabel="frequency (Hz)", title=r"Frequencies $\omega_n/2\pi$")
    _ax2.legend(fontsize=8)
    _ax2.grid(alpha=0.3)
    _fig.tight_layout()

    _orth = np.abs(str_rho.T @ str_rho - np.eye(str_rho.shape[1])).max()
    mo.vstack(
        [
            mode_n,
            mo.hstack(
                [
                    mo.stat(label="Eigenvalue error vs exact formula", value=f"{str_lam_err:.1e}"),
                    mo.stat(label="Orthonormality error", value=f"{_orth:.1e}"),
                    mo.stat(label="Frequency of this mode", value=f"{str_k[_n - 1] * str_v / (2 * np.pi):.1f} Hz"),
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
    Low modes are sine waves and the frequencies follow the continuum line
    $f_n=nv/2L$. At high $n$ the grid frequencies fall below the continuum line: a wave
    with only a few grid points per wavelength is not well represented by the finite
    difference second derivative. That is the (second-order) discretisation error of
    9.1.1 showing up in the spectrum.

    ### Demonstration 2: solving the string with the spectral recipe

    The function below implements steps 2 and 3 of the recipe. Given the eigenvectors and
    wavenumbers, and the initial displacement and velocity, it returns a function that
    evaluates $\phi$ **and** $\partial\phi/\partial t$ at any time directly.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def spectral_wave(rho, k, v, phi0, psi0):
        """Spectral solution of the wave equation, Eqs. (9.84)-(9.91).

        Returns a function t -> (phi, dphi/dt) on the grid points, with no time stepping.
        """
        omega = v * k
        alpha = rho.T @ phi0                  # alpha_n = rho_n . phi(0)
        eta = -(rho.T @ psi0) / omega         # eta_n = -(1 / v k_n) rho_n . dphi/dt(0)

        def solution(t):
            c, s = np.cos(omega * t), np.sin(omega * t)
            phi = rho @ (alpha * c - eta * s)
            dphi = rho @ (-alpha * omega * s - eta * omega * c)
            return phi, dphi

        return solution

    return (spectral_wave,)


@app.cell(hide_code=True)
def _(
    mo,
    np,
    solve_ivp,
    spectral_wave,
    str_Lap,
    str_a,
    str_k,
    str_phi0,
    str_psi0,
    str_rho,
    str_v,
):
    _sol = spectral_wave(str_rho, str_k, str_v, str_phi0, str_psi0)
    _n = len(str_phi0)

    def _rhs(t, y):
        p, q = y[:_n], y[_n:]
        return np.concatenate([q, (str_v / str_a) ** 2 * (str_Lap @ p)])

    _rows = []
    for _t in (0.002, 0.05, 0.1):
        _ref = solve_ivp(_rhs, [0, _t], np.concatenate([str_phi0, str_psi0]),
                         method="DOP853", rtol=1e-11, atol=1e-14).y[:_n, -1]
        _phi, _ = _sol(_t)
        _rows.append((_t, np.abs(_phi - _ref).max(), np.abs(_ref).max()))

    _table = "| t (s) | largest difference | size of the solution |\n|---|---|---|\n" + "\n".join(
        f"| {_t:.3f} | {_d:.1e} | {_m:.1e} |" for _t, _d, _m in _rows
    )
    mo.md(
        "**Check against an independent solver.** The spectral formula is exact for the grid "
        "equations (it solves the semi-discrete system without any time-stepping error). Here "
        "it is compared with SciPy's `DOP853` integrating the same system with tight "
        f"tolerances:\n\n{_table}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    wave_t_ms = mo.ui.slider(0, 200, value=25, step=1, label="Time t (ms)")
    return (wave_t_ms,)


@app.cell(hide_code=True)
def _(
    mo,
    plt,
    spectral_wave,
    str_k,
    str_phi0,
    str_psi0,
    str_rho,
    str_v,
    str_x,
    wave_energy,
    wave_t_ms,
):
    _sol = spectral_wave(str_rho, str_k, str_v, str_phi0, str_psi0)
    _t = wave_t_ms.value * 1e-3
    _phi, _dphi = _sol(_t)
    _phi_e, _dphi_e = _sol(0.0)

    _fig, _ax = plt.subplots(figsize=(8.5, 3.8))
    _ax.plot(str_x, 1e3 * _phi, color="#1d4ed8", lw=2)
    _ax.axhline(0, color="#888888", lw=0.8)
    _ax.set(xlabel="x (m)", ylabel="displacement (mm)", title=f"Piano string at t = {wave_t_ms.value} ms (spectral method, no time-stepping)")
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    mo.vstack(
        [
            wave_t_ms,
            mo.hstack(
                [
                    mo.stat(label="Energy now", value=f"{wave_energy(_phi, _dphi):.6e}"),
                    mo.stat(label="Energy at t = 0", value=f"{wave_energy(_phi_e, _dphi_e):.6e}"),
                ],
                justify="center",
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, np, plt, str_k, str_phi0, str_psi0, str_rho, str_v):
    _sol_omega = str_v * str_k
    _alpha = str_rho.T @ str_phi0
    _eta = -(str_rho.T @ str_psi0) / _sol_omega
    _t = np.linspace(0, 0.06, 301)
    _field = str_rho @ (_alpha[:, None] * np.cos(np.outer(_sol_omega, _t)) - _eta[:, None] * np.sin(np.outer(_sol_omega, _t)))

    _fig, _ax = plt.subplots(figsize=(8.5, 4.2))
    _im = _ax.imshow(1e3 * _field.T, origin="lower", aspect="auto", extent=[0, 1, 0, 60], cmap="RdBu_r",
                     vmin=-1e3 * np.abs(_field).max(), vmax=1e3 * np.abs(_field).max())
    _fig.colorbar(_im, ax=_ax, label="displacement (mm)")
    _ax.set(xlabel="x (m)", ylabel="t (ms)", title="Space-time picture of the struck string")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                "Evaluating the spectral formula on a grid of times gives the whole space-time "
                "picture at once. The disturbance travels at $v=100$ m/s, reflects at the fixed "
                "ends with a sign change, and the pattern repeats with period $2L/v=20$ ms."
            ),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Demonstration 3: The same string with FTCS

    For comparison, here is the **FTCS** method (Section 9.3.1 / Exercise 9.9) applied to
    the same string: the two first-order equations

    $$
    \phi(x,t+h)=\phi(x,t)+h\,\psi(x,t),\qquad
    \psi(x,t+h)=\psi(x,t)+h\frac{v^2}{a^2}\big[\phi(x+a,t)+\phi(x-a,t)-2\phi(x,t)\big]
    $$

    with the recommended time step $h=10^{-6}$ s, i.e. $10^5$ steps to reach 100 ms. The
    run below is performed once, storing snapshots and the energy along the way.
    """)
    return


@app.cell(hide_code=True)
def _(np, str_a, str_phi0, str_psi0, str_v, time, wave_energy):
    _h = 1e-6
    _p = str_phi0.copy()
    _q = str_psi0.copy()
    _snap_steps = {2000: "2 ms", 50000: "50 ms", 100000: "100 ms"}
    ftcs_snaps = {}
    _e_t, _e = [0.0], [wave_energy(_p, _q)]
    _c = _h * (str_v / str_a) ** 2
    _t0 = time.perf_counter()
    for _step in range(1, 100001):
        _pad = np.concatenate([[0.0], _p, [0.0]])
        _lap = _pad[2:] + _pad[:-2] - 2 * _p
        _p, _q = _p + _h * _q, _q + _c * _lap
        if _step in _snap_steps:
            ftcs_snaps[_snap_steps[_step]] = _p.copy()
        if _step % 500 == 0:
            _e_t.append(_step * _h)
            _e.append(wave_energy(_p, _q))
    ftcs_seconds = time.perf_counter() - _t0
    ftcs_energy_t = np.array(_e_t)
    ftcs_energy = np.array(_e)
    return ftcs_energy, ftcs_energy_t, ftcs_seconds, ftcs_snaps


@app.cell(hide_code=True)
def _(
    ftcs_snaps,
    mo,
    plt,
    spectral_wave,
    str_k,
    str_phi0,
    str_psi0,
    str_rho,
    str_v,
    str_x,
):
    _sol = spectral_wave(str_rho, str_k, str_v, str_phi0, str_psi0)
    _times = {"2 ms": 2e-3, "50 ms": 50e-3, "100 ms": 100e-3}
    _fig, _axes = plt.subplots(2, 3, figsize=(12, 5.6))
    for _j, (_label, _t) in enumerate(_times.items()):
        _axes[0, _j].plot(str_x, 1e3 * ftcs_snaps[_label], color="#c2410c")
        _axes[0, _j].set_title(f"FTCS, t = {_label}")
        _axes[1, _j].plot(str_x, 1e3 * _sol(_t)[0], color="#1d4ed8")
        _axes[1, _j].set_title(f"spectral, t = {_label}")
        for _i in (0, 1):
            _axes[_i, _j].grid(alpha=0.3)
            _axes[_i, _j].set_xlabel("x (m)")
    _axes[0, 0].set_ylabel("displacement (mm)")
    _axes[1, 0].set_ylabel("displacement (mm)")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.center(_fig),
            mo.md(
                "At 2 ms the two agree. By 50 ms FTCS shows fine ripples, and by 100 ms the "
                "vertical scale of the FTCS panel is hundreds of times the size of the true "
                "solution: the high-frequency modes grow exponentially (**numerical "
                "instability**). Each panel has its own vertical scale, so compare the numbers "
                "on the axes. The spectral solution is unaffected."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    ftcs_energy,
    ftcs_energy_t,
    ftcs_seconds,
    mo,
    np,
    plt,
    spectral_wave,
    str_k,
    str_phi0,
    str_psi0,
    str_rho,
    str_v,
    time,
    wave_energy,
):
    _sol = spectral_wave(str_rho, str_k, str_v, str_phi0, str_psi0)
    _ts = np.linspace(0, 0.1, 101)
    _e_spec = np.array([wave_energy(*_sol(_t)) for _t in _ts])

    _t0 = time.perf_counter()
    for _rep in range(100):
        _sol(0.1)
    _t_spec = (time.perf_counter() - _t0) / 100

    _fig, _ax = plt.subplots(figsize=(8.5, 3.8))
    _ax.semilogy(ftcs_energy_t * 1e3, ftcs_energy / ftcs_energy[0], color="#c2410c", label="FTCS")
    _ax.semilogy(_ts * 1e3, _e_spec / _e_spec[0], color="#1d4ed8", lw=2, label="spectral")
    _ax.set(xlabel="t (ms)", ylabel="energy / initial energy", title="The string's energy should be constant")
    _ax.grid(True, which="both", alpha=0.3)
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(label="FTCS: energy at 100 ms / initial", value=f"{ftcs_energy[-1] / ftcs_energy[0]:.1e}"),
                    mo.stat(label="Spectral: variation of the energy", value=f"{(_e_spec.max() - _e_spec.min()) / _e_spec[0]:.1e}"),
                    mo.stat(label="FTCS: 100000 steps", value=f"{ftcs_seconds:.2f} s"),
                    mo.stat(label="Spectral: one evaluation at t = 100 ms", value=f"{1e3 * _t_spec:.3f} ms"),
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
    The energy of a frictionless string is conserved. The spectral solution keeps it
    constant to rounding error; FTCS creates energy from nothing and the energy grows by
    many orders of magnitude. And the cost: FTCS needs $10^5$ steps to reach 100 ms; the
    spectral method needs a single evaluation of the sum, after a one-off diagonalisation.

    ### A shortcut for evenly spaced grids

    For this grid the eigenvectors are exactly sine waves, so the coefficient formulas
    $\alpha_n=\boldsymbol\rho_n\cdot\boldsymbol\phi(0)$ and the sum (9.84) are **discrete
    sine transforms**. These can be evaluated with the fast Fourier transform method in $O(N\log N)$ operations instead of $O(N^2)$, which makes the spectral
    method almost as cheap per evaluation as FTCS per step, with the bonus of skipping
    straight to any time. The cell below checks that the fast transform reproduces the
    eigenvector solution and compares the cost for a finer grid.
    """)
    return


@app.cell(hide_code=True)
def _(
    dst,
    mo,
    np,
    spectral_wave,
    str_N,
    str_k,
    str_phi0,
    str_psi0,
    str_rho,
    str_v,
    time,
):
    def _dst_solution(phi0, psi0, k, v, N, t):
        omega = v * k
        alpha = dst(phi0, type=1) / np.sqrt(2 * N)
        eta = -dst(psi0, type=1) / np.sqrt(2 * N) / omega
        coeff = alpha * np.cos(omega * t) - eta * np.sin(omega * t)
        return dst(coeff, type=1) / np.sqrt(2 * N)

    _t = 0.05
    _slow = spectral_wave(str_rho, str_k, str_v, str_phi0, str_psi0)(_t)[0]
    _fast = _dst_solution(str_phi0, str_psi0, str_k, str_v, str_N, _t)
    _agree = np.abs(_slow - _fast).max()

    # timing on a finer grid, using the exact sine eigenvectors
    _Nf = 2000
    _a = 1.0 / _Nf
    _j = np.arange(1, _Nf)
    _kf = 2 * np.sin(_j * np.pi / (2 * _Nf)) / _a
    _x = _j * _a
    _phi0 = np.zeros(_Nf - 1)
    _psi0 = _x * (1 - _x) * np.exp(-((_x - 0.1) ** 2) / (2 * 0.3**2))
    _rho = np.sqrt(2.0 / _Nf) * np.sin(np.outer(_j, _j) * np.pi / _Nf)
    _sol_mat = spectral_wave(_rho, _kf, str_v, _phi0, _psi0)

    _t0 = time.perf_counter()
    for _rep in range(20):
        _sol_mat(_t)
    _t_mat = (time.perf_counter() - _t0) / 20
    _t0 = time.perf_counter()
    for _rep in range(20):
        _dst_solution(_phi0, _psi0, _kf, str_v, _Nf, _t)
    _t_dst = (time.perf_counter() - _t0) / 20
    _same = np.abs(_sol_mat(_t)[0] - _dst_solution(_phi0, _psi0, _kf, str_v, _Nf, _t)).max()

    mo.hstack(
        [
            mo.stat(label="N = 100: fast transform vs eigenvectors", value=f"{_agree:.1e}", caption="largest difference"),
            mo.stat(label="N = 2000: matrix-vector evaluation", value=f"{1e3 * _t_mat:.2f} ms"),
            mo.stat(label="N = 2000: fast sine transform", value=f"{1e3 * _t_dst:.2f} ms", caption=f"difference between the two: {_same:.1e}"),
        ],
        justify="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Other geometries, other equations

    - **Any dimension.** The recipe only needs the Laplacian as a symmetric matrix. In
      two dimensions that is the matrix $\mathbf L$ of Eq. (9.8), built in 9.1.1.
    - **Triangulated regions.** On the irregular grids of Section 9.1.4 the Laplacian takes
      the cotangent form (9.18), and the eigenvectors needed are those of $\mathbf A^{-1}\mathbf L$,
      where $\mathbf A$ is the diagonal "mass matrix" (9.19). That matrix is not symmetric,
      so the usual QR algorithm does not apply. Instead one solves the *generalised*
      eigenvector equation
      $$\mathbf L\boldsymbol\rho=-k^2\mathbf A\boldsymbol\rho,\qquad (9.92)$$
      which is symmetric and can be solved quickly with a variant of the QR algorithm,
      for example `scipy.linalg.eigh(L, A)`.
    - **Other equations.** The same idea works for the diffusion equation and for the
      Schrödinger equation. **Limitation:** it needs a *linear* equation, because we
      add individual solutions together. Forward integration methods do not have this
      restriction.

    A quick check of the generalised form: if the mass matrix is simply $\mathbf A=a^2\mathbf I$,
    the generalised problem must reproduce the ordinary one.
    """)
    return


@app.cell(hide_code=True)
def _(eigh, mo, np, str_Lap, str_a, str_k):
    _A = str_a**2 * np.eye(str_Lap.shape[0])
    _w, _ = eigh(-str_Lap, _A)  # L rho = -k^2 A rho  =>  (-L) rho = k^2 A rho
    mo.stat(
        label="eigh(-L, A) with A = a^2 I: largest difference in k^2 vs the ordinary eigenproblem",
        value=f"{np.abs(np.sort(_w) - np.sort(str_k**2)).max():.1e}",
        caption=f"relative to the largest k^2 = {np.max(str_k**2):.1e}",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 9.14: the Schrödinger equation by the spectral method

    The spectral method is not limited to the wave equation. For the time-dependent
    Schrödinger equation of a particle in a box with impenetrable walls,

    $$
    -\frac{\hbar^2}{2M}\frac{\partial^2\psi}{\partial x^2}=i\hbar\frac{\partial\psi}{\partial t},
    $$

    try $\psi(x,t)=\rho(x)e^{-iEt/\hbar}$. This gives $-\frac{\hbar^2}{2M}\rho''=E\rho$,
    and on the grid

    $$
    \frac{1}{a^2}\mathbf L\boldsymbol\rho=-\frac{2ME}{\hbar^2}\boldsymbol\rho,
    $$

    with $\mathbf L$ the one-dimensional Laplacian matrix: **the eigenvectors of $\mathbf L$
    are the energy eigenstates and the energies are proportional to its eigenvalues,**
    $E_n=-\hbar^2\lambda_n/(2Ma^2)$. The equation is first order in time, so only
    *one* initial condition is needed and

    $$
    \boldsymbol\psi(t)=\sum_{n=1}^{N}b_n\,\boldsymbol\rho_n\,e^{-iE_nt/\hbar},
    \qquad
    b_n=\boldsymbol\rho_n\cdot\boldsymbol\psi(0).
    $$

    **The system** (same as Exercise 9.13): an electron, $M=9.109\times10^{-31}$ kg, in a box
    of length $D=10^{-8}$ m, starting in the Gaussian wave packet

    $$
    \psi(x,0)=\exp\!\Big[-\frac{(x-x_0)^2}{2\sigma^2}\Big]e^{i\kappa x},
    \quad x_0=\tfrac{D}{2},\ \sigma=10^{-10}\,\text{m},\ \kappa=5\times10^{10}\,\text{m}^{-1},
    $$

    with $\psi=0$ at both walls, on $N=1000$ slices. (The wavefunction is not
    normalised; since the equation is linear the constant plays no role.)
    """)
    return


@app.cell(hide_code=True)
def _(eigh_tridiagonal, np):
    sch_hbar = 1.0546e-34
    sch_M = 9.109e-31
    sch_D = 1e-8
    sch_N = 1000
    sch_a = sch_D / sch_N
    sch_x = np.arange(1, sch_N) * sch_a
    _x0, _sigma, _kappa = sch_D / 2, 1e-10, 5e10
    sch_psi0 = np.exp(-((sch_x - _x0) ** 2) / (2 * _sigma**2)) * np.exp(1j * _kappa * sch_x)
    sch_kappa = _kappa

    _lam, _rho = eigh_tridiagonal(-2.0 * np.ones(sch_N - 1), np.ones(sch_N - 2))
    sch_rho = _rho[:, ::-1].copy()
    for _n in range(sch_rho.shape[1]):
        if sch_rho[0, _n] < 0:
            sch_rho[:, _n] = -sch_rho[:, _n]
    sch_E = -sch_hbar**2 * _lam[::-1] / (2 * sch_M * sch_a**2)   # E_n, n = 1 ... N-1
    sch_b = sch_rho.T @ sch_psi0                                   # b_n = rho_n . psi(0)
    return (
        sch_D,
        sch_E,
        sch_M,
        sch_b,
        sch_hbar,
        sch_kappa,
        sch_psi0,
        sch_rho,
        sch_x,
    )


@app.cell(hide_code=True)
def _(mo, np, plt, sch_D, sch_E, sch_M, sch_b, sch_hbar, sch_kappa):
    _n = np.arange(1, len(sch_E) + 1)
    _E_exact = (_n * np.pi * sch_hbar / sch_D) ** 2 / (2 * sch_M)
    _eV = 1.602e-19
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.9))
    _ax1.plot(_n, sch_E / _eV, color="#1d4ed8", lw=2, label="lattice eigenvalues")
    _ax1.plot(_n, _E_exact / _eV, "k--", lw=1, label=r"$n^2\pi^2\hbar^2/2MD^2$")
    _ax1.set(xlabel="n", ylabel="energy (eV)", title=r"Energy levels $E_n$")
    _ax1.set_xlim(0, 400)
    _ax1.set_ylim(0, 450)
    _ax1.legend()
    _ax1.grid(alpha=0.3)
    _ax2.plot(_n, np.abs(sch_b) ** 2 / np.sum(np.abs(sch_b) ** 2), color="#c2410c")
    _ax2.axvline(sch_kappa * sch_D / np.pi, color="k", ls="--", lw=1, label=r"$n=\kappa D/\pi$")
    _ax2.set(xlabel="n", ylabel=r"$|b_n|^2$ (normalised)", title="Weight of each energy state in the packet")
    _ax2.set_xlim(0, 400)
    _ax2.legend()
    _ax2.grid(alpha=0.3)
    _fig.tight_layout()

    _mean_E = np.sum(np.abs(sch_b) ** 2 * sch_E) / np.sum(np.abs(sch_b) ** 2)
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(label="Ground-state energy E_1", value=f"{sch_E[0] / _eV * 1e3:.3f} meV", caption=f"exact: {_E_exact[0] / _eV * 1e3:.3f} meV"),
                    mo.stat(label="Mean energy of the packet", value=f"{_mean_E / _eV:.1f} eV", caption=f"hbar^2 kappa^2 / 2M = {(sch_hbar * sch_kappa) ** 2 / (2 * sch_M) / _eV:.1f} eV"),
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
    The lattice energies match the exact levels $n^2\pi^2\hbar^2/(2MD^2)$ for low $n$ and fall
    below them at high $n$ (the same lattice effect as for the string). The packet is built
    mostly from states around $n\approx\kappa D/\pi\approx159$, i.e. its mean energy is the free-particle value
    $\hbar^2\kappa^2/2M$.

    Part (c) of the exercise asks for the real part of $\psi(x,t)$ at an arbitrary time,
    tested at $t=10^{-16}$ s. Part (d) animates it; the slider below plays that role.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt, sch_E, sch_b, sch_hbar, sch_rho, sch_x):
    _t = 1e-16
    _psi = sch_rho @ (sch_b * np.exp(-1j * sch_E * _t / sch_hbar))
    _fig, _ax = plt.subplots(figsize=(8.5, 3.4))
    _ax.plot(sch_x * 1e9, np.real(_psi), color="#1d4ed8", label=r"Re $\psi(x,t)$ at $t=10^{-16}$ s")
    _ax.plot(sch_x * 1e9, np.abs(_psi), color="#888888", lw=1, label=r"$|\psi|$")
    _ax.set(xlabel="x (nm)", ylabel=r"$\psi$ (arbitrary units)")
    _ax.legend(loc="upper right", fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    sch_t_fs = mo.ui.slider(0, 10, value=0.5, step=0.05, label="Time t (femtoseconds)")
    return (sch_t_fs,)


@app.cell(hide_code=True)
def _(mo, np, plt, sch_E, sch_b, sch_hbar, sch_psi0, sch_rho, sch_t_fs, sch_x):
    _t = sch_t_fs.value * 1e-15
    _psi = sch_rho @ (sch_b * np.exp(-1j * sch_E * _t / sch_hbar))
    _fig, _axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    _axes[0].plot(sch_x * 1e9, np.real(_psi), color="#1d4ed8")
    _axes[0].set_ylabel(r"Re $\psi$")
    _axes[1].plot(sch_x * 1e9, np.abs(_psi) ** 2, color="#c2410c")
    _axes[1].set_ylabel(r"$|\psi|^2$")
    _axes[1].set_xlabel("x (nm)")
    _axes[0].set_title(f"Electron in a box, t = {sch_t_fs.value:.2f} fs")
    for _ax in _axes:
        _ax.grid(alpha=0.3)
        _ax.set_ylim(None, None)
    _axes[0].set_ylim(-1.1, 1.1)
    _axes[1].set_ylim(0, 1.2)
    _fig.tight_layout()

    _norm0 = float(np.sum(np.abs(sch_psi0) ** 2))
    mo.vstack(
        [
            sch_t_fs,
            mo.stat(label="Total probability now / at t = 0", value=f"{np.sum(np.abs(_psi) ** 2) / _norm0:.12f}"),
            mo.center(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, np, plt, sch_E, sch_b, sch_hbar, sch_rho, sch_x):
    _t = np.linspace(0, 10e-15, 500)
    _psi_t = sch_rho @ (sch_b[:, None] * np.exp(-1j * np.outer(sch_E, _t) / sch_hbar))
    _fig, _ax = plt.subplots(figsize=(9, 4.2))
    _im = _ax.imshow(np.abs(_psi_t) ** 2, origin="lower", aspect="auto",
                     extent=[0, 10, sch_x[0] * 1e9, sch_x[-1] * 1e9], cmap="magma", vmax=0.4)
    _fig.colorbar(_im, ax=_ax, label=r"$|\psi|^2$")
    _ax.set(xlabel="t (fs)", ylabel="x (nm)", title="Probability density in space and time")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.center(_fig),
            mo.md(
                "**What is going on?** The wave packet is a localised electron moving to the right with the group "
                "velocity $\\hbar\\kappa/M\\approx5.8\\times10^{6}$ m/s. It reaches the wall at "
                "$x=D$ after about 0.9 fs, where it is reflected (the wavefunction must vanish there), "
                "and travels back across the box, reaching the other wall after about 2.6 fs. Meanwhile it spreads out: the packet contains a "
                "range of momenta, so its different components move at slightly different speeds. "
                "Total probability stays equal to one throughout."
            ),
        ]
    )
    return

if __name__ == "__main__":
    app.run()
