import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="PDEs: Triangulations & Crank-Nicolson",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d

    plt.rcParams.update({"figure.dpi": 110, "axes.grid": False})
    return Delaunay, Voronoi, mo, np, plt, sp, spla, voronoi_plot_2d


@app.cell
def _(mo):
    mo.md(r"""
    # Partial Differential Equations
    ## Chapter 9 — Two Deep Dives

    **§9.1.4  Non-square grids** (triangulations & the cotangent Laplacian)
    **§9.3.3  The implicit and Crank–Nicolson methods**

    *Based on Newman, Computational Physics — Chapter 9.*

    ---
    *Tip:* run with `marimo run <this file>`, or `marimo edit` and switch to the
    **Slides** layout in the bottom-right menu. Every section has live widgets.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Roadmap

    | Part | Problem type | Key idea |
    |---|---|---|
    | **A. §9.1.4** | Boundary value (Laplace) | Replace the square lattice by a *triangulation*; the Laplacian becomes the **cotangent form** |
    | **B. §9.3.3** | Initial value (wave eq.) | FTCS is *never* stable → use **implicit**, then the best-of-both **Crank–Nicolson** |

    Common thread: *turn a PDE into a (sparse) linear algebra problem* and ask whether
    the resulting numerics are well behaved.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Part A — §9.1.4 Non-square grids
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why leave the square lattice?

    Square grids work when the boundary is rectangular. Real problems have

    - diagonal or **curved** surfaces,
    - **holes** cut out of the space,
    - regions where we need **more resolution** than others.

    **Solution:** divide space into **triangles** (a *triangulation*).

    - Triangles can have different sizes *and* shapes in the same mesh
    - Small triangles where the solution varies rapidly, large ones elsewhere
    - Works in 3D too (tetrahedra)

    **Price:** the Laplacian is no longer the simple "+1,+1,+1,+1,−4" stencil.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The cotangent Laplacian

    Label the $N$ grid points $i=1\ldots N$ at positions $\mathbf r_i$. In 2D,

    $$
    \frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}
    \;=\;4\,\frac{\sum_{j\in\mathcal N_i}(\cot\alpha_{ij}+\cot\beta_{ij})(\phi_j-\phi_i)}
    {\sum_{j\in\mathcal N_i}(\cot\alpha_{ij}+\cot\beta_{ij})\,|\mathbf r_j-\mathbf r_i|^2}
    \tag{9.17}
    $$

    - $\mathcal N_i$ = neighbours of $i$
    - $\alpha_{ij},\beta_{ij}$ = the two angles **opposite** the $i$–$j$ edge

    **Sanity check:** if point $i$ has four neighbours (up/down/left/right on a square lattice of
    spacing $a$), (9.17) reduces to the usual $\dfrac{1}{a^2}\left[\phi_{\rm up}+\phi_{\rm down}+\phi_{\rm left}+\phi_{\rm right}-4\phi_i\right]$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Matrix form: $L$ and $A$

    Because the cotangents depend only on the mesh, compute them **once**:

    $$
    L_{ij}=\begin{cases}\tfrac12(\cot\alpha_{ij}+\cot\beta_{ij}) & i,j\ \text{adjacent}\\[2pt]
    -\sum_{j\in\mathcal N_i}\tfrac12(\cot\alpha_{ij}+\cot\beta_{ij}) & i=j\\[2pt]
    0 & \text{otherwise}\end{cases}\tag{9.18}
    $$

    $$
    A_{ii}=\frac18\sum_{j\in\mathcal N_i}(\cot\alpha_{ij}+\cot\beta_{ij})\,|\mathbf r_j-\mathbf r_i|^2
    \qquad(\text{diagonal "mass matrix"})\tag{9.19}
    $$

    $$
    \nabla^2\phi \;=\; A^{-1}L\,\phi \tag{9.20}
    $$

    $A$ plays the role the grid spacing $a$ played on the square lattice: it holds the
    *physical* spacing, while $L$ holds the *geometry*.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Solving Laplace's equation on a triangulation

    $\nabla^2\phi=0 \;\Rightarrow\; \sum_j L_{ij}\phi_j + L_{ii}\phi_i = 0$, so

    $$
    \phi_i=-\frac{1}{L_{ii}}\sum_{j\in\mathcal N_i}L_{ij}\phi_j
    =\frac{\sum_j L_{ij}\phi_j}{\sum_j L_{ij}}\tag{9.22–9.23}
    $$

    $\phi_i$ is a **weighted average of its neighbours** — exactly like the square-lattice
    Jacobi method, with weights from the geometry. So we iterate

    $$
    \phi_i' = -\frac{1}{L_{ii}}\sum_{j\in\mathcal N_i}L_{ij}\phi_j \tag{9.24}
    $$

    until convergence, with the boundary values held fixed. Note that $A_{ii}$ **cancels** —
    the mesh spacing doesn't appear in the solution (just as $a$ dropped out in Eq. 9.10).
    """)
    return


@app.cell
def _(Delaunay, np):
    def make_mesh(h, r=0.2, seed=1):
        """Unit square with a circular hole of radius r at the centre.
        Returns points, triangles, kind (0 interior, 1 outer boundary, 2 hole boundary)."""
        rng = np.random.default_rng(seed)
        n = int(round(1 / h))
        t = np.arange(n) / n
        outer = np.vstack(
            [
                np.c_[t, np.zeros(n)],
                np.c_[np.ones(n), t],
                np.c_[1 - t, np.ones(n)],
                np.c_[np.zeros(n), 1 - t],
            ]
        )
        m = max(8, int(round(2 * np.pi * r / h)))
        th = 2 * np.pi * np.arange(m) / m
        inner = np.c_[0.5 + r * np.cos(th), 0.5 + r * np.sin(th)]

        g = (np.arange(1, n)) / n
        gx, gy = np.meshgrid(g, g)
        interior = np.c_[gx.ravel(), gy.ravel()]
        interior = interior + rng.uniform(-0.25, 0.25, interior.shape) / n
        dist = np.hypot(interior[:, 0] - 0.5, interior[:, 1] - 0.5)
        interior = interior[dist > r + 0.7 * h]

        pts = np.vstack([outer, inner, interior])
        kind = np.r_[np.ones(len(outer)), 2 * np.ones(len(inner)), np.zeros(len(interior))].astype(int)

        tri = Delaunay(pts).simplices
        cen = pts[tri].mean(axis=1)
        keep = np.hypot(cen[:, 0] - 0.5, cen[:, 1] - 0.5) > r
        tri = tri[keep]
        # drop degenerate slivers
        p = pts[tri]
        area = 0.5 * np.abs(
            (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
            - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
        )
        tri = tri[area > 1e-12]
        return pts, tri, kind

    return (make_mesh,)


@app.cell
def _(mo):
    h_slider = mo.ui.slider(
        steps=[0.12, 0.09, 0.07, 0.05, 0.04, 0.03], value=0.07, label="target point spacing h", show_value=True
    )
    mo.md("## Demo 1 — A triangulation with a hole  *(Fig. 9.7-style)*")
    mo.vstack([h_slider])
    return (h_slider,)


@app.cell
def _(h_slider, make_mesh, plt):
    pts, tris, kind = make_mesh(h_slider.value)
    _fig, _ax = plt.subplots(figsize=(5.2, 5.2))
    _ax.triplot(pts[:, 0], pts[:, 1], tris, lw=0.6, color="0.35")
    _ax.plot(*pts[kind == 0].T, "o", ms=2, color="tab:blue", label="interior")
    _ax.plot(*pts[kind == 1].T, "s", ms=3, color="tab:red", label="outer boundary")
    _ax.plot(*pts[kind == 2].T, "s", ms=3, color="tab:green", label="hole boundary")
    _ax.set_aspect("equal")
    _ax.set_title(f"{len(pts)} points, {len(tris)} triangles")
    _ax.legend(loc="upper right", fontsize=7)
    _fig.tight_layout()
    _fig
    return kind, pts, tris


@app.cell
def _(mo):
    mo.md(r"""
    ## Building the triangulation: Delaunay

    Given points, which triangles? Only triangles are allowed (no quadrilaterals).
    The standard answer is the **Delaunay triangulation**:

    1. For each grid point build its **Voronoi cell** = all locations closer to that point than to any other.
    2. Join by a line any two points whose Voronoi cells **meet**.

    Voronoi cells meet in threes at corners, so the result is always triangles.
    Writing this yourself is hard — use `scipy.spatial.Delaunay`.

    *(Special case: when four cells meet at one point, e.g. a perfect rectangle, pick one diagonal at random.)*
    """)
    return


@app.cell
def _(mo):
    npts_slider = mo.ui.slider(6, 40, value=14, label="number of random points", show_value=True)
    npts_slider
    return (npts_slider,)


@app.cell
def _(Delaunay, Voronoi, np, npts_slider, plt, voronoi_plot_2d):
    _rng = np.random.default_rng(3)
    _p = _rng.uniform(0.05, 0.95, (npts_slider.value, 2))
    _vor = Voronoi(_p)
    _d = Delaunay(_p)
    _fig, _ax = plt.subplots(figsize=(5.2, 5.2))
    voronoi_plot_2d(
        _vor, ax=_ax, show_vertices=False, show_points=False, line_colors="tab:orange", line_width=1.2, line_alpha=0.8
    )
    _ax.triplot(_p[:, 0], _p[:, 1], _d.simplices, color="tab:blue", lw=1.2)
    _ax.plot(*_p.T, "ko", ms=4)
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)
    _ax.set_aspect("equal")
    _ax.set_title("Voronoi cells (orange)  →  Delaunay triangles (blue)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Computing $L$ and $A$ in code

    For a triangle with corners $\mathbf r_1,\mathbf r_2,\mathbf r_3$ the angle at corner 3 satisfies
    $\mathbf r_{13}\cdot\mathbf r_{23}=|\mathbf r_{13}||\mathbf r_{23}|\cos\alpha_3$
    (Exercise 9.5c). Even simpler: $\cot\alpha = \dfrac{\mathbf u\cdot\mathbf v}{|\mathbf u\times\mathbf v|}$.

    Loop over triangles; each corner contributes $\tfrac12\cot(\text{angle})$ to the
    **opposite** edge. An interior edge gets contributions from both neighbouring triangles —
    that's the $\tfrac12(\cot\alpha+\cot\beta)$.
    """)
    return


@app.cell
def _(np, sp):
    def cotangent_laplacian(pts, tris):
        """Return sparse graph-Laplacian L (Eq. 9.18) and diagonal mass vector A_ii (Eq. 9.19)."""
        I, J, W = [], [], []
        mass = np.zeros(len(pts))
        for tri in tris:
            for k in range(3):
                m, i, j = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
                u, v = pts[i] - pts[m], pts[j] - pts[m]
                cot = np.dot(u, v) / abs(u[0] * v[1] - u[1] * v[0])
                w = 0.5 * cot
                I += [i, j]
                J += [j, i]
                W += [w, w]
                d2 = np.sum((pts[j] - pts[i]) ** 2)
                mass[i] += 0.25 * w * d2
                mass[j] += 0.25 * w * d2
        off = sp.coo_matrix((W, (I, J)), shape=(len(pts),) * 2).tocsr()
        L = off - sp.diags(np.asarray(off.sum(axis=1)).ravel())
        return L.tocsr(), mass

    return (cotangent_laplacian,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Demo 2 — Laplace's equation with a hole

    Inner circle at **1 V**, outer square at **0 V** (Exercise 9.5 setup).
    Jacobi iteration, Eq. (9.24), on the mesh above.
    """)
    return


@app.cell
def _(cotangent_laplacian, kind, np, pts, sp, spla, tris):
    L, mass = cotangent_laplacian(pts, tris)
    _fixed = kind > 0
    _vals = np.where(kind == 2, 1.0, 0.0)

    # --- Jacobi iteration (Eq. 9.24) ---
    _diag = L.diagonal()
    _off = L - sp.diags(_diag)
    phi = np.where(_fixed, _vals, 0.0)
    n_iter = 0
    for n_iter in range(1, 60001):
        _new = phi.copy()
        _upd = -(_off @ phi) / _diag
        _new[~_fixed] = _upd[~_fixed]
        _delta = np.abs(_new - phi).max()
        phi = _new
        if _delta < 1e-6:
            break

    # --- direct solve for comparison ---
    _I = np.where(~_fixed)[0]
    _B = np.where(_fixed)[0]
    _rhs = -(L[_I][:, _B] @ _vals[_B])
    _sol = spla.spsolve(L[_I][:, _I].tocsc(), _rhs)
    phi_direct = _vals.copy()
    phi_direct[_I] = _sol
    jacobi_err = np.abs(phi - phi_direct).max()
    return L, jacobi_err, mass, n_iter, phi


@app.cell
def _(jacobi_err, mo, n_iter, phi, plt, pts, tris):
    _fig, _ax = plt.subplots(figsize=(5.8, 5))
    _c = _ax.tricontourf(pts[:, 0], pts[:, 1], tris, phi, levels=20, cmap="viridis")
    _ax.triplot(pts[:, 0], pts[:, 1], tris, lw=0.3, color="white", alpha=0.5)
    _fig.colorbar(_c, ax=_ax, label=r"$\phi$ (V)")
    _ax.set_aspect("equal")
    _ax.set_title("Potential between a 1 V disc and a grounded box")
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Jacobi converged in **{n_iter}** iterations; "
                f"max difference from a direct sparse solve = **{jacobi_err:.1e}** V."
            ),
        ]
    )
    return


@app.cell
def _(L, mass, mo, np):
    _sym = abs(L - L.T).max()
    _rowsum = np.abs(np.asarray(L.sum(axis=1))).max()
    mo.callout(
        mo.md(
            f"""
            **Sanity checks on $L$:** symmetric (max |L−Lᵀ| = {_sym:.1e}), rows sum to zero
            (max |row sum| = {_rowsum:.1e}), all mass-matrix entries positive: {bool((mass > 0).all())}.

            *Caveat:* very obtuse triangles give negative off-diagonal weights, so mesh quality matters —
            another reason Delaunay (which maximises the minimum angle) is the standard choice.
            """
        ),
        kind="neutral",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Neumann conditions & higher dimensions

    - **Fixed-gradient (Neumann) boundaries:** simply *delete* the boundary points
      from the mesh — nothing can flow through, and the form of $L$ is unchanged.
    - **3D:** divide space into **tetrahedra**; both Delaunay and the Laplacian have natural generalisations.
    - **Cost:** a $d$-dimensional grid with $L$ points per edge has $L^d$ points. Time-dependent
      problems need $T\,L^d$ values - a billion for $L=T=1000$ in 3D.

    ### Takeaways (Part A)

    1. Triangulations handle arbitrary shapes and variable resolution.
    2. The Laplacian becomes the **cotangent form**; geometry lives in the matrices $L$ and $A$.
    3. The relaxation (Jacobi) update is still a weighted neighbour average.
    4. `scipy.spatial.Delaunay` builds the triangulation for you.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part B — §9.3.3 Implicit & Crank–Nicolson methods
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where we left off: the wave equation

    $$
    \frac{\partial^2\phi}{\partial t^2}=v^2\frac{\partial^2\phi}{\partial x^2}
    $$

    Discretise space (spacing $a$) and split into two first-order equations with $\psi=\partial\phi/\partial t$:

    $$
    \frac{d\phi}{dt}=\psi,\qquad
    \frac{d\psi}{dt}=\frac{v^2}{a^2}\big[\phi(x{+}a)+\phi(x{-}a)-2\phi(x)\big]\tag{9.43}
    $$

    **FTCS** (Euler in time, centred in space), Eq. (9.44):

    $$
    \phi(x,t+h)=\phi(x,t)+h\,\psi(x,t),\qquad
    \psi(x,t+h)=\psi(x,t)+h\frac{v^2}{a^2}\big[\phi(x{+}a,t)+\phi(x{-}a,t)-2\phi(x,t)\big]
    $$

    ⚠️ Fig. 9.11: looks fine at 2 ms, bad at 50 ms, overflow at 100 ms. **Numerically unstable.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Recap: von Neumann stability analysis

    Insert one Fourier mode $\begin{pmatrix}\phi\\\psi\end{pmatrix}=\begin{pmatrix}c_\phi(t)\\c_\psi(t)\end{pmatrix}e^{ikx}$.
    Each time-step multiplies the coefficient vector by a $2\times2$ matrix:

    $$
    \mathbf c(t+h)=\mathsf A\,\mathbf c(t),\quad
    \mathsf A=\begin{pmatrix}1&h\\-hr^2&1\end{pmatrix},\quad
    r=\frac{2v}{a}\sin\tfrac12 ka
    $$

    After $m$ steps: $\mathbf c(t+mh)=\alpha_1\lambda_1^m\mathbf v_1+\alpha_2\lambda_2^m\mathbf v_2$.

    > **Stable ⇔ both eigenvalues have $|\lambda|\le 1$.**

    FTCS: $\lambda=1\pm ihr$ so
    $$|\lambda|=\sqrt{1+h^2r^2}\;>\;1\quad\text{for every }h>0,\ k\ne0.$$

    **FTCS is *never* stable for the wave equation** — however small $h$ is.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Idea 1 — The implicit method

    Take the FTCS equations and substitute $h\to-h$ ("step *backwards*"), then shift $t\to t+h$:

    $$
    \phi(x,t)\!\!\!\!\quad\text{(old)}\;=\;\phi(x,t{+}h)-h\,\psi(x,t{+}h)\tag{9.58a}
    $$
    $$
    \psi(x,t)=\psi(x,t{+}h)-h\frac{v^2}{a^2}\big[\phi(x{+}a,t{+}h)+\phi(x{-}a,t{+}h)-2\phi(x,t{+}h)\big]\tag{9.58b}
    $$

    The new values are only defined **implicitly** — we must **solve simultaneous equations**
    (Chapter 6) each step. In matrix form $\mathsf B\,\mathbf c(t+h)=\mathbf c(t)$ with
    $\mathsf B=\begin{pmatrix}1&-h\\hr^2&1\end{pmatrix}$, so $\mathbf c(t+h)=\mathsf B^{-1}\mathbf c(t)$ and

    $$
    \lambda=\frac{1\pm ihr}{1+h^2r^2},\qquad |\lambda|=\frac{1}{\sqrt{1+h^2r^2}}\le 1
    $$

     **Unconditionally stable.**
     But $|\lambda|<1$: every wave **decays**. Real waves don't (they propagate forever).
    The implicit method *over-corrects* FTCS's growth.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Idea 2 — Crank–Nicolson: split the difference

    Average the FTCS (forward) and implicit (backward) equations
    (John Crank & Phyllis Nicolson):

    $$
    \phi(x,t{+}h)-\tfrac12 h\,\psi(x,t{+}h)=\phi(x,t)+\tfrac12 h\,\psi(x,t)\tag{9.64a}
    $$
    $$
    \psi(x,t{+}h)-\frac{hv^2}{2a^2}\big[\phi(x{+}a,t{+}h)+\phi(x{-}a,t{+}h)-2\phi(x,t{+}h)\big]
    =\psi(x,t)+\frac{hv^2}{2a^2}\big[\phi(x{+}a,t)+\phi(x{-}a,t)-2\phi(x,t)\big]\tag{9.64b}
    $$

    Von Neumann analysis gives $\mathsf B\,\mathbf c(t+h)=\mathsf A\,\mathbf c(t)$ with $r=\frac{v}{a}\sin\frac12ka$ and

    $$
    \lambda=\frac{1-h^2r^2\pm 2ihr}{1+h^2r^2}
    \quad\Longrightarrow\quad |\lambda|=\frac{\sqrt{(1-h^2r^2)^2+4h^2r^2}}{1+h^2r^2}=1 .
    $$

     **Waves neither grow nor decay**
    """)
    return


@app.cell
def _(mo):
    h_stab = mo.ui.slider(steps=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4], value=1e-5, label="time step h (s)", show_value=True)
    a_stab = mo.ui.slider(steps=[0.002, 0.005, 0.01, 0.02, 0.05], value=0.01, label="grid spacing a (m)", show_value=True)
    v_stab = mo.ui.slider(10, 400, value=100, step=10, label="wave speed v (m/s)", show_value=True)
    mo.md("## Demo 3 — Amplification factor $|\\lambda|$ per time-step")
    mo.hstack([h_stab, a_stab, v_stab], justify="start")
    return a_stab, h_stab, v_stab


@app.cell
def _(a_stab, h_stab, mo, np, plt, v_stab):
    _h, _a, _v = h_stab.value, a_stab.value, v_stab.value
    _ka = np.linspace(1e-3, np.pi, 400)
    _r_full = (2 * _v / _a) * np.sin(_ka / 2)
    _r_half = (_v / _a) * np.sin(_ka / 2)
    _ftcs = np.sqrt(1 + (_h * _r_full) ** 2)
    _impl = 1 / np.sqrt(1 + (_h * _r_full) ** 2)
    _cn = np.sqrt((1 - (_h * _r_half) ** 2) ** 2 + 4 * (_h * _r_half) ** 2) / (1 + (_h * _r_half) ** 2)

    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.plot(_ka, _ftcs, label="FTCS  (>1: unstable)", color="tab:red")
    _ax.plot(_ka, _impl, label="Implicit  (<1: decays)", color="tab:blue")
    _ax.plot(_ka, _cn, "--", label="Crank–Nicolson  (=1)", color="tab:green", lw=2)
    _ax.axhline(1, color="k", lw=0.5)
    _ax.set_xlabel(r"wavenumber  $ka$")
    _ax.set_ylabel(r"$|\lambda|$")
    _ax.legend()
    _fig.tight_layout()

    _T = 0.01
    _m = _T / _h
    _worst_f = _m * np.log10(_ftcs.max())
    _worst_i = _m * np.log10(_impl.min())
    mo.vstack(
        [
            _fig,
            mo.md(
                f"""
                Over **{_T*1000:.0f} ms** (= {_m:,.0f} steps) the worst-case mode changes in amplitude by a factor of:

                - FTCS: **10^{_worst_f:.2f}**
                - Implicit: **10^{_worst_i:.2f}**
                - Crank–Nicolson: **1** (to rounding error)

                *Try shrinking h: FTCS's per-step growth shrinks, but you need more steps — it never goes away.*
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Implementation: banded linear systems

    Interior grid points $a,2a,\dots,(N{-}1)a$ with $\phi=\psi=0$ at the walls. Stack the unknowns:

    $$
    \mathbf u(t)=\big(\phi_1,\psi_1,\phi_2,\psi_2,\phi_3,\dots\big)^T
    $$

    Then Crank–Nicolson reads $\;\mathsf A\,\mathbf u(t+h)=\mathsf B\,\mathbf u(t)$ with
    $c_1=\tfrac12h,\ c_2=\dfrac{v^2h}{2a^2}$:

    $$
    \mathsf A=\begin{pmatrix}1&-c_1&&&\\2c_2&1&-c_2&&\\&&1&-c_1&\\-c_2&0&2c_2&1&-c_2\\&&&&\ddots\end{pmatrix},\quad
    \mathsf B=\begin{pmatrix}1&c_1&&&\\-2c_2&1&c_2&&\\&&1&c_1&\\c_2&0&-2c_2&1&c_2\\&&&&\ddots\end{pmatrix}
    $$

    Each step: **(1)** $\mathbf v=\mathsf B\mathbf u$ (cheap, banded) **(2)** solve $\mathsf A\mathbf x=\mathbf v$.

    - $\mathsf A$ is **independent of time** → factorise **once**, reuse every step.
    - $\mathsf A$ is **banded** → fast solvers (Sec. 6.1.6 / `banded.py`) are cheap.
    - So CN is more complicated than FTCS but still relatively fast.

    **One code path for all three methods:** blend with $\theta\in[0,1]$, using
    $\mathsf A_\theta$ with $c_1=\theta h$, $c_2=\theta v^2h/a^2$ and
    $\mathsf B_\theta$ with $(1-\theta)$ in place of $\theta$.
    $\theta=0$ → FTCS, $\theta=\tfrac12$ → Crank–Nicolson, $\theta=1$ → implicit.
    """)
    return


@app.cell
def _(np, sp, spla):
    def simulate(theta, h, T, N=100, L=1.0, v=100.0, d=0.1, C=1.0, sigma=0.3, rec=10):
        """Piano string struck at x=d (Exercises 9.9/9.10).
        theta = 0: FTCS, 0.5: Crank–Nicolson, 1: implicit."""
        a = L / N
        x = np.arange(1, N) * a
        n = N - 1
        psi0 = C * x * (L - x) / L**2 * np.exp(-((x - d) ** 2) / (2 * sigma**2))

        def build(s1, s2, sign):
            rows, cols, vals = [], [], []
            for i in range(n):
                p, q = 2 * i, 2 * i + 1  # phi_i, psi_i
                rows += [p, p, q, q]
                cols += [p, q, q, p]
                vals += [1.0, sign * s1, 1.0, -2 * sign * s2]
                if i > 0:
                    rows.append(q)
                    cols.append(2 * (i - 1))
                    vals.append(sign * s2)
                if i < n - 1:
                    rows.append(q)
                    cols.append(2 * (i + 1))
                    vals.append(sign * s2)
            return sp.csc_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n))

        k = v**2 * h / a**2
        A = build(theta * h, theta * k, -1)
        B = build((1 - theta) * h, (1 - theta) * k, +1)
        lu = spla.splu(A)  # factorise once, reuse every step

        u = np.zeros(2 * n)
        u[1::2] = psi0

        def energy(u):
            phi = np.r_[0.0, u[0::2], 0.0]
            psi = u[1::2]
            return 0.5 * a * np.sum(psi**2) + 0.5 * v**2 * a * np.sum((np.diff(phi) / a) ** 2)

        steps = int(round(T / h))
        ts, Es, snaps = [0.0], [energy(u)], [np.r_[0.0, u[0::2], 0.0]]
        with np.errstate(all="ignore"):
            for s in range(1, steps + 1):
                u = lu.solve(B @ u)
                if s % rec == 0:
                    ts.append(s * h)
                    Es.append(energy(u))
                    snaps.append(np.r_[0.0, u[0::2], 0.0])
        xs = np.linspace(0, L, N + 1)
        return xs, np.array(ts), np.array(Es), np.array(snaps)

    return (simulate,)


@app.cell
def _(mo):
    h_sim = mo.ui.dropdown({"1e-5 s": 1e-5, "2e-5 s": 2e-5, "5e-5 s": 5e-5}, value="1e-5 s", label="time step h")
    T_sim = mo.ui.slider(0.01, 0.06, step=0.01, value=0.04, label="simulated time T (s)", show_value=True)
    mo.md(
        r"""
        ## Demo 4 — A struck piano string (Exercises 9.9 & 9.10)

        $L=1$ m, $v=100$ m/s, hammer at $d=10$ cm, $\sigma=0.3$ m, $N=100$ ($a=1$ cm).
        Initial condition $\phi=0$, $\psi(x)=C\frac{x(L-x)}{L^2}e^{-(x-d)^2/2\sigma^2}$.
        We run **all three** methods with the identical code path.
        """
    )
    mo.hstack([h_sim, T_sim], justify="start")
    return T_sim, h_sim


@app.cell
def _(T_sim, h_sim, simulate):
    res = {name: simulate(th, h_sim.value, T_sim.value) for name, th in
           [("FTCS", 0.0), ("Implicit", 1.0), ("Crank–Nicolson", 0.5)]}
    return (res,)


@app.cell
def _(mo, plt, res):
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _colors = {"FTCS": "tab:red", "Implicit": "tab:blue", "Crank–Nicolson": "tab:green"}
    for _name, (_xs, _ts, _E, _snaps) in res.items():
        _ax.semilogy(_ts * 1000, _E / _E[0], label=_name, color=_colors[_name], lw=2)
    _ax.set_xlabel("time (ms)")
    _ax.set_ylabel("energy / initial energy")
    _ax.set_title("Total wave energy:  FTCS explodes, implicit decays, CN conserves")
    _ax.legend()
    _ax.set_ylim(1e-3, 1e12)
    _fig.tight_layout()
    mo.vstack([mo.md("### Energy history"), _fig])
    return


@app.cell
def _(mo, res):
    _nrec = len(res["Crank–Nicolson"][1])
    snap_slider = mo.ui.slider(0, _nrec - 1, value=min(_nrec - 1, 10), label="snapshot index (time)", show_value=True, full_width=True)
    mo.md("### Snapshots of the string")
    snap_slider
    return (snap_slider,)


@app.cell
def _(plt, res, snap_slider):
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.2))
    _colors = {"FTCS": "tab:red", "Implicit": "tab:blue", "Crank–Nicolson": "tab:green"}
    _k = snap_slider.value
    for _ax, (_name, (_xs, _ts, _E, _snaps)) in zip(_axes, res.items()):
        _ax.plot(_xs, _snaps[_k], color=_colors[_name])
        _ax.set_title(f"{_name},  t = {_ts[_k] * 1000:.1f} ms")
        _ax.set_xlabel("x (m)")
    _axes[0].set_ylabel(r"$\phi(x,t)$")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, res):
    _cn = res["Crank–Nicolson"][2]
    _drift = abs(_cn[-1] / _cn[0] - 1)
    _im = res["Implicit"][2]
    _ft = res["FTCS"][2]
    mo.callout(
        mo.md(
            f"""
            **Final / initial energy** over the run:

            | Method | E(T)/E(0) |
            |---|---|
            | FTCS | {_ft[-1]/_ft[0]:.3g} |
            | Implicit | {_im[-1]/_im[0]:.3g} |
            | Crank–Nicolson | {_cn[-1]/_cn[0]:.6f}  (drift {_drift:.1e}) |

            Watch the string in the snapshots: with CN the pulse keeps travelling and reflecting at the
            fixed ends with constant amplitude; the implicit pulse is smeared out and fades.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparison

    | | FTCS | Implicit | Crank–Nicolson |
    |---|---|---|---|
    | Equation for new values | explicit | simultaneous linear system | simultaneous linear system |
    | $\lvert\lambda\rvert$ (wave eq.) | $\sqrt{1+h^2r^2}>1$ | $1/\sqrt{1+h^2r^2}<1$ | exactly $1$ |
    | Behaviour | amplitudes **blow up** | amplitudes **decay** | amplitudes **preserved** |
    | Stability | never | unconditional | unconditional |
    | Cost per step | cheapest | banded solve | banded solve (matrix $\mathsf A$ reused) |
    | Right for waves? | no | no (too damped) | yes |

    **Note:** FTCS *is* fine for the diffusion equation, provided $h\le a^2/2D$ (Eq. 9.48).
    Stability is a property of the *method and the equation together*.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    **§9.1.4 — Non-square grids**
    - Triangulate; build with **Delaunay** (Voronoi-cell neighbours).
    - Cotangent Laplacian: $L$ (geometry) and $A$ (mass); Jacobi becomes $\phi_i'=-\frac1{L_{ii}}\sum_jL_{ij}\phi_j$.

    **§9.3.3 — Implicit & Crank–Nicolson**
    - FTCS is never stable for waves; implicit is stable but damps; **CN is stable *and* conserving**.
    - Cost: a banded linear solve each step, with a constant matrix factorised once.

    **Exercises to try:** 9.4, 9.5 (triangulations) · 9.9, 9.10, 9.13 (wave & Schrödinger equations).
    """)
    return


if __name__ == "__main__":
    app.run()
