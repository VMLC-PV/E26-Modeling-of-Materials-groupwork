# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.11.2",
#     "numpy>=2.5.3",
#     "pandas>=3.0.6",
#     "scipy>=1.18.1",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(layout_file="layouts/lec_4_exercise_6.10.slides.json")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import pandas as pd
    from scipy.integrate import quad


    return mo, np, pd, plt


@app.cell
def _(np, plt):
    plt.figure(figsize=(5, 5))
    well = plt.gca()

    def V(x):
        return 0.8 * np.sin(x) + 0.35 * np.sin(2*x - 1) + 0.15*x + 2

    L = 5
    x = np.linspace(0, L, 1000)
    v = V(x)

    plt.plot(x, v, color='red', label='h(x)=$2x^3 + 2x^2 - 3x + 3$')

    well.axes.get_xaxis().set_ticks([])
    well.axes.get_yaxis().set_ticks([])
    well.spines['top'].set_visible(False)

    well.set_xlim([0, L])
    well.set_ylim([0, 10])

    plt.xlabel('L') 
    plt.ylabel('V(x)')

    plt.show()
    return well, x


@app.cell
def _(mo):
    exercise_6_10_title = mo.md(
        r"""
        ## Exercise 6.10: Asymmetric Quantum Well"""
    )

    exercise_6_10_text = mo.md(
        r"""
        A particle with mass $M$ in a 1D well with varying potential within the well. 

        $\psi(x)$ obeys the time-independent Schrödinger equation, 
        $$\hat{H}\psi(x)=E\psi(x)$$

        $$ \left[- \frac{\hbar^2}{2M} \frac{d^2}{dx^2} + V(x) \right] \psi(x) = E\psi(x) $$

        We assume that the well's walls are infinitely high so the wavefunction outside of the walls is zero, $\psi = 0$.
        With this the wavefunction can be expressed as the following Fourier series:

        $$\psi(x) = \sum_{n=1}^\infty \psi_n \sin {\frac{\pi n x}{L}}$$"""
    )
    return exercise_6_10_text, exercise_6_10_title


@app.cell
def _(exercise_6_10_text, exercise_6_10_title, mo, well):
    mo.vstack([exercise_6_10_title, well, exercise_6_10_text])
    return


@app.cell
def _(mo):
    text_a_1 = mo.md(
        r"""
        Knowing the wavefunction as a Fourier series: 
        $$\psi(x) = \sum_{n=1}^\infty \psi_n \sin {\frac{\pi n x}{L}}$$

        We can understand it as: 
        * $\psi_n$ are coefficients of of the wavefunction.
        * the equation is comparable to writing a vector as:
        $$ \mathbf{v} = v_1\mathbf{e_1} + v_2\mathbf{e_2} + ... + v_n\mathbf{e_n}$$
        However instead of vectors $\mathbf{e_n}$ we have the function
        $$\sin {\frac{\pi n x}{L}}$$
        """
    )
    return (text_a_1,)


@app.cell
def _(mo, text_a_1):
    mo.vstack([text_a_1])
    return


@app.cell
def _(mo):
    # https://www.youtube.com/watch?v=cXNEIHpnYlg
    # https://www.youtube.com/watch?v=UXoj6kzl384

    text_a_2 = mo.md(
        r"""
        ## Question a (orthogonality)
        The exercise refers to the following equation:

        $$
        \int_0^L \sin\left(\frac{\pi m x}{L}\right) \sin\left(\frac{\pi n x}{L}\right)\,dx
        =
        \begin{cases}
        \frac{L}{2} & \text{if } m=n,\\
        0 & \text{otherwise}.
        \end{cases}
        $$

        Which is the orthogonality relation of sine
        """
    )
    return (text_a_2,)


@app.cell
def _(mo):
    m_slider = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=2,
        label="m value:",
    )

    n_slider = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=3,
        label="n value:",
    )

    L_slider = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=5,
        label="L value:",
    )
    return L_slider, m_slider, n_slider


@app.cell
def _(L_slider, m_slider, np):
    def f_orth(x_A):
        return np.sin((np.pi*m*x_A)/L_orth)

    m = m_slider.value
    L_orth = L_slider.value
    x_orth = np.linspace(0, L_orth, 1000)
    y_orth = f_orth(x_orth)
    return L_orth, x_orth, y_orth


@app.cell
def _(L_orth, n_slider, np, x_orth):
    def g_orth(x_B):
        return np.sin((np.pi*n*x_B)/L_orth)

    n = n_slider.value
    yy_orth = g_orth(x_orth)
    return (yy_orth,)


@app.cell
def _(mo, np, x_orth, y_orth, yy_orth):
    fg = y_orth * yy_orth
    integral_val = np.trapezoid(fg, x_orth)

    orthogonal_integral = mo.md(
        f"""
        $$\int_0^L f(x)g(x)\,dx = {integral_val} $$
        """
    )
    return fg, orthogonal_integral


@app.cell
def _(L_orth, fg, mo, plt, x, x_orth, y_orth, yy_orth):
    #plt.figure(figsize=(5, 5))

    plt.plot(x_orth, y_orth, color='red', label='f(x)')
    plt.plot(x_orth, yy_orth, color='green', label='g(x)')

    plt.plot(x, fg, color="#8AB4E8", linestyle='dashed', label="f(x)g(x)")

    plt.fill_between(
        x_orth, fg, 0,
        where=(fg >= 0),
        color="#8AB4E8",
        alpha=0.25
    )

    plt.fill_between(
        x_orth, fg, 0,
        where=(fg < 0),
        color="#8AB4E8",
        alpha=0.25
    )

    plt.gca().set_xlim([0, L_orth])

    plt.title('Orthogonality')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    orthogonality = mo.mpl.interactive(plt.gcf())
    return (orthogonality,)


@app.cell
def _(
    L_slider,
    m_slider,
    mo,
    n_slider,
    orthogonal_integral,
    orthogonality,
    text_a_2,
):
    mo.vstack([text_a_2, 
    mo.hstack([L_slider, m_slider, n_slider], align="center", justify="center"),
    orthogonal_integral, orthogonality])
    return


@app.cell
def _(mo):
    text_a_3 = mo.md(
        r"""
        ## Question a (Schrödinger Eq.)
        The exercise gives the following equation: 
        $$\sum_{n=1}^\infty \psi(x) \int_0^L \sin\left(\frac{\pi m x}{L}\right) \hat{H}\sin\left(\frac{\pi n x}{L}\right)\,dx = \frac{1}{2} LE\psi_m$$

        To find this we can work from the Schrödinger's equation 
        knowing the definition of the wavefunction and the orthogonality relation:

        $$\hat{H}\psi = E\psi$$
        $$Substitute &rarr; \psi(x) = \sum_{n=1}^\infty \psi_n \sin {\frac{\pi n x}{L}}$$

        $$ \hat{H} \sum_{n=1}^\infty \psi_n \sin {\frac{\pi n x}{L}} = E \sum_{n=1}^\infty \psi_n \sin {\frac{\pi n x}{L}}$$

        As $\hat{H}$ is a linear operator and operates on $\psi:$

        $$\sum_{n=1}^\infty \psi_n \hat{H} \sin {\frac{\pi n x}{L}} = E \sum_{n=1}^\infty \psi_n \sin {\frac{\pi n x}{L}}$$

        """
    )
    return (text_a_3,)


@app.cell
def _(mo, text_a_3):
    mo.vstack([text_a_3])
    return


@app.cell
def _(mo):
    text_a_4 = mo.md(
        r"""
        ## Question a (using orthogonality on Schrödinger Eq)

        So we obtained the following Schrödinger formula: 
        $$\sum_{n=1}^\infty \psi_n \hat{H} \sin {\frac{\pi n x}{L}} = E \sum_{n=1}^\infty \psi_n \sin {\frac{\pi n x}{L}}$$

        From this we can then use the orthogonality relation by multiplying by $\sin{\frac{\pi m x}{L}}$ 
        and integrating from 0 to L:

        $$ \sum_{n=1}^\infty \psi_n \int_0^L \sin\left(\frac{\pi m x}{L}\right) \hat{H} \sin\left(\frac{\pi n x}{L}\right)\,dx = E \sum_{n=1}^\infty \psi_n \int_0^L \sin\left(\frac{\pi m x}{L}\right) \sin\left(\frac{\pi n x}{L}\right)\,dx$$

        From the orthogonality rule everything on the right hand side will cancel except for $\frac{L}{2}E\psi_n$ :

        $$\sum_{n=1}^\infty \psi(x) \int_0^L \sin\left(\frac{\pi m x}{L}\right) \hat{H}\sin\left(\frac{\pi n x}{L}\right)\,dx = \frac{1}{2} LE\psi_m$$

        """
    )
    return (text_a_4,)


@app.cell
def _(mo, text_a_4):
    mo.vstack([text_a_4])
    return


@app.cell
def _(mo):
    text_a_5 = mo.md(
        r"""
        ## Question a (Hamiltonian operator to matrix)
        The exercise defines the following matrix **H**:
        $$\textbf{H}_{mn} = \frac{2}{L} \int_0^L \sin\left(\frac{\pi m x}{L}\right) \hat{H}\sin\left(\frac{\pi n x}{L}\right)\,dx $$
        $$= \frac{2}{L} \int_0^L \sin\left(\frac{\pi m x}{L}\right) \left[ - \frac{\hbar^2}{2M} \frac{d^2}{dx^2} + V(x) \right] \sin\left(\frac{\pi n x}{L}\right)\,dx$$

        From which we can get:
        $$\sum_{n=1}^\infty \textbf{H}_{mn} \psi_n = E\psi_n$$

        Which as the exercise asks shows that the Schrödinger equation can be written as:
        $$\textbf{H}\mathbf{\psi} = E\mathbf{\psi}$$

        $$ \begin{bmatrix}
            H_{11} & H_{12} & H_{13} & \dots \\
            H_{21} & H_{22} & H_{23} & \dots \\
            \vdots & \vdots & \vdots & \ddots
        \end{bmatrix} \begin{bmatrix}
            \psi_1 \\
            \psi_2 \\
            \vdots
        \end{bmatrix} = E \begin{bmatrix}
            \psi_1 \\
            \psi_2 \\
            \vdots
        \end{bmatrix}$$

        Where $\mathbf{\psi}$ is the eigenvector of **H** with the eigenvalue of E. 
        Calculating the eigenvalues will give the allowed energy states in the well:
        $$\textbf{H}\mathbf{\psi} = E\mathbf{\psi}$$
        $$\textbf{H}\mathbf{\psi} - E\mathbf{\psi} = 0$$
        $$\textbf{H}\mathbf{\psi} - E \textbf{I} \mathbf{\psi} = 0$$
        $$(\textbf{H} - E \textbf{I})\mathbf{\psi} = 0$$
        $$&rarr; det(\textbf{H} - E \textbf{I})=0$$

        """
    )
    return (text_a_5,)


@app.cell
def _(mo, text_a_5):
    mo.vstack([text_a_5])
    return


@app.cell
def _(mo):
    text_a_6 = mo.md(
        r"""
        ## Question a (Kinetic Energy)
        The $\hat{H}$ is the sum of the kinetic and potential energy of the system.
        $$\hat{H} = -\frac{\hbar^2}{2M} \frac{d^2}{dx^2} + V(x)$$

        Applying this to the Hamiltonian matrix **H** as before: 

        $$\mathbf{H}_{mn} = \frac{2}{L} \int_0^L \sin\left(\frac{\pi m x}{L}\right) \left[ - \frac{\hbar^2}{2M} \frac{d^2}{dx^2} + V(x) \right] \sin\left(\frac{\pi n x}{L}\right)\,dx$$

        We can look at the kinetic energy portion, where: 
        $$ \frac{d^2}{dx^2} \sin{\frac{\pi n x}{L}} = - \left(\frac{\pi n}{L}\right)^2 \sin{\frac{\pi n x}{L}}$$

        Therefore, adding back the constant:
        $$- \frac{\hbar^2}{2M} \frac{d^2}{dx^2} \sin{\frac{\pi n x}{L}} = \frac{\hbar^2 \pi^2 n^2}{2ML^2}\sin{\frac{\pi n x}{L}}$$

        Using orthogonality by multiplying with $\sin{\frac{\pi m x}{L}}$ and integrating from 0 to L. Then when m=n we get: 
        $$K = \frac{\hbar^2 \pi^2 n^2}{2ML^2}$$

        """
    )
    return (text_a_6,)


@app.cell
def _(mo, text_a_6):
    mo.vstack([text_a_6])
    return


@app.cell
def _(mo):
    text_a_7 = mo.md(
        r"""
        ## Question a (Potential Energy)
        The potential energy portion of the Hamiltonian depends on the asymmetric potential of the well:

        $$ V = \frac{2}{L} \int_0^L \sin\left(\frac{\pi m x}{L}\right) V(x) \sin\left(\frac{\pi n x}{L}\right)\,dx $$

        Giving a total Hamiltonian equation as: 
        $$ H_{mn} = \frac{\hbar^2 \pi^2 n^2}{2ML^2} + \frac{2}{L} \int_0^L \sin\left(\frac{\pi m x}{L}\right) V(x) \sin\left(\frac{\pi n x}{L}\right)\,dx$$
        when m = n

        """
    )
    return (text_a_7,)


@app.cell
def _(mo, text_a_7):
    mo.vstack([text_a_7])
    return


@app.cell
def _(mo):
    text_b_1 = mo.md(
        r"""
        ## Question b (the system)
        Goal: evaluate $\mathbf{H_{mn}}$ integral and find general expression

        We are given: 
        * $V(x) = \frac{ax}{L}$
        * Well width = 5 angstroms 
        * a = 10 eV
        * Particle is an electron, therefore: 
            * m = $9.1094*10^{-31}$ [kg]
            * e = $1.6022*10^{-19}$ [C]
        * The solution for the integral:
        $$ \int_0^L x\sin{\frac{\pi m x}{L}} \sin{\frac{\pi n x}{L}}\,dx = \begin{cases} 0 & \text{if m} ≠ \text{n and they are both even or odd},\\ -\left( \frac{2L}{\pi}\right)^2 \frac{mn}{(m^2 - n^2)^2} & \text{if m} ≠ \text{n and one is even and the other odd}, \\ \frac{L^2}{4} & \text{if m=n}. \end{cases} $$
        """
    )
    return (text_b_1,)


@app.cell
def _(mo, text_b_1):
    mo.vstack([text_b_1])
    return


@app.cell
def _(mo):
    text_b_2 = mo.md(
        r"""
        ## Question b (kinetic & potential energy)

        As found in question a the kinetic energy term is (if m = n): 
        $$K = \frac{\hbar^2 \pi^2 n^2}{2ML^2}$$

        We are given the potential energy $V(x) = \frac{ax}{L}$, which we can substitute to define the potential energy term:
        $$V = \frac{2a}{L^2} \int_0^L x \sin\left(\frac{\pi m x}{L}\right) \sin\left(\frac{\pi n x}{L}\right)\,dx $$

        With the exercise's given definition for the integral, we can determine:
        * if m ≠ n and they are both even or odd: 
        $$ V = 0$$
        * if m ≠ n and one is even and the other odd
        $$ V = \frac{2a}{L^2} \left[ -\left( \frac{2L}{\pi}\right)^2 \frac{mn}{(m^2 - n^2)62} \right]$$
        $$ = - \frac{8amn}{\pi^2(m^2 - n^2)^2}$$
        * if m = n:
        $$ V = \frac{2a}{L^2} \frac{L^2}{4} = \frac{a}{2}$$


        """
    )
    return (text_b_2,)


@app.cell
def _(mo, text_b_2):
    mo.vstack([text_b_2])
    return


@app.cell
def _(mo):
    text_b_3 = mo.md(
        r"""
        ## Question b (**H**$_{mn}$)   
        Adding the potential and kinetic terms together the final expression for the Hamiltonian becomes:
        * if m ≠ n and they are both even or odd
        $$H_{mn} = 0$$
        * if m ≠ n and one is even and the other odd
        $$- \frac{8amn}{\pi^2(m^2 - n^2)^2}$$ 
        * if m = n
        $$\frac{\hbar^2 \pi^2 n^2}{2ML^2} + \frac{a}{2}$$

        """
    )
    return (text_b_3,)


@app.cell
def _(mo, text_b_3):
    mo.vstack([text_b_3])
    return


@app.cell
def _(np):
    # Constants
    hbar = 1.054571817e-34       # J s
    M_b = 9.1094e-31               # kg
    L_b = 5e-10                    # m
    a_b = 10.0                     # eV

    # Conversion
    eV = 1.6022e-19              # J

    # Kinetic-energy coefficient in eV
    E0 = (hbar**2 * np.pi**2 / (2 * M_b * L_b**2)) / eV

    def H_b(m_b, n_b):
        # Diagonal - m = n
        if m_b == n_b:
            return E0 * n_b**2 + a_b/2

        # Same parity -> zero
        if m_b % 2 == n_b % 2:
            return 0.0

        # Opposite parity
        return -8 * a_b * m_b * n_b / (np.pi**2 * (m_b**2 - n_b**2)**2)

    return H_b, L_b


@app.cell
def _(mo):
    question_b_title = mo.md(
        r"""
        ## Question b (finding $\mathbf{H_{mn}}$)
        """
    )

    m_bb_slider = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=1,
        label="m value:",
    )

    n_bb_slider = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=1,
        label="n value:",
    )
    return m_bb_slider, n_bb_slider, question_b_title


@app.cell
def _(H_b, m_bb_slider, mo, n_bb_slider):
    m_bb = m_bb_slider.value
    n_bb = n_bb_slider.value

    question_b = mo.md(
        f"""
        **H** [{m_bb},{n_bb}] = 
        $$ {H_b(m_bb,n_bb):.6f} $$
        """
    )
    return (question_b,)


@app.cell
def _(m_bb_slider, mo, n_bb_slider, question_b, question_b_title):
    mo.vstack([question_b_title, mo.hstack([m_bb_slider, n_bb_slider]), question_b])
    return


@app.cell
def _(mo):
    question_c_title = mo.md(
        r"""
        ## Question c (finding $\mathbf{H_{mn}}$ and E for 10x10)

        Using np.linalg.eigvalsh to find the eigenvalues
        """
    )
    return (question_c_title,)


@app.cell
def _(mo, question_c_title):
    mo.vstack([question_c_title])
    return


@app.cell
def _(H_b, np):
    N_c = 10
    Hmatrix = np.zeros((N_c, N_c))

    for m_c1 in range(1, N_c + 1):
        for n_c1 in range(1, N_c + 1):
            Hmatrix[m_c1-1, n_c1-1] = H_b(m_c1, n_c1)

    print(Hmatrix)
    return


@app.cell
def _(H_b, np):
    N = 10

    Hmatrix_c = np.zeros((N, N))

    for m_c in range(1, N+1):
        for n_c in range(1, N+1):
            Hmatrix_c[m_c-1, n_c-1] = H_b(m_c, n_c)

    # Find eigenvalues
    energies = np.linalg.eigvalsh(Hmatrix_c)

    print("Energy eigenvalues:")
    for E in energies:
        print(f"{E:.6f} eV")
    return (energies,)


@app.cell
def _(H_b, np):
    N_d1 = 100
    Hmatrix_d = np.zeros((N_d1, N_d1))

    for m_d in range(1, N_d1+1):
        for n_d in range(1, N_d1+1):
            Hmatrix_d[m_d-1, n_d-1] = H_b(m_d, n_d)

    # Find eigenvalues
    energies_d = np.linalg.eigvalsh(Hmatrix_d)

    print("First 10 energy eigenvalues:")
    for D in energies_d[:10]:
        print(f"{D:.6f} eV")
    return Hmatrix_d, N_d1, energies_d


@app.cell
def _(energies, energies_d, mo, np, pd):
    energy_diff = np.abs(energies - energies_d[:10])

    comparison = pd.DataFrame({
        "10x10 energies (eV)": energies,
        "100x100 energies (eV)": energies_d[:10],
        "absolute difference": energy_diff
    })

    #print(table)

    table = mo.ui.table(
        data=comparison,
        # use pagination when your table has many rows
        label="Comparison of exercise C and D",
    )
    return (table,)


@app.cell
def _(mo):
    question_d_title = mo.md(
        r"""
        ## Question D (Comparison)
        """
    )

    question_d_text = mo.md(
        r"""
        The 10x10 array will only have 10 basis states compared to the 100x100 array's 100 basis states. 
        This gives the Hamiltonian calculation in the 100x100 more information to determine each energy.
        As the number of basis states increases the energy converges, affecting the accuracy. 
        """
    )
    return question_d_text, question_d_title


@app.cell
def _(mo, question_d_text, question_d_title, table):
    mo.vstack([question_d_title, table, table.value, question_d_text])
    return


@app.cell
def _(H_b, np):
    N_20 = 20

    Hmatrix_20 = np.zeros((N_20, N_20))

    for m_20 in range(1, N_20+1):
        for n_20 in range(1, N_20+1):
            Hmatrix_20[m_20-1, n_20-1] = H_b(m_20, n_20)

    # Find eigenvalues
    energies_20 = np.linalg.eigvalsh(Hmatrix_20)
    return (energies_20,)


@app.cell
def _(H_b, np):
    Hmatrix_40 = np.zeros((40, 40))

    for m_40 in range(1, 40+1):
        for n_40 in range(1, 40+1):
            Hmatrix_40[m_40-1, n_40-1] = H_b(m_40, n_40)

    # Find eigenvalues
    energies_40 = np.linalg.eigvalsh(Hmatrix_40)
    return (energies_40,)


@app.cell
def _(H_b, np):
    Hmatrix_60 = np.zeros((60, 60))

    for m_60 in range(1, 60+1):
        for n_60 in range(1, 60+1):
            Hmatrix_60[m_60-1, n_60-1] = H_b(m_60, n_60)

    # Find eigenvalues
    energies_60 = np.linalg.eigvalsh(Hmatrix_60)
    return (energies_60,)


@app.cell
def _(H_b, np):
    Hmatrix_80 = np.zeros((80, 80))

    for m_80 in range(1, 80+1):
        for n_80 in range(1, 80+1):
            Hmatrix_80[m_80-1, n_80-1] = H_b(m_80, n_80)

    # Find eigenvalues
    energies_80 = np.linalg.eigvalsh(Hmatrix_80)
    return (energies_80,)


@app.cell
def _(
    energies,
    energies_20,
    energies_40,
    energies_60,
    energies_80,
    energies_d,
    mo,
    pd,
):
    convergence = pd.DataFrame({
        "Matrix size": [10, 20, 40, 60, 80, 100],
        "E1 (eV)": [
            energies[0],
            energies_20[0],
            energies_40[0],
            energies_60[0],
            energies_80[0],
            energies_d[0]
        ],
        "E2 (eV)": [
            energies[1],
            energies_20[1],
            energies_40[1],
            energies_60[1],
            energies_80[1],
            energies_d[1]
        ],
        "E3 (eV)": [
            energies[2],
            energies_20[2],
            energies_40[2],
            energies_60[2],
            energies_80[2],
            energies_d[2]
        ]
    })


    table_conv = mo.ui.table(
        data=convergence,
        # use pagination when your table has many rows
        label="Comparison of different energy basis",
    )

    question_d_convergence = mo.md(
        r"""
        ## Question D (Convergence of different energy basis)
        """
    )
    return question_d_convergence, table_conv


@app.cell
def _(mo, question_d_convergence, table_conv):
    mo.vstack([question_d_convergence, table_conv, table_conv.value])
    return


@app.cell
def _(mo):
    question_e = mo.md(
        r"""
        ## Question E (wavefunctions)

        The normalised infinite square well is: 
        $$ \psi(x) = \sqrt{\frac{2}{L}} \sin{\frac{n \pi x}{L}}$$

        That satisfies the normalisation requirement
        $$ \int_0^L |\psi(x)|^2 \,dx = 1$$

        As discussed previously we have:
        $\begin{bmatrix}
            H_{11} & H_{12} & H_{13} & \dots \\
            H_{21} & H_{22} & H_{23} & \dots \\
            \vdots & \vdots & \vdots & \ddots
        \end{bmatrix} \begin{bmatrix}
            \psi_1 \\
            \psi_2 \\
            \vdots
        \end{bmatrix} = E \begin{bmatrix}
            \psi_1 \\
            \psi_2 \\
            \vdots
        \end{bmatrix}$

        Where $\mathbf{\psi}$ is the eigenvector of **H** with the eigenvalue of E. 

        Which will give us:

        $$\psi(x) = \sqrt{\frac{2}{L}} \left(\psi_1 \sin{\frac{\pi x}{L}} + \psi_2 \sin{\frac{2\pi x}{L}} + \psi_3 \sin{\frac{3\pi x}{L}} + ...\right)$$
        """
    )
    return (question_e,)


@app.cell
def _(mo, question_e):
    mo.vstack([question_e])
    return


@app.cell
def _(Hmatrix_d, L_b, N_d1, energies_d, np, x):
    print("Ground-state energy =", energies_d[0], "eV")
    print("1st excited energy  =", energies_d[1], "eV")
    print("2nd excited energy  =", energies_d[2], "eV")

    x_e = np.linspace(0, L_b, 1000)

    # --------------------------------------------------
    # Basis functions
    # --------------------------------------------------

    basis = np.zeros((N_d1, len(x)))

    for n_e in range(1, N_d1 + 1):
        basis[n_e-1] = np.sqrt(2/L_b) * np.sin(n_e*np.pi*x_e/L_b)

    energies_e, eigenvectors = np.linalg.eigh(Hmatrix_d)

    psi0 = eigenvectors[:, 0] @ basis
    psi1 = eigenvectors[:, 1] @ basis
    psi2 = eigenvectors[:, 2] @ basis

    # Normalise 
    psi0 /= np.sqrt(np.trapezoid(np.abs(psi0)**2, x_e))
    psi1 /= np.sqrt(np.trapezoid(np.abs(psi1)**2, x_e))
    psi2 /= np.sqrt(np.trapezoid(np.abs(psi2)**2, x_e))

    print("\nNormalization:")
    print(np.trapezoid(np.abs(psi0)**2, x_e))
    print(np.trapezoid(np.abs(psi1)**2, x_e))
    print(np.trapezoid(np.abs(psi2)**2, x_e))

    return psi0, psi1, psi2, x_e


@app.cell
def _(mo, np, plt, psi0, psi1, psi2, x_e):
    plt.figure(figsize=(8, 5))

    plt.plot(x_e * 1e9, np.abs(psi0)**2, label="Ground state")

    plt.plot(x_e * 1e9, np.abs(psi1)**2, label="1st excited state")

    plt.plot(x_e * 1e9, np.abs(psi2)**2, label="2nd excited state")

    plt.xlabel("$x [nm]$")
    plt.ylabel(r"$|\psi(x)|^2$")
    plt.title("Probability Density in the Well")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    probability_density = mo.mpl.interactive(plt.gcf())
    return (probability_density,)


@app.cell
def _(mo, probability_density):
    mo.vstack([probability_density])
    return


@app.cell
def _(mo, np, plt, psi0, psi1, psi2, x_e):
    E_e = [5.8, 11.2, 18.7]

    scale = 4.0

    probability_e = [np.abs(psi0)**2, np.abs(psi1)**2, np.abs(psi2)**2]

    for energy, density in zip(E_e, probability_e):
        density = density / np.max(density)
        #plt.plot(x_e, energy + scale*(density -0.5))
    
        plt.axhline(energy, color="gray", linestyle="--", alpha=0.4, zorder=1)
        plt.plot(x_e, energy + scale * density, linewidth=2, zorder=2)

    plt.xlabel("$x [nm]$")
    plt.ylabel(r"$|\psi(x)|^2$, $E$")
    plt.title("Probability Density in the Well")

    probability_density_E = mo.mpl.interactive(plt.gcf())

    return (probability_density_E,)


@app.cell
def _(mo, probability_density_E):
    mo.vstack([probability_density_E])
    return


if __name__ == "__main__":
    app.run()
