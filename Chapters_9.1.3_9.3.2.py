import marimo

__generated_with = "0.25.1"
app = marimo.App(layout_file="layouts/Chapters_9.1.3_9.3.2.slides.json")


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    from numpy import empty,zeros,max
    import numpy as np

    return Circle, empty, max, mo, plt, zeros


@app.cell
def _(Circle, mo, plt):
    slide_1_title = mo.md(
        r"""
    ## Chapter 9.1.3 - **Types of Boundary Conditions**

        """
    )
    slide_1 = mo.md(
        r"""
    **Boundary value problem** is when the (partial) differential equation is constrained to boundary values.

        """
    )
    slide_1_1 = mo.md(
        r"""
    * Dirichlet - **fixed-value** 
    * Neumann - **fixed-derivative / fixed-gradient**

        """
    )

    fig, ax = plt.subplots(figsize=(3, 3))
    circle = Circle((0, 0), 0.5, color='green', fill=True)
    ax.add_patch(circle)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()
    circlething = plt.gca()

    mo.vstack([slide_1_title, mo.hstack([slide_1, circlething]), slide_1_1 ])
    return


@app.cell
def _(mo, plt):
    fixed, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(4, 2))
    fixed.suptitle('Fixed-Value Boundaries')
    fixed_x = [0, -1, 0, 1, 0]
    fixed_y = [0, -1, -1, -1, -2]

    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticks([])
    ax1.plot(fixed_x, fixed_y, 'ro')
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.axhline(0, color='black', linestyle='dashed')

    ax1.text(0, -0.15, '0', fontsize=12, ha='center', va='top')
    ax1.text(-0.85, -1, '1', fontsize=12, ha='left', va='top')
    ax1.text(0.85, -1, '1', fontsize=12, ha='right', va='top')
    ax1.text(0, -2.15, '1', fontsize=12, ha='center', va='top')
    ax1.text(0, -1.15, '-4', fontsize=12, ha='center', va='top')

    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    ax2.plot(fixed_x, fixed_y, 'ro')
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.axhline(0, color='black', linestyle='dashed')
    ax2.axvline(-1, color='black', linestyle='dashed')

    ax2.text(0, -0.15, '0', fontsize=12, ha='center', va='top')
    ax2.text(-0.85, -1, '0', fontsize=12, ha='left', va='top')
    ax2.text(0.85, -1, '1', fontsize=12, ha='right', va='top')
    ax2.text(0, -2.15, '1', fontsize=12, ha='center', va='top')
    ax2.text(0, -1.15, '-4', fontsize=12, ha='center', va='top')

    ############################################################################

    gradient, (ax3, ax4) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(4, 2))
    gradient.suptitle('Fixed-Gradient Boundaries')
    gradient_x = [-1, 0, 1, 0]
    gradient_y = [-1, -1, -1, -2]

    ax3.axes.get_xaxis().set_ticks([])
    ax3.axes.get_yaxis().set_ticks([])
    ax3.plot(gradient_x, gradient_y, 'ro', color='blue')
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.axhline(0, color='black', linestyle='dashed')

    ax3.text(-0.85, -1, '1', fontsize=12, ha='left', va='top')
    ax3.text(0.85, -1, '1', fontsize=12, ha='right', va='top')
    ax3.text(0, -2.15, '1', fontsize=12, ha='center', va='top')
    ax3.text(0, -1.15, '-3', fontsize=12, ha='center', va='top')

    gradientcorner_x = [0, 1, 0]
    gradientcorner_y = [-1, -1, -2]

    ax4.axes.get_xaxis().set_ticks([])
    ax4.axes.get_yaxis().set_ticks([])
    ax4.plot(gradientcorner_x, gradientcorner_y, 'ro', color='blue')
    ax4.spines['left'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.axhline(0, color='black', linestyle='dashed')
    ax4.axvline(-1, color='black', linestyle='dashed')

    ax4.text(0.85, -1, '1', fontsize=12, ha='right', va='top')
    ax4.text(0, -2.15, '1', fontsize=12, ha='center', va='top')
    ax4.text(0, -1.15, '-2', fontsize=12, ha='center', va='top')

    ###########################################################

    slide_2_title = mo.md(
        r"""
    ## Diagrammatic Representation

    Grid edges and corners for <span style="color:red">fixed-value of 0</span> and <span style="color:blue">fixed-gradient of 0</span> on boundaries.

        """
    )

    mo.vstack([slide_2_title, mo.hstack([fixed, gradient])])
    return fixed, gradient


@app.cell
def _(fixed, mo):
    slide_3_title = mo.md(
        r"""
    ## Fixed-Value Boundary
        """
    ) 

    slide_3 = mo.md(
        r"""
    $$u_c = \frac{u_l + u_r + u_t + u_b}{4}$$
        """
    )
    slide_3_1 = mo.md(
        r"""
    $$u_c = \frac{1 + 1 + 0 + 1}{4} = \frac{3}{4}$$
        """
    )
    slide_3_2 = mo.md(
        r"""
    $$\nabla^2 u = 0$$
        """
    )
    slide_3_3 = mo.md(
        r"""
    $$ -4u_c + u_l + u_r + u_t + u_b = 0$$
        """
    )

    mo.vstack([slide_3_title, mo.hstack([fixed, mo.vstack ([slide_3, slide_3_1, slide_3_2, slide_3_3])], align='center')])
    return


@app.cell
def _(gradient, mo):
    slide_4_title = mo.md(
        r"""
    ## Fixed-Gradient Boundary
        """
    )

    slide_4 = mo.md(
        r"""
    $$\frac{\partial u}{\partial n} = 0$$
        """
    )
    slide_4_1 = mo.md(
        r"""
    $$\frac{\partial u}{\partial y} = \frac{u_{boundary} - u_{inside}}{\Delta y} = 0$$
        """
    )
    slide_4_2 = mo.md(
        r"""
    $$u_c = u_t$$
        """
    )
    slide_4_3 = mo.md(
        r"""
    $$-4u_c + u_l + u_r + u_c + u_b = 0$$
        """
    )
    slide_4_4 = mo.md(
        r"""
    $$-3u_c + u_l + u_r + u_b = 0$$
        """
    )

    mo.vstack([slide_4_title, mo.hstack([gradient, mo.vstack ([slide_4, slide_4_1, slide_4_2, slide_4_3, slide_4_4])], align='center')])
    return


@app.cell
def _(mo):
    slide_5_title = mo.md(
        r"""
    ### Ex 9.3: **Steady State of Heat Equation with Neumann Boundaries**

    * Solve for a temperature profile in a 2D system (geometry of fig 9.4)
    * Where the two squares plates have fixed temperatures $T_1$ and $T_2$ 
    * <span style="color:blue">Perfectly insulating walls</span> 

    **Neumann Boundary Conditions** - perpendicular derivative of T must be 0 everywhere on the walls
        """
    )

    mo.vstack([slide_5_title])
    return


@app.cell
def _(mo):
    slide_6_title = mo.md(
        r"""
    * We look at the temperature at a location ($r$) at time ($t$), so that **$T(r,t)$**
    * Thermal conduction is governed by **diffusion equation**:
    $$D\nabla^2T = \frac{\partial T}{\partial t} \ \text{, where $D$ is a constant thermal diffusity}$$ 
    * When the system is at **steady state** we have the **laplacian equation**
    $$\frac{\partial T}{\partial t} = 0$$
    $$\nabla^2 T = 0$$
        """
    )

    mo.vstack([slide_6_title])
    return


@app.cell
def _(empty, max, plt, zeros):
    # Constants
    L = 1.0         # Length of box in meters
    M = 100         # Grid squares on a side
    D = 1e-6        # Thermal diffusivity
    T1,T2 = 0.0,100.0
    target = 1e-2   # Target accuracy

    # Create arrays
    T = zeros([M+1,M+1],float)
    for i in range(20,41):
        for j in range(60,81):
            T[i,j] = T1
            T[j,i] = T2
    Tprime = empty([M+1,M+1],float)

    # Main loop
    delta = 1.0
    while delta>target:

        # Calculate new values of the potential at the corners
        Tprime[0,0] = (T[0,1]+T[1,0])/2
        Tprime[0,M] = (T[0,M-1]+T[1,M])/2
        Tprime[M,0] = (T[M,1]+T[M-1,0])/2
        Tprime[M,M] = (T[M,M-1]+T[M-1,M])/2

        # And the edges
        for i in range(1,M):
            Tprime[0,i] = (T[0,i-1]+T[0,i+1]+T[1,i])/3
            Tprime[M,i] = (T[M,i-1]+T[M,i+1]+T[M-1,i])/3
            Tprime[i,0] = (T[i-1,0]+T[i+1,0]+T[i,1])/3
            Tprime[i,M] = (T[i-1,M]+T[i+1,M]+T[i,M-1])/3

        # And the interior
        for i in range(1,M):
            for j in range(1,M):
                Tprime[i,j] = (T[i-1,j]+T[i+1,j]+T[i,j-1]+T[i,j+1])/4

        # Fix the temperature on the two plates
        for i in range(20,41):
            for j in range(60,81):
                Tprime[i,j] = T1
                Tprime[j,i] = T2

        # Calculate maximum change on any grid point
        delta = max(abs(T-Tprime))
        print(delta)

        # Swap the two arrays around
        T,Tprime = Tprime,T

    # Make a plot of the results
    exercise_9_3 = plt.imshow(T)
    # plt.gray()
    plt.show()
    return (exercise_9_3,)


@app.cell
def _(exercise_9_3, mo):

    slide_7_title = mo.md(
        r"""
    ### Ex 9.3: **Steady State of Heat Equation with Neumann Boundaries**
        """
    )
    slide_7 = mo.md(
        r"""
    $$T_1 = 0 \ ^oC$$
        """
    )
    slide_7_1 = mo.md(
        r"""
    $$T_2 = 100 \ ^oC$$
        """
    )
    slide_7_2 = mo.md(
        r"""
    $$D = 10^{-6} \ m^2 s^{-1}$$
        """
    )
    mo.vstack([slide_7_title, mo.hstack([exercise_9_3, mo.vstack([slide_7, slide_7_1, slide_7_2]) ], align='center')])
    return


@app.cell
def _(exercise_9_3, mo):
    slide_8_title = mo.md(
        r"""
    ## What the programme does
    * ignores any point outside of the grid 
    * no grid points on the boundary 

    ## The result 
    * temperature gradient from the hot plate to the cold 
    * perpendicular gradient at the walls are 0 
        """
    )

    mo.vstack([slide_8_title, exercise_9_3])
    return


@app.cell
def _(mo):
    slide_9_title = mo.md(
        r"""
    ## Dirichlet & Neumann Differences
        """
    )
    slide_9 = mo.md(
        r"""
    **<span style="color:red"> Dirichlet </span>**
    * specifies u
    * boundary point has defined value 
    * substitute boundary terms
        """
    )

    slide_9_1 = mo.md(
        r"""
    **<span style="color:blue"> Neumann </span>**
    * specifies $\frac{\partial u}{\partial n}$
    * boundary point equals its adjacent interior point for zero gradient 
    * boundary terms can be combined with the central-point coefficient
        """
    )

    mo.vstack([slide_9_title, mo.hstack([slide_9, slide_9_1]) ])
    return


@app.cell
def _(mo):
    slide_10 = mo.md(
        r"""
    ## Chapter 9.3.2 - **Numerical Stability**

    What we have seen previously is the **Forward-Time Centered-Space Method**

    $$\phi(x,t+h) = \phi(x,t)+h\frac{D}{a^2} [\phi(x+a,t)+\phi(x-a, t) - 2\phi(x,t)]$$

    * know $\phi$ at grid point $x$ at time $t$
    * find $\phi$ at time $t+h$

    <span style="color:mediumorchid"> Von Neumann Analysis of the Diffusion Equation </span> 
       """
    )

    mo.vstack([slide_10])
    return


@app.cell
def _(mo):
    slide_11 = mo.md(
        r"""
    ## Von Neumann Stability Analysis 

    **Idea:** small errors in the numerical solution grow or decrease as time progresses

    $$\phi (x,t)=\sum_k{c_k(t) e^{ikx}}$$

    Where we have: 
    * $k$ as wave vector 
    * $c_k(t)$ as coefficients (an amplitude at time $t$)
    * $e^{ikx}$ as a spatial osciallation
       """
    )

    mo.vstack([slide_11])
    return


@app.cell
def _(mo):
    slide_12 = mo.md(
        r"""
    Substituting <span style="color:mediumorchid"> von Neumann stability analysis </span> 

    $$\phi(x,t+h) = \phi(x,t)+h\frac{D}{a^2} \left[\phi(x+a,t)+\phi(x-a, t) - 2\phi(x,t) \right]$$

    $$\phi(x,t+h) = \color{mediumorchid}{c_k(t) e^{ikx}} \color{black}{+ h\frac{D}{a^2}} \color{mediumorchid} {c_k(t) \left[e^{ik(x+a)} +e^{ik(x-a)} - 2e^{ikx} \right]}$$

    Solve and simplify with Euler's and trigonometric identity:
    $$c_k (t+h) = \left[1-h \frac{4D}{a^2} \sin^2 {\frac{1}{2}ka} \right] c_k(t)$$

       """ 
    )

    mo.vstack([slide_12])
    return


@app.cell
def _(mo):
    slide_13 = mo.md(
        r"""
    ## How does it become unstable?

    $$c_k (t+h) = \color{mediumorchid} {\left[1-h \frac{4D}{a^2} \sin^2 {\frac{1}{2}ka} \right]} \color{black} {c_k(t)}$$

    At each step of $h$ the $c_k$ is multiplied by a **<span style="color:mediumorchid"> factor </span>**

    * grows **exponentially**
    * depends on $k$ but not on $x$ or $t$

       """ 
    )

    mo.vstack([slide_13])
    return


@app.cell
def _(mo):
    slide_14 = mo.md(
        r"""
    ## Amplification Factor

    $$ -1 ≤ \color{mediumorchid} {\left[1-h \frac{4D}{a^2} \sin^2 {\frac{1}{2}ka} \right]} \color{black} {≤ 1 }$$

    $$ \frac{4hD}{a^2} \sin^2{\frac{ka}{2}} ≤ 2$$

    $$h ≤ \frac{a^2} {2D}$$

       """ 
    )

    mo.vstack([slide_14])
    return


@app.cell
def _(mo):
    slide_15 = mo.md(
        r"""
    ## Diffusion Equation

    If $h$ **>** $\frac{a^2} {2D}$

    * diverges
    * cool object does not become infintely hot

    If $h$ **<** $\frac{a^2} {2D}$

    * exponential decay
    * except for k=0 &rarr; uniform state

       """ 
    )

    mo.vstack([slide_15])
    return


@app.cell
def _(mo):
    slide_a = mo.md(
        r"""
    ## "The FTCS method fails badly when applied to the wave equation, and it does so for interesting reasons."

       """
    )

    mo.vstack([slide_a])
    return


@app.cell
def _(mo):
    slide_aaa = mo.md(
        r"""
    ## "This is not mere rounding error"

    ##"This is something more serious. The calculation has become <span style="color:limegreen">numerically unstable</span>. "

       """
    )

    mo.vstack([slide_aaa])
    return


@app.cell
def _(mo):
    slide_16 = mo.md(
        r"""
    ## Wave Equation

    $$ \frac{\partial^2 \phi}{\partial x^2} - \frac{1}{v^2} \frac{\partial^2 \phi}{\partial t^2} = 0$$

    for each grid point:

    $$ \frac{d\phi}{dt} = \psi (x,t) $$

    $$\frac{d\psi}{dt} = h\frac{v^2}{a^2} \left[ \phi(x+a, t) +\phi(x-a, t) - 2\phi(x,t) \right]$$

       """ 
    )

    mo.vstack([slide_16])
    return


@app.cell
def _(mo):
    slide_17 = mo.md(
        r"""
    ## Wave Equation FTC

    $$ \phi (x, t+h) = \phi(x,t) + h\psi(x,t)$$

    $$ \psi (x, t+h) = \psi(x,t) + h\frac{v^2}{a^2} \left[ \phi(x+a, t) +\phi(x-a, t) - 2\phi(x,t) \right]$$

    <span style="color:limegreen">applying von Neumann analysis:</span>

    \[
    \begin{pmatrix}
    \phi(x,t) \\
    \psi(x,t)
    \end{pmatrix}
    =
    \begin{pmatrix}
    c_\phi(t) \\
    c_\psi(t)
    \end{pmatrix}
    e^{ikx}.
    \]
       """ 
    )

    mo.vstack([slide_17])
    return


@app.cell
def _(mo):
    slide_18 = mo.md(
        r"""
    ## Next time step

    $$ c_\phi(t+h) = c_\phi(t) + h c_\psi(t) $$

    $$ c_\psi(t+h) = c_\psi(t) - h c_\phi(t)\frac{4v^2}{a^2} \sin^2\left(\frac{1}{2}ka\right) $$

    ## in vector form:

    \[
    \mathbf{c}(t+h) = A\mathbf{c}(t).
    \]
    \[
    A =
    \begin{pmatrix}
    1 & h \\
    -hr^2 & 1
    \end{pmatrix},
    \qquad
    r = \frac{2v}{a}\sin\left(\frac{1}{2}ka\right).
    \]

       """ 
    )

    mo.vstack([slide_18])
    return


@app.cell
def _(mo):
    slide_19 = mo.md(
        r"""
    ## Substituting together

    $$ \mathbf{c}(t+h) = A(\alpha_1\mathbf{v}_1+\alpha_2\mathbf{v}_2) $$

    $$ = \alpha_1\lambda_1\mathbf{v}_1 + \alpha_2\lambda_2\mathbf{v}_2 $$

    where 
    * $\lambda$ are eigenvalues

       """ 
    )

    mo.vstack([slide_19])
    return


@app.cell
def _(mo):
    slide_20 = mo.md(
        r"""
    ## The next time steps

    $$ \mathbf{c}(t+2h) = \alpha_1\lambda_1^2\mathbf{v}_1 + \alpha_2\lambda_2^2\mathbf{v}_2 $$

    ## m steps
    $$
    \mathbf{c}(t+mh) = \alpha_1\lambda_1^m\mathbf{v}_1 + \alpha_2\lambda_2^m\mathbf{v}_2 $$

       """ 
    )

    mo.vstack([slide_20])
    return


@app.cell
def _(mo):
    slide_21 = mo.md(
        r"""
    ## When is the system stable?

    magntiude of 
    $$\lambda ≤ 1$$

    Otherwise 
    * diverges to $\infty$
       """ 
    )

    mo.vstack([slide_21])
    return


@app.cell
def _(mo):
    slide_22 = mo.md(
        r"""
    ## The eigenvalues 

    $$ \det(A-\lambda I)=0 $$


    $$(1-\lambda)^2+h^2r^2=0$$

    The magnitudes:

    $$ |\lambda| = \sqrt{1+h^2r^2} $$

    $$ = \sqrt{1+\frac{4h^2v^2}{a^2} \sin^2\left(\frac{1}{2}ka\right)} $$
       """ 
    )

    mo.vstack([slide_22])
    return


if __name__ == "__main__":
    app.run()
