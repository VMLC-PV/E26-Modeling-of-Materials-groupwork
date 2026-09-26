import marimo

__generated_with = "0.24.2"
app = marimo.App(layout_file="layouts/testing.slides.json")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from numpy.linalg import solve
    import cmath
    import matplotlib.pyplot as plt

    return cmath, mo, np


@app.cell
def _(mo):
    node = mo.ui.dropdown(
        options={
            "Node 1": "1",
            "Node 2": "2",
            "Node 3": "3",
        },
        value="Node 1",
        label="Choose the node:"
    )
    return (node,)


@app.cell
def _(mo, node):

    selected = node.value

    # --------------------------------------------------------
    # Colors for highlighting
    # --------------------------------------------------------

    normal = "black"
    highlight = "#e63946"

    # Node 1 components
    r1_color = highlight if selected == "1" else normal
    r4_color = highlight if selected == "1" else normal
    c1_color = highlight if selected in ["1", "2"] else normal

    # Node 2 components
    r2_color = highlight if selected == "2" else normal
    r5_color = highlight if selected == "2" else normal
    c2_color = highlight if selected in ["2", "3"] else normal

    # Node 3 components
    r3_color = highlight if selected == "3" else normal
    r6_color = highlight if selected == "3" else normal


    circuit = f"""
    <svg
        width="100%"
        height="auto"
        viewBox="0 0 650 500"
        xmlns="http://www.w3.org/2000/svg"
    >


        <line x1="80" y1="60" x2="570" y2="60"
              stroke="black" stroke-width="3"/>

        <line x1="80" y1="440" x2="570" y2="440"
              stroke="black" stroke-width="3"/>

        <line x1="170" y1="60" x2="170" y2="100"
              stroke="{r1_color}" stroke-width="3"/>

        <polyline
            points="170,100 155,120 185,140 155,160
                    185,180 155,200 185,220 170,240"
            fill="none"
            stroke="{r1_color}"
            stroke-width="3"
        />

        <line x1="170" y1="240" x2="170" y2="270"
              stroke="{r1_color}" stroke-width="3"/>

        <text x="125" y="175"
              font-size="21"
              font-family="serif"
              fill="{r1_color}">
            R₁
        </text>

        <line x1="170" y1="270" x2="170" y2="300"
              stroke="{r4_color}" stroke-width="3"/>

        <polyline
            points="170,300 155,320 185,340 155,360
                    185,380 155,400 170,440"
            fill="none"
            stroke="{r4_color}"
            stroke-width="3"
        />

        <text x="125" y="370"
              font-size="21"
              font-family="serif"
              fill="{r4_color}">
            R₄
        </text>


        <line x1="325" y1="60" x2="325" y2="100"
              stroke="{r2_color}" stroke-width="3"/>

        <polyline
            points="325,100 310,120 340,140 310,160
                    340,180 310,200 340,220 325,240"
            fill="none"
            stroke="{r2_color}"
            stroke-width="3"
        />

        <line x1="325" y1="240" x2="325" y2="270"
              stroke="{r2_color}" stroke-width="3"/>

        <text x="280" y="175"
              font-size="21"
              font-family="serif"
              fill="{r2_color}">
            R₂
        </text>


        <!-- ============================================== -->
        <!-- R5 -->
        <!-- ============================================== -->

        <line x1="325" y1="270" x2="325" y2="300"
              stroke="{r5_color}" stroke-width="3"/>

        <polyline
            points="325,300 310,320 340,340 310,360
                    340,380 310,400 325,440"
            fill="none"
            stroke="{r5_color}"
            stroke-width="3"
        />

        <text x="280" y="370"
              font-size="21"
              font-family="serif"
              fill="{r5_color}">
            R₅
        </text>


        <!-- ============================================== -->
        <!-- R3 -->
        <!-- ============================================== -->

        <line x1="480" y1="60" x2="480" y2="100"
              stroke="{r3_color}" stroke-width="3"/>

        <polyline
            points="480,100 465,120 495,140 465,160
                    495,180 465,200 495,220 480,240"
            fill="none"
            stroke="{r3_color}"
            stroke-width="3"
        />

        <line x1="480" y1="240" x2="480" y2="270"
              stroke="{r3_color}" stroke-width="3"/>

        <text x="435" y="175"
              font-size="21"
              font-family="serif"
              fill="{r3_color}">
            R₃
        </text>


        <!-- ============================================== -->
        <!-- R6 -->
        <!-- ============================================== -->

        <line x1="480" y1="270" x2="480" y2="300"
              stroke="{r6_color}" stroke-width="3"/>

        <polyline
            points="480,300 465,320 495,340 465,360
                    495,380 465,400 480,440"
            fill="none"
            stroke="{r6_color}"
            stroke-width="3"
        />

        <text x="435" y="370"
              font-size="21"
              font-family="serif"
              fill="{r6_color}">
            R₆
        </text>


        <!-- ============================================== -->
        <!-- C1 -->
        <!-- ============================================== -->

        <line x1="170" y1="270" x2="270" y2="270"
              stroke="{c1_color}" stroke-width="3"/>

        <line x1="270" y1="235" x2="270" y2="305"
              stroke="{c1_color}" stroke-width="4"/>

        <line x1="290" y1="235" x2="290" y2="305"
              stroke="{c1_color}" stroke-width="4"/>

        <line x1="290" y1="270" x2="325" y2="270"
              stroke="{c1_color}" stroke-width="3"/>

        <text x="263" y="220"
              font-size="21"
              font-family="serif"
              fill="{c1_color}">
            C₁
        </text>


        <!-- ============================================== -->
        <!-- C2 -->
        <!-- ============================================== -->

        <line x1="325" y1="270" x2="425" y2="270"
              stroke="{c2_color}" stroke-width="3"/>

        <line x1="425" y1="235" x2="425" y2="305"
              stroke="{c2_color}" stroke-width="4"/>

        <line x1="445" y1="235" x2="445" y2="305"
              stroke="{c2_color}" stroke-width="4"/>

        <line x1="445" y1="270" x2="480" y2="270"
              stroke="{c2_color}" stroke-width="3"/>

        <text x="418" y="220"
              font-size="21"
              font-family="serif"
              fill="{c2_color}">
            C₂
        </text>


        <!-- ============================================== -->
        <!-- NODES -->
        <!-- ============================================== -->

        <circle cx="170" cy="60" r="6" fill="black"/>
        <circle cx="325" cy="60" r="6" fill="black"/>
        <circle cx="480" cy="60" r="6" fill="black"/>

        <circle cx="170" cy="270" r="9"
                fill="{highlight if selected == '1' else 'black'}"/>

        <circle cx="325" cy="270" r="9"
                fill="{highlight if selected == '2' else 'black'}"/>

        <circle cx="480" cy="270" r="9"
                fill="{highlight if selected == '3' else 'black'}"/>

        <circle cx="170" cy="440" r="6" fill="black"/>
        <circle cx="325" cy="440" r="6" fill="black"/>
        <circle cx="480" cy="440" r="6" fill="black"/>


        <!-- ============================================== -->
        <!-- NODE NUMBERS -->
        <!-- ============================================== -->

        <text x="155" y="260"
              font-size="20"
              font-family="serif">
            1
        </text>

        <text x="310" y="260"
              font-size="20"
              font-family="serif">
            2
        </text>

        <text x="465" y="260"
              font-size="20"
              font-family="serif">
            3
        </text>


        <!-- ============================================== -->
        <!-- VOLTAGE LABELS -->
        <!-- ============================================== -->

        <text x="585" y="68"
              font-size="22"
              font-family="serif">
            V₊
        </text>

        <text x="585" y="448"
              font-size="22"
              font-family="serif">
            0 V
        </text>

    </svg>
    """

    circuit_visual = mo.Html(circuit)
    return (circuit_visual,)


@app.cell
def _(mo, node):

    if node.value == "1":

        explanation = mo.md(
            r"""
            ## Node 1
            ### Currents
            Through \(R_1\):
            \[I_{R_1}=\frac{V_1-V_+}{R_1}\]

            Through \(R_4\):
            \[I_{R_4}=\frac{V_1}{R_4}\]

            Through \(C_1\):
            \[I_{C_1} = i\omega C_1(V_1-V_2)\]

            ### Kirchhoff's current law
            Sum of currents leaving node 1:
            \[\frac{V_1-V_+}{R_1} + \frac{V_1}{R_4} + i\omega C_1(V_1-V_2) =0\]

            Collecting terms:
            \[\left(\frac1{R_1} +\frac1{R_4} + i\omega C_1 \right)V_1 - i\omega C_1V_2 = \frac{V_+}{R_1}\]

            Finally, substituting
            \[V_j=x_je^{i\omega t}\]

            gives:
            \[\boxed{ \left( \frac1{R_1} + \frac1{R_4} + i\omega C_1 \right)x_1 - i\omega C_1x_2 = \frac{x_+}{R_1} }\]
            """
        )

    elif node.value == "2":

        explanation = mo.md(
            r"""
            ## Node 2
            ### Currents
            Through \(R_2\):
            \[I_{R_2}=\frac{V_2-V_+}{R_2}\]

            Through \(R_5\):
            \[I_{R_5}=\frac{V_2}{R_5}\]

            Through \(C_1\):
            \[I_{C_1}=i\omega C_1(V_2-V_1)\]

            Through \(C_2\):
            \[I_{C_2}=i\omega C_2(V_2-V_3)\]

            ### Kirchhoff's current law
            \[\frac{V_2-V_+}{R_2} + \frac{V_2}{R_5} + i\omega C_1(V_2-V_1) + i\omega C_2(V_2-V_3) =0\]

            Collecting terms:
            \[-i\omega C_1V_1 + \left( \frac1{R_2} + \frac1{R_5} + i\omega C_1 + i\omega C_2 \right)V_2 - i\omega C_2V_3 = \frac{V_+}{R_2}\]

            Therefore,
            \[\boxed{-i\omega C_1x_1  +  \left( \frac1{R_2} + \frac1{R_5} + i\omega C_1 + i\omega C_2 \right)x_2 - i\omega C_2x_3 = \frac{x_+}{R_2} }\]
            """
        )

    else:

        explanation = mo.md(
            r"""
            ## Node 3
            ### Currents
            Through \(R_3\):
            \[I_{R_3} = \frac{V_3-V_+}{R_3}\]

            Through \(R_6\):
            \[I_{R_6} = \frac{V_3}{R_6}\]

            Through \(C_2\):
            \[I_{C_2} = i\omega C_2(V_3-V_2)\]

            ### Kirchhoff's current law
            \[\frac{V_3-V_+}{R_3} + \frac{V_3}{R_6} + i\omega C_2(V_3-V_2) =0\]

            Collecting terms:
            \[-i\omega C_2V_2 + \left( \frac1{R_3} + \frac1{R_6} + i\omega C_2 \right)V_3 = \frac{V_+}{R_3}\]

            Therefore,
            \[\boxed{ -i\omega C_2x_2 + \left( \frac1{R_3} + \frac1{R_6} + i\omega C_2 \right)x_3 = \frac{x_+}{R_3}}\]
            """ 
        )
    return (explanation,)


@app.cell
def _(circuit_visual, explanation, mo, node):
    mo.vstack([
        mo.md("# Exercise 6.5"),
        node,
        mo.hstack(
            [
            mo.vstack([mo.md("### Circuit"), circuit_visual]),
                explanation,
            ],
            widths=[1, 1],
            gap="2rem",
        ),
    ])
    return


@app.cell
def _(mo):
    equations = mo.md(
            r"""
            ### Equations

            \[
            \left(\frac1{R_1} + \frac1{R_4} + i\omega C_1 \right)x_1 - i\omega C_1x_2 = \frac{x_+}{R_1}
            \]

            \[
            -i\omega C_1x_1 + \left(\frac1{R_2} + \frac1{R_5} + i\omega C_1 + i\omega C_2 \right)x_2 - i\omega C_2x_3 = \frac{x_+}{R_2}
            \]

            \[ 
            -i\omega C_2x_2 + \left(\frac1{R_3} + \frac1{R_6} + i\omega C_2 \right)x_3 = \frac{x_+}{R_3}
            \]
            """
        )
    return (equations,)


@app.cell
def _(mo):
    values = mo.md(
            r"""
            #### with:

            \[R_1=R_3=R_5=1k\Omega\]
            \[R_2=R_4=R_6=2k\Omega\]
            \[C_1=1\mu F, C_2=0.5\mu F\]
            \[x_+=3V, \omega=1000s^{-1}\]
            """
        )
    return (values,)


@app.cell
def _(mo):
    circuit_values = """
    <svg
        width="100%"
        height="auto"
        viewBox="0 0 760 520"
        xmlns="http://www.w3.org/2000/svg"
    >

        <!-- TOP AND BOTTOM RAILS -->

        <line x1="80" y1="60" x2="600" y2="60"
              stroke="black" stroke-width="3"/>

        <line x1="80" y1="440" x2="600" y2="440"
              stroke="black" stroke-width="3"/>


        <!-- R1 = 1 kΩ -->

        <line x1="170" y1="60" x2="170" y2="100"
              stroke="black" stroke-width="3"/>

        <polyline
            points="170,100 155,120 185,140 155,160
                    185,180 155,200 185,220 170,240"
            fill="none"
            stroke="black"
            stroke-width="3"
        />

        <line x1="170" y1="240" x2="170" y2="270"
              stroke="black" stroke-width="3"/>

        <text x="195" y="175"
              font-size="18"
              font-family="serif">
            R₁ = 1 kΩ
        </text>


        <!-- R4 = 2 kΩ -->

        <line x1="170" y1="270" x2="170" y2="300"
              stroke="black" stroke-width="3"/>

        <polyline
            points="170,300 155,320 185,340 155,360
                    185,380 155,400 170,440"
            fill="none"
            stroke="black"
            stroke-width="3"
        />

        <text x="195" y="370"
              font-size="18"
              font-family="serif">
            R₄ = 2 kΩ
        </text>


        <!-- R2 = 2 kΩ -->

        <line x1="325" y1="60" x2="325" y2="100"
              stroke="black" stroke-width="3"/>

        <polyline
            points="325,100 310,120 340,140 310,160
                    340,180 310,200 340,220 325,240"
            fill="none"
            stroke="black"
            stroke-width="3"
        />

        <line x1="325" y1="240" x2="325" y2="270"
              stroke="black" stroke-width="3"/>

        <text x="350" y="175"
              font-size="18"
              font-family="serif">
            R₂ = 2 kΩ
        </text>


        <!-- R5 = 1 kΩ -->

        <line x1="325" y1="270" x2="325" y2="300"
              stroke="black" stroke-width="3"/>

        <polyline
            points="325,300 310,320 340,340 310,360
                    340,380 310,400 325,440"
            fill="none"
            stroke="black"
            stroke-width="3"
        />

        <text x="350" y="370"
              font-size="18"
              font-family="serif">
            R₅ = 1 kΩ
        </text>


        <!-- R3 = 1 kΩ -->

        <line x1="480" y1="60" x2="480" y2="100"
              stroke="black" stroke-width="3"/>

        <polyline
            points="480,100 465,120 495,140 465,160
                    495,180 465,200 495,220 480,240"
            fill="none"
            stroke="black"
            stroke-width="3"
        />

        <line x1="480" y1="240" x2="480" y2="270"
              stroke="black" stroke-width="3"/>

        <text x="505" y="175"
              font-size="18"
              font-family="serif">
            R₃ = 1 kΩ
        </text>


        <!-- R6 = 2 kΩ -->

        <line x1="480" y1="270" x2="480" y2="300"
              stroke="black" stroke-width="3"/>

        <polyline
            points="480,300 465,320 495,340 465,360
                    495,380 465,400 480,440"
            fill="none"
            stroke="black"
            stroke-width="3"
        />

        <text x="505" y="370"
              font-size="18"
              font-family="serif">
            R₆ = 2 kΩ
        </text>


        <!-- C1 = 1 μF -->

        <line x1="170" y1="270" x2="270" y2="270"
              stroke="black" stroke-width="3"/>

        <line x1="270" y1="235" x2="270" y2="305"
              stroke="black" stroke-width="4"/>

        <line x1="290" y1="235" x2="290" y2="305"
              stroke="black" stroke-width="4"/>

        <line x1="290" y1="270" x2="325" y2="270"
              stroke="black" stroke-width="3"/>

        <text x="245" y="220"
              font-size="17"
              font-family="serif">
            C₁ = 1 μF
        </text>


        <!-- C2 = 0.5 μF -->

        <line x1="325" y1="270" x2="425" y2="270"
              stroke="black" stroke-width="3"/>

        <line x1="425" y1="235" x2="425" y2="305"
              stroke="black" stroke-width="4"/>

        <line x1="445" y1="235" x2="445" y2="305"
              stroke="black" stroke-width="4"/>

        <line x1="445" y1="270" x2="480" y2="270"
              stroke="black" stroke-width="3"/>

        <text x="395" y="220"
              font-size="17"
              font-family="serif">
            C₂ = 0.5 μF
        </text>


        <!-- GROUND -->

        <circle cx="480" cy="440" r="6" fill="black"/>


        <!-- SOURCE -->

        <text x="615" y="68"
              font-size="20"
              font-family="serif">
            x₊ = 3 V
        </text>

        <text x="615" y="448"
              font-size="20"
              font-family="serif">
            0 V
        </text>


        <!-- ANGULAR FREQUENCY -->

        <text x="280" y="495"
              font-size="20"
              font-family="serif">
            ω = 1000 s⁻¹
        </text>

    </svg>
    """

    circuit_values = mo.Html(circuit_values)
    return (circuit_values,)


@app.cell
def _(equations, mo):
    mo.vstack([equations])
    return


@app.cell
def _(circuit_values, mo, values):
    mo.vstack([
        mo.hstack(
            [
            circuit_values,
            mo.vstack([values]),
            ],
            widths=[1, 1],
            gap="2rem",
        ),
    ])
    return


@app.cell
def _(mo):
    equations_matrix = mo.md(
        r"""
        ### Equations in Matrix Form

        $$
        \begin{bmatrix}
        \frac{1}{R_1}+\frac{1}{R_4}+i\omega C_1 & -i\omega C_1 & 0 \\[6pt]
        -i\omega C_1 & \frac{1}{R_2}+\frac{1}{R_5}+i\omega C_1+i\omega C_2 & -i\omega C_2 \\[6pt]
        0 & -i\omega C_2 & \frac{1}{R_3}+\frac{1}{R_6}+i\omega C_2
        \end{bmatrix}
        \begin{bmatrix}
        x_1\\
        x_2\\
        x_3
        \end{bmatrix}
        =
        \begin{bmatrix}
        \frac{x_+}{R_1}\\
        \frac{x_+}{R_2}\\
        \frac{x_+}{R_3}
        \end{bmatrix}
        $$
        """
    )
    return (equations_matrix,)


@app.cell
def _(mo):
    values_matrix = mo.md(
        r"""
        ### Using given values:

        $$ 
        \begin{bmatrix}
        0.0015 + 0.001i & -0.001i & 0 \\[6pt]
        -0.001i & 0.0015 + 0.0015i & -0.0005i \\[6pt]
        0 & -0.0005i & 0.0015 + 0.0005i
        \end{bmatrix}
        \begin{bmatrix}
        x_1\\
        x_2\\
        x_3
        \end{bmatrix}
        =
        \begin{bmatrix}
        0.003\\
        0.0015\\
        0.003
        \end{bmatrix}
        $$
        """
    )
    return (values_matrix,)


@app.cell
def _(equations_matrix, mo, values_matrix):
    mo.vstack([equations_matrix, values_matrix])
    return


@app.cell
def _(np):
    A_circuit = np.array(
        [
            [0.0015 + 0.001j, -0.001j, 0.0],
            [-0.001j,0.0015 + 0.0015j, -0.0005j],
            [0.0,-0.0005j, 0.0015 + 0.0005j],
        ]
    )

    v_circuit = np.array(
        [0.003, 0.0015, 0.003]
    )
    return A_circuit, v_circuit


@app.cell
def _(A_circuit, cmath, mo, np, v_circuit):
    x_circuit = np.linalg.solve(A_circuit, v_circuit)

    results = []

    for _i, _voltage in enumerate(x_circuit, start=1):
        amplitude, phase = cmath.polar(_voltage)
        phase_degrees = np.degrees(phase)

        results.append(
            f"""
    **V{_i}**

    - Complex voltage: `[{_voltage.real:.4f} {_voltage.imag:+.4f}j] V`
    - Amplitude: **{amplitude:.4f} V**
    - Phase: **{phase_degrees:.2f}°**
    """
        )

    solution = mo.md(
        r"""
    ### Numerical Solution

    The numerical matrix has the form

    $$ A\mathbf{x}=\mathbf{b} $$

    where $A$ is the circuit matrix, $\mathbf{x}$ contains the unknown node voltages, and $\mathbf{b}$ is the source vector.

    We use `np.linalg.solve(A, b)` to solve this system directly.


    Since the voltages are complex phasors, each result can be written as

    $$ V_k = |V_k|\angle\phi_k $$

    where $|V_k|$ is the amplitude and $\phi_k$ is the phase angle.
    """
        + "\n\n---\n\n"
        + "\n\n---\n\n".join(results)
    )

    solution
    return (x_circuit,)


@app.cell
def _(A_circuit, mo, np, v_circuit, x_circuit):
    v_check = A_circuit @ x_circuit

    # Calculate the maximum absolute error
    error = np.max(np.abs(v_check - v_circuit))

    # Numerical tolerance for floating-point roundoff
    tolerance = 1e-12

    verification = mo.md(
        f"""
    ### Verification

    Substituting the calculated node voltages back into the original
    nodal equations gives:

    $$
    \\left(
    0.0015 + i0.001
    \\right)
    \\left({x_circuit[0].real:.4f} {x_circuit[0].imag:+.4f}j\\right)
    -i0.001
    \\left({x_circuit[1].real:.4f} {x_circuit[1].imag:+.4f}j\\right)
    = {v_check[0].real:.4f}
    $$

    $$
    -i0.001
    \\left({x_circuit[0].real:.4f} {x_circuit[0].imag:+.4f}j\\right)
    +
    \\left(
    0.0015 + i0.0015
    \\right)
    \\left({x_circuit[1].real:.4f} {x_circuit[1].imag:+.4f}j\\right)
    -i0.0005
    \\left({x_circuit[2].real:.4f} {x_circuit[2].imag:+.4f}j\\right)
    = {v_check[1].real:.4f}
    $$

    $$
    -i0.0005
    \\left({x_circuit[1].real:.4f} {x_circuit[1].imag:+.4f}j\\right)
    +
    \\left(
    0.0015 + i0.0005
    \\right)
    \\left({x_circuit[2].real:.4f} {x_circuit[2].imag:+.4f}j\\right)
    = {v_check[2].real:.4f}
    $$

    The resulting values are:

    $$
    A\\mathbf{{x}} =
    \\begin{{bmatrix}}
    {v_check[0].real:.4f} \\\\
    {v_check[1].real:.4f} \\\\
    {v_check[2].real:.4f}
    \\end{{bmatrix}}
    \\approx
    \\begin{{bmatrix}}
    {v_circuit[0]:.4f} \\\\
    {v_circuit[1]:.4f} \\\\
    {v_circuit[2]:.4f}
    \\end{{bmatrix}}
    = \\mathbf{{b}}
    $$

    The maximum absolute error is:

    $$
    \\max |A\\mathbf{{x}}-\\mathbf{{b}}|
    = {error:.2e}
    $$

    the error is **within floating-point roundoff error**. The calculated node voltages satisfy the original nodal equations.
    """
    )

    verification
    return


if __name__ == "__main__":
    app.run()
