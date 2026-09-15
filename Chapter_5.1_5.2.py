# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.11.1",
#     "numpy>=2.5.3",
#     "scipy>=1.18.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    from scipy.interpolate import interp1d

    return interp1d, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.1 - Fundamental methods for evaluating integrals
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fundamental Methods for Evaluating Integrals

    In an integral the domain of a function is divided into small units. At a point, $x$, in each unit,
    the function, $f(x)$, is multiplied by the measurement of the unit. All the products are then summed.

    This chapter will cover the numerical evaluation of an integral of a function of a single variable
    over a finite range.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.1.1 - Trapezoidal Rule
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calculating the integral of a function is equivalent to calculating the area underneath
    the function's curve within a domain.

    For instance, function $f(x)$ with domain $x=a$ to $x=b$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Trapezodial Rule Formula:**

    $$\int_{a}^{b} f(x) \, dx $$

    $$\approx h \left[ \frac{1}{2} f(a) + \frac{1}{2} f(b) + \sum_{k=1}^{n-1} f(a+kh) \right]$$

    $ n $ is number of slices

    $ h $ is slice width
    $$ h = (b-a)/n$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code
    """)
    return


@app.cell
def _(mo):
    # User controls integer input
    number_input = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=1,
        # label="Select an interval $h$ to approximate the integral (1 - 100):",
    )
    num_md = mo.md(f"""### Select an interval $n$ to approximate the integral (1 - 100):""")
    num_in = mo.vstack([num_md, number_input], align="center")
    return num_in, number_input


@app.cell
def _(mo, np, number_input, plt):
    def function(x):
        return x**2 -2*x+1

    n = int(number_input.value)
    a = 2
    b = 10

    # width of interval
    h = (b-a)/n

    # Values for the approximation 
    x = np.linspace(a, b, n+1)
    y = function(x)

    # Values for the function 
    x_function = np.linspace(0, 15, 1000)
    y_function = function(x_function)

    plt.plot(x_function, y_function, color='red', label='f(x)=$x^2 -2x+1$')
    plt.vlines(x, ymin=0, ymax=y) # draws the verticle lines for the approx
    plt.axhline(y=0, xmin=0, xmax=1, color='black') # draws the horizontal lines for the approx
    plt.plot(x, y, marker='.', color='black', markersize=10, label='approximation')

    plt.title('Trapezoidal Rule')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    # plt.show()
    graph = mo.mpl.interactive(plt.gcf())
    return graph, h, n, y


@app.cell
def _(h, mo, n, y):
    area = (h/2) * (y[0] + (2*sum(y[1:n])) + y[n])
    trap_area = mo.md(f"""### Approximated area: {area}""")
    return area, trap_area


@app.cell
def _(mo):
    trap_title = mo.md(
        r"""
        ## Trapezodial Approximation of $I = \int_{2}^{10} \ (x^2-2x+1) \, dx$"""
    )
    return (trap_title,)


@app.cell
def _(area, mo, trap_area):
    # actual integral value from x=2 to x=10
    integral = 728/3

    actual_val = mo.md(f"""### Actual integral value: {integral}""")

    results = mo.hstack(
                [trap_area, actual_val],
                justify="center", gap=5
            )


    # absolute error
    abs_error = abs(integral - area)

    # relative error
    rel_error = abs_error/integral

    absolute_error = mo.md(f"""### Absolute error: {abs_error}""")

    relative_error = mo.md(f"""### Relative error: {rel_error}""")

    error = mo.hstack(
                [absolute_error, relative_error],
                justify="center", gap=5
            )
    return error, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visuals
    """)
    return


@app.cell
def _(error, graph, mo, num_in, results, trap_title):
    mo.vstack([trap_title, num_in, results, error, graph])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.1.2 - Simpson’s rule
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **Trapezoidal Rule** estimates the area under a curve by approximating the curve with linear segments.

    The **Simpson's Rule** approximates the function with quadratic curves.

    To do define a quadratic curve we need 3 points. These 3 points can then be used to fit a quadratic within the "slices" used to integrate.

    Such that a slice with width $h$ could have the points $x = +h$,  $0$,  $-h$.

    Then the Simpson's rule will give the area underneath two adjacent slices. To do so the number of slices must therefore be even.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Simpson's Rule Formula:**

    $$\int_{a}^{b} f(x) \, dx $$

    $$\approx \frac{h}{3} \left[f(a) + f(b) + 4 * \sum_{k odd} f(a+kh) + 2* \sum_{k even} f(a+kh)\right]$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code
    """)
    return


@app.cell
def _(mo):
    # get number of slices 
    number_slices = mo.ui.number(
        start=0,
        stop=100,
        step=2,
        value=2,
        # label="Select an interval $h$ to approximate the integral (1 - 100):",
    )
    num_slice_md = mo.md(f"""### Select a number of slices $n$ to approximate the integral (must be even integer):""")
    num_slice_in = mo.vstack([num_slice_md, number_slices], align="center")

    # title     x1 * np.exp(-0.5 * x1) 
    simpson_title = mo.md(
        r"""
        ## Simpson's Rule Approximation of $I = \int_{0}^{12} \ (xe^{-0.5x}) \, dx$"""
    )
    return num_slice_in, number_slices, simpson_title


@app.cell
def _(np):
    def g(x_x):
        return x_x * np.exp(-0.5 * x_x) 

    # data for function
    x1 = np.linspace(0, 15, 1000)
    y1 = g(x1)
    return g, x1, y1


@app.cell
def _(g, mo, np, number_slices):
    # Approximate the integral with Simpson's rule 
    def approx_integral(a1, b1, n1):
        h1 = (b1 - a1)/n1 

        g_sum = (g(a1) + g(b1))
        sum_odd = 0
        sum_even = 0

        # calculate for odd terms
        for k in range(1, n1, 2): 
            sum_odd += ( g(a1+k*h1)) 
        sum_odd *= 4

        # calculate for even terms 
        for k in range (2, n1, 2):
            sum_even += ( g(a1+k*h1))
        sum_even *= 2 

        total_sum = h1/3 * (g_sum + sum_odd + sum_even)

        return total_sum

    a1 = 0
    b1 = 12
    n1 = int(number_slices.value)

    # Integral Values 

    simpson = approx_integral(a1, b1, n1)
    simpson_value = mo.md(f"""### Simpson approximation: {simpson}""")

    actual_simpson = 4 - 28 * np.exp(-6)
    actual_simpson_val = mo.md(f"""### Actual integral value: {actual_simpson}""")

    simpson_results = mo.hstack(
                [simpson_value, actual_simpson_val],
                justify="center", gap=5
            )
    return a1, b1, n1, simpson_results


@app.cell
def _(a1, b1, n1):
    # Finding the x values : 

    list = []

    def calculate_x (a1, h1, n1):
        a_x = a1
        for i in range (0, n1):
            a_xx = a_x+i*h1
            list.append(a_xx)
        print (list)

        return list


    h1 = (b1 - a1)/n1 
    calculate_x (a1, h1, n1)
    return


@app.cell
def _(a1, b1, g, interp1d, mo, n1, np, plt, x1, y1):
    # For simpson's rule plot 
    # https://www.youtube.com/watch?v=WM3GXyHGGUQ 
    x_simpson = np.linspace(a1, b1, n1+1)
    y_simpson = g(x_simpson)

    j = 0
    while j in range(n1):
        x_quad = np.linspace(x_simpson[j], x_simpson[j+2], 25)
        quadratic_func = interp1d(x_simpson, y_simpson, kind='quadratic')
        plt.plot(x_quad, quadratic_func(x_quad), linestyle='dashed', label='Simpson approx.')
        plt.fill_between(x_quad, quadratic_func(x_quad), alpha=0.15)
        j = j + 2


    # plot of the funtion
    plt.plot (x1, y1, label = '$g(x)=xe^{-0.5x}$')

    plt.title("Simpson's Rule")
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.legend()

    graph_simpson = mo.mpl.interactive(plt.gcf())
    return (graph_simpson,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visuals
    """)
    return


@app.cell
def _(graph_simpson, mo, num_slice_in, simpson_results, simpson_title):
    mo.vstack([simpson_title, num_slice_in, simpson_results, graph_simpson])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trapezodial vs Simpson
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code
    """)
    return


@app.cell
def _(mo):
    # get number of slices 
    num_n_com = mo.ui.number(
        start=0,
        stop=100,
        step=2,
        value=2,
        # label="Select an interval $h$ to approximate the integral (1 - 100):",
    )
    num_txt = mo.md(f"""### Select a number of slices $n$ to approximate the integral (must be even integer):""")
    num_txt_com = mo.vstack([num_txt, num_n_com], align="center")
    return num_n_com, num_txt_com


@app.cell
def _(np, num_n_com):
    def f_com(t):
        return 2*t**3 + 2*t**2 - 3*t + 3

    a_com = -1
    b_com = 1

    n_com = int(num_n_com.value)

    # slice width
    h_com = (b_com - a_com)/n_com

    # variables for approximation
    x_com = np.linspace(a_com, b_com, n_com+1)
    y_com = f_com(x_com)

    # Trapezodial Rule 
    trapezodial_com = (h_com/2) * (y_com[0] + (2*sum(y_com[1:n_com])) + y_com[n_com])

    # Simpson's Rule 
    simpsons_com = (h_com/3) * (y_com[0] + 4*sum(y_com[1:n_com:2]) + 2*sum(y_com[2:n_com-1:2]) + y_com[n_com])
    return f_com, n_com, simpsons_com, trapezodial_com, x_com, y_com


@app.cell
def _(mo, simpsons_com, trapezodial_com):
    # Integral Values 

    int_trapezodial_com = mo.md(f"""### Trapezodial approximation: {trapezodial_com}""")

    int_simpson_com = mo.md(f"""### Simpson approximation: {simpsons_com}""")

    actual_int_com = 22/3
    int_com = mo.md(f"""### Actual integral value: {actual_int_com}""")

    com_results = mo.hstack(
                [int_trapezodial_com, int_simpson_com, int_com],
                justify="center", gap=5
            )
    return (com_results,)


@app.cell
def _(f_com, mo, np, plt, x_com, y_com):
    x_func_com = np.linspace(-1.5, 1.5, 100)
    y_func_com = f_com(x_func_com)

    # Trapezodial Rule Plot 

    plt.plot(x_func_com, y_func_com, color='red', label='h(x)=$2x^3 + 2x^2 - 3x + 3$')
    plt.vlines(x_com, ymin=0, ymax=y_com) # draws the verticle lines for the approx
    plt.axhline(y=0, xmin=0, xmax=1, color='black') # draws the horizontal lines for the approx
    plt.plot(x_com, y_com, marker='.', color='black', markersize=10, label='approximation')

    plt.title('Trapezoidal Rule')
    plt.xlabel('x')
    plt.ylabel('h(x)')
    plt.legend()

    graph_trapezodial_com= mo.mpl.interactive(plt.gcf())
    return graph_trapezodial_com, x_func_com, y_func_com


@app.cell
def _(interp1d, mo, n_com, np, plt, x_com, x_func_com, y_com, y_func_com):
    q = 0
    while q in range(n_com):
        x_quad_com = np.linspace(x_com[q], x_com[q+2], 25)
        quadratic_func_com = interp1d(x_com, y_com, kind='quadratic')
        plt.plot(x_quad_com, quadratic_func_com(x_quad_com), linestyle='dashed', label='Simpson approx.')
        plt.fill_between(x_quad_com, quadratic_func_com(x_quad_com), alpha=0.15)
        q = q + 2


    # plot of the funtion
    plt.plot (x_func_com, y_func_com, color='red', label = '$h(x)=2x^3 + 2x^2 - 3x + 3$')

    plt.title("Simpson's Rule")
    plt.xlabel('x')
    plt.ylabel('h(x)')
    plt.legend()

    graph_simpson_com= mo.mpl.interactive(plt.gcf())
    return (graph_simpson_com,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visuals
    """)
    return


@app.cell
def _(com_results, graph_simpson_com, graph_trapezodial_com, mo, num_txt_com):
    compare_title = mo.md(
        r"""
        ## Comparison between Trapezodial and Simpson's rules for $I = \int_{-1}^{1} \ (2x^3 + 2x^2 - 3x + 3) \, dx$"""
    )


    compare_graphs = mo.hstack(
                [graph_trapezodial_com, graph_simpson_com],
                justify="center", gap=5
            )

    mo.vstack([compare_title, num_txt_com, com_results, compare_graphs])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 5.2 - Errors on integrals
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    In numerical calculations there is usually a rounding error. But as these integration rules are only approximations the main source of error is the **approximation error**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trapezodial Rule
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Considering one slice between $x_{k-1}$ and $x_k$, a **Taylor Expansion** of the function can be made.

    $$f(x)=\sum_{n=0}^\infty \left[ \frac{f^n(x_{k-1})}{n!} (x-x_{k-1})^n \right]$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The taylor expression is then integrated form $x_{k-1}$ to $x_k$.

    $$ \int_{x_{k-1}}^{x_k} f(x) \, dx$$

    $$ ⋮ $$

    $$ =  \frac{1}{2} h \left[ f(x_{k-1}) + f(x_k)\right] + \frac{1}{4}h^2 \left[ f'(x_{k-1}) - f'(x_k) \right] + \frac{1}{12}h^3 \left[ f''(x_{k-1}) + f''(x_k)\right] + O(h^4)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This expression can then be summed over all the slices, $k$, to obtain the full integral.

    $$\sum_{k=1}^{N} \int_{x_{k-1}}^{x_k} f(x) \, dx$$

    $$= \frac{1}{2}h \sum_{k=1}^{N} \left[ f(x_{k-1}) + f(x_k) \right] + \frac{1}{4}h^2 \left[ f'(a) - f'(b)\right] + \frac{1}{12}h^3 \sum_{k=1}^{N} \left[ f''(x_{k-1}) + f''(x_k)\right] + O(h^4)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first sum is the trapezodial rule.

    The rest of the series is equal to the approximation error.

    The second term (of $h^2$) is the function evaluated at the first ($a$) and last ($b$) terms. A similar case can be seen for all **even** powers of $h$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The third term (of $h^3$) is the trapezodial rule approximation of the integral of the second derivative of the function.

    $$ \int_{a}^{b} f''(x) \, dx $$

    Multiplying by $\frac{1}{6}h^2$ and substituting $\int f''(x)\, dx = f'(x)$

    $$ ⋮ $$

    The final expression is

    $$\int_{a}^{b} f(x) \, dx $$
    $$= \frac{1}{2}h \sum_{k=1}^{N} \left[ f(x_{k-1}) + f(x_k)\right] + \frac{1}{12}h^2 \left[ f'(a) - f'(b)\right] + O(h^4)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the final expression the terms not constituting the trapezodial rule are the approximation error, $\delta$

    $$ \delta = \frac{1}{12} h^2 \left[ f'(a) - f'(b)\right]$$

    This is the first term of the **Euler–Maclaurin formula**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Rounding Error
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    is the value of the integral multiplied by the machine precision, $\epsilon \approx 10^{-16}$

    Increasing the number of slices reduces the approximation error. However there is no point in increasing the number of slices so that the approximation error becomes smaller than the rounding error as then the roudning error will dominate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Increasing the number of slices helps until the approximation and rounding errors are equal

    $$\frac{1}{12}h^2 \left[ f'(a) - f'(b)\right] \approx \epsilon \int_{a}^{b} f''(x) \, dx $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simpson's Rule
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    approximation error:

    $$\delta = \frac{1}{180} h^4 \left[ f'''(a) - f'''(b)\right]$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    rounding error:

    $$\epsilon \int_{a}^{b} f(x) \, dx$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conclusion

    **Trapezodial rule**
        - 1st order integration rule
        - Increasing n too much will cause rounding error to dominate

    **Simpson's rule**
        - 3rd order integration rule
        - Increasing n too much ($\approx$ a few thousand) slices causes the calculation to reach the limits of precision of the computer
    """)
    return


@app.cell
def _(mo, np):
    def func_e(x_e):
        return x_e**4 -2*x_e+1

    n_e1 = 10
    n_e2 = 20
    a_e = 0
    b_e = 2

    # width of interval
    h_e1 = (b_e-a_e)/n_e1
    h_e2 = (b_e-a_e)/n_e2

    # Values for the approximation 
    x_e1 = np.linspace(a_e, b_e, n_e1+1)
    y_e1 = func_e(x_e1)

    area_e1 = (h_e1/2) * (y_e1[0] + (2*sum(y_e1[1:n_e1])) + y_e1[n_e1])

    x_e2 = np.linspace(a_e, b_e, n_e2+1)
    y_e2 = func_e(x_e2)

    area_e2 = (h_e2/2) * (y_e2[0] + (2*sum(y_e2[1:n_e2])) + y_e2[n_e2])

    # Values for the function 

    # x_func_e = np.linspace(0, 15, 1000)
    # y_func_e = func_e(x_func_e)

    delta = 1/3 * (area_e2 - area_e1)

    delta_error = mo.md(f"""### The approximation error for the Trapezodial rule is: {delta}""")
    return (delta_error,)


@app.cell
def _(mo):
    integral_e = mo.md(f"""### Approximation error for: $\int_{0}^{2} x^{4}-2x+1 \, dx$ with number of slices $n=10$ and $n=20$""")

    error_formula = mo.md(f"""### $\delta=\frac{1}{3}(I_2-I_1)$ """)
    return (integral_e,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$\delta=\frac{1}{3}(I_2-I_1)$$
    """)
    return


@app.cell
def _(delta_error, integral_e, mo):
    mo.vstack([integral_e, delta_error])
    return


if __name__ == "__main__":
    app.run()
