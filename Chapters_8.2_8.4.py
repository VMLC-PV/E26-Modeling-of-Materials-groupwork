import marimo

__generated_with = "0.25.1"
app = marimo.App(layout_file="layouts/lec_6_chapters_8.2_8.4.slides.json")


@app.cell
def _():
    from math import sin
    from numpy import array, arange
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo

    return arange, array, mo, plt, sin


@app.cell
def _(mo):
    slide_1 = mo.md(
        r"""
        ## Chapter 8.2 

        We have seen Euler's method and the Runge-Kutta method for $1^{st}$ order differential equations with one variable.

        ### But what if we have multiple variables? 
        #### &rarr; **simultaneous differential equations**
        """
    )
    mo.vstack([slide_1])
    return


@app.cell
def _(mo):
    slide_2 = mo.md(
        r"""
        ## Simultaneous Differential Equations

        * #### Only one independant variable
        * #### Derivative of each variable depends on independent variable and any/all other variables 
        """
    )
    mo.vstack([slide_2])
    return


@app.cell
def _(mo):
    slide_2a = mo.md(
        r"""
        #### The general form is:
        $$\frac{dx}{dt} = f_{x}(x, y, t) \ , \ \ \ \ \frac{dy}{dt} = f_{y}(x, y, t)$$
        An example: 
        $$\frac{dx}{dt} = xy - x \ , \ \ \ \ \frac{dy}{dt}=y - xy + \sin^{2}(\omega t)$$
        """
    )
    mo.vstack([slide_2a])
    return


@app.cell
def _(mo):
    slide_3 = mo.md(
        r"""
        ## Vector Notation
        $$\frac{d\mathbf{r}}{dt} = \mathbf{f} \ (\mathbf{r}, \ t)$$

        * **r** is a vector of the non-independent variables, $\mathbf{r}=(x, \ y, \ ...)$
        * **f** is a vector of functions, $\mathbf{f} \ (\mathbf{r}, \ t)=(f_{x} \ (\mathbf{r}, t), \ f_{y} (\mathbf{r}, t), \ ...)$
        """
    )

    mo.vstack([slide_3])
    return


@app.cell
def _(mo):
    slide_4 = mo.md(
        r"""
        ## Taylor Expansion of r

        $$\mathbf{r}(t + h) = \mathbf{r}(t) + h \frac{d\mathbf{r}}{dt}+O(h^2)$$

        $$= \mathbf{r}(t) + h \mathbf{f}(\mathbf{r}, \ t) + O(h^2) $$

        * $O(h^2)$ includes terms that are $h^2$ or higher

        """
    )

    mo.vstack([slide_4])
    return


@app.cell
def _(mo):
    slide_4a = mo.md(
        r"""
        #### Euler's Method:

        $$\mathbf{r}(t + h) = \mathbf{r}(t) + h \mathbf{f} (\mathbf{r}, t)$$

        #### Derive 4th Order Multi-variabled Runge-Kutta Method:


        $$\mathbf{r}(t + h) = \mathbf{r}(t) + \frac{1}{6}(\mathbf{k_1} + 2\mathbf{k_2} + 2\mathbf{k_3} + \mathbf{k_4})$$

        """
    )

    mo.vstack([slide_4a])
    return


@app.cell
def _(mo):
    slide_4b = mo.md(
        r"""
        #### Vector generalisation of the single variabled Runge-Kutta Method

        $$\mathbf{k_1} = h\mathbf{f}(\mathbf{r}, \ t)$$
        $$\mathbf{k_2} = h\mathbf{f}(\mathbf{r} + \frac{1}{2}\mathbf{k_1}, \ t + \frac{1}{2}h)$$
        $$\mathbf{k_3} = h\mathbf{f}(\mathbf{r} + \frac{1}{2}\mathbf{k_2}, \ t + \frac{1}{2}h)$$
        $$\mathbf{k_4} = h\mathbf{f}(\mathbf{r} + \mathbf{k_3}, \ t+h)$$

        """
    )

    mo.vstack([slide_4b])
    return


@app.cell
def _(mo):
    slide_4c = mo.md(
        r"""
        ## What the programme does:
        * #### Uses arrays to represent vectors 
        * #### Takes vector argument $\mathbf{r}$ as an input 
        * #### $f(\mathbf{r}, \ t)$ is split into the two non-independent variables, $f_x$ and $f_y$
        * #### The two values are produced as an output together in an array 

        """
    )

    mo.vstack([slide_4c])
    return


@app.cell
def _(array, sin):
    def f(r,t):
        x = r[0]
        y = r[1]
        fx = x*y - x
        fy = y - x*y + sin(t)**2
        return array([fx, fy], float)


    return (f,)


@app.cell
def _(mo):
    t1_slider = mo.ui.slider(
        start=0,
        stop=10,
        step=1,
        value=0,
        label="$t_1$ value:",
    )

    t2_slider = mo.ui.slider(
        start=0,
        stop=10,
        step=1,
        value=10,
        label="$t_2$ value:",
    )
    return t1_slider, t2_slider


@app.cell
def _(arange, array, f, plt, t1_slider, t2_slider):
    a = t1_slider.value
    b = t2_slider.value
    N = 1000
    #interval
    h = (b-a)/N

    # creates array of evenly spaced values
    tpoints = arange(a, b, h)
    xpoints = []
    ypoints = []

    r = array ([1.0, 1.0], float)
    for t in tpoints:
        xpoints.append(r[0])
        ypoints.append(r[1])
        k1 = h*f(r, t)
        k2 = h*f(r+0.5*k1, t+0.5*h)
        k3 = h*f(r+0.5*k2, t+0.5*h)
        k4 = h*f(r+k3, t+h)
        r += (k1+2*k2+2*k3+k4)/6

    plt.plot(tpoints, xpoints, color="#f55f74", label="x-component")
    plt.plot(tpoints, ypoints, color="#50ad9f", label="y-component")
    plt.title("Example 8.5: Simultaneous Ordinary Differential Equations")
    plt.xlabel("t")
    plt.legend()
    # plt.show()
    example_8_5 = plt.gca()
    return (example_8_5,)


@app.cell
def _(example_8_5, mo, t1_slider, t2_slider):
    slide_5 = mo.md(
        r"""
        #### Example 8.5:
        """
    )
    mo.vstack([mo.hstack([t1_slider, t2_slider], align="center", justify="center"),
               example_8_5], align="center", justify="center")
    return


@app.cell
def _(mo):
    slide_84_1 = mo.md(
        r"""
        ## Chapter 8.4 

        We have seen single variabled, multi variabled, and 2nd order differential equations. They have followed similar solving techniques and have thus far all used a **fixed step size** $h$. 

        ### But what if we have a function with regions that vary at different "speeds"?
        #### &rarr; **Adaptive Step Size**
        """
    )
    mo.vstack([slide_84_1])
    return


@app.cell
def _(mo):
    slide_84_2 = mo.md(
        r"""
        ## When to use adaptive step sizing?
        """
    )

    slide_84_2a = mo.md(
        r"""
        * slow varying &rarr; large $h$
        * fast varying &rarr; small $h$
        """
    )

    _src = ("ADD FILE LOCATION")

    #graph_84 = mo.image(src=_src, width="575px", height="371px")

    graph_84_book_example = mo.image(src=_src, width="546px", height="353px")


    mo.vstack([slide_84_2, mo.hstack([graph_84_book_example, slide_84_2a], align="center", justify="center") ])
    return


@app.cell
def _(mo):
    slide_84_3 = mo.md(
        r"""
        ## Adaptive Step Sizing 

        ### The error per interval of $t$ remains constant

        Bearing in mind that:

        Using a smaller step size will require more steps and thus increase the error

        &rarr; Therefore each individual step will have to be more accurate overall 
        """
    )

    mo.vstack([slide_84_3])
    return


@app.cell
def _(mo):
    slide_84_4 = mo.md(
        r"""
        ## How to find the Step Size: 

        * #### Estimate the error on the step 
        * #### Find error of the required accuracy 
        * #### Compare and increase/decrease step size to match the accuracies 

        """
    )

    mo.vstack([slide_84_4])
    return


@app.cell
def _(mo):
    slide_84_5 = mo.md(
        r"""
        ## Find Step Size Error: 

        * #### Small initial $h$
        * #### Do two steps of $h$ &rarr; from $t$ to $t + 2h$ &rarr; estimate of $x(t+2h) = x_1$
        * #### Back at $t$ do one step of $2h$ &rarr; estimate of $x(t+2h) = x_2$
        * #### **Difference &rarr; error**

        """
    )

    mo.vstack([slide_84_5])
    return


@app.cell
def _(mo):
    slide_84_6 = mo.md(
        r"""
        ## Fourth Order Runge-Kutta Error:

        * #### $ch^5$ &rarr; ($5^{th}$ Order)
        * #### For 2 steps: $\ x_1 + 2ch^5$
        * #### For 1 step: $\ x_2 + c(2h)^5 = x_2 + 32ch^5$
        * #### $x_1 = x_2 + 30ch^5$
        * #### **Error per Step:** $\ ch^5 = \frac{1}{30} (x_1 - x_2) = \delta$

        """
    )

    mo.vstack([slide_84_6])
    return


@app.cell
def _(mo):
    slide_84_7 = mo.md(
        r"""
        ### What step size do we need to get the target error?

        * ##### the 'perfect' step size: $\ h'$
        * ##### the 'perfect' error: $\ \delta'= ch'\ ^5$
        $$ \delta' = \frac{1}{30} |x_1 - x_2| \left(\frac{h'}{h} \right)^5$$
        * ##### $\Delta$ target accuracy per time unit **&rarr; $\Delta h'$**

        """
    )

    mo.vstack([slide_84_7])
    return


@app.cell
def _(mo):
    slide_84_8 = mo.md(
        r"""
        ## Finding $h'$ \ :

        $$h' = h(\frac{30h\Delta}{|x_1 - x_2|})^{\frac{1}{4}}$$
        $$= h \rho ^{\frac{1}{4}}$$

        * #### $\rho$ is the ratio used to determine if the accuracy is sufficient 
        """
    )

    mo.vstack([slide_84_8])
    return


@app.cell
def _(mo):
    slide_84_9 = mo.md(
        r"""
        ### What does $\rho$ tell us?

        $$\mathbf {\rho > 1}$$
        * The accuracy of the current Runge-Kutta step is **better** than the target accuracy
        * "wasteful" &rarr; continue to next time-point but with a bigger step 
    
        substituting the current value of $\rho$ into $h' = h \rho ^{\frac{1}{4}}$
        * tells what the next step value needs to be 

        """
    )

    mo.vstack([slide_84_9])
    return


@app.cell
def _(mo):
    slide_84_10 = mo.md(
        r"""
    
        $$\mathbf {\rho < 1}$$
        * The accuracy of the current Runge-Kutta step is **worse** than the target accuracy
        * The same step needs to be done again with smaller step size
        * Substituting $\rho$ will give needed step size

        ### Why do the calculation again?
        **&rarr; errors are cumulative**

        """
    )

    mo.vstack([slide_84_10])
    return


@app.cell
def _(mo):
    slide_84_11 = mo.md(
        r"""
    
        ## What if $x_1 \ = \ x_2$ ?
        * $h'$ can become very large and/or diverge 

        **Preventative Step:**
        * place an upper limit on $h$ that can not be exceeded for each step 
        * rule of thumb: should not exceed a factor of 2 for a pair of successive steps

        """
    )

    mo.vstack([slide_84_11])
    return


@app.cell
def _(mo):
    slide_84_12 = mo.md(
        r"""
    
        ## Can we use adaptive steps for differential equations? 

        * Each variable will have seperated errors, ex:
        $$\delta_x=\frac{1}{30}(x_1 - x_2), \ \delta_y=\frac{1}{30}(y_1 - y_2)$$

        #### When determining $\rho$:
        * what kind of error do we need? 
        * does the variable impact the overall accuracy?

        """
    )

    mo.vstack([slide_84_12])
    return


@app.cell
def _(mo):
    slide_84_13 = mo.md(
        r"""
    
        ## Adaptive sizing is more work so is it worth it? 
        * For every $n$ amount of Runge-Kutta steps we use to calculate 
        &rarr; we need to do $n+1$ Runge-Kutta steps 
        * If target accuracy is missed &rarr; it has to be repeated 

        #### However:
        * Achieve the desired accuracy 
        * Takes less computational time
        &rarr; use larger steps when the solution changes slowly

        """
    )

    mo.vstack([slide_84_13])
    return


@app.cell
def _(mo):
    slide_84_14 = mo.md(
        r"""
    
        ## Local Extrapolation 

        We have the solution for the diff eq. at $t+2h$:
        $$x(t+2h) = x_1 + 2ch^5 + O(h^6) $$
        We know the per step error:
        $$ch^5 = \frac{1}{30}(x_1-x_2)$$

        $$x(t+2h) = x_1 + \frac{1}{15}(x_1-x_2) + O(h^6)$$

        One order better &rarr; Accurate to order of $h^5$ with error $h^6$

        """
    )

    mo.vstack([slide_84_14])
    return


if __name__ == "__main__":
    app.run()
