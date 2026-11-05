import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

n0 = 1e23  # m^-3
k1 = 1e6  # s^-1
k2 = 1e-16  # m^3 / s
t_max = 1e-6  # s

# Task 1 - ODE of the carrier density as a function of time
def dndt(t, n, k1, k2):
    n_t = -k1*n-k2*n**2
    return n_t

# Task 2 
t_eval = np.linspace(0, t_max, 500) # default solver returns only 8 time points
scipy_sol=solve_ivp(dndt, (0, t_max), [n0],args=(k1, k2), t_eval=t_eval, rtol=1e-10, atol=1e-10 * n0) # rtol and atol are set to be very small to ensure accuracy

n_scipy = scipy_sol.y[0]
pl_scipy = n_scipy**2
pl_scipy_norm = pl_scipy / np.max(pl_scipy)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(scipy_sol.t, pl_scipy_norm)
ax[0].set(xlabel="time", ylabel="PL intensity (normalized)", title="PL intensity vs time")
ax[1].semilogy(scipy_sol.t, pl_scipy_norm)
ax[1].set(xlabel="time", ylabel="PL intensity (normalized)", title="PL intensity vs time (log scale)")
plt.tight_layout()
plt.show()

# Task 3 

#Euler step
def euler_step(f, x, t, h):
    """Perform a single step h using Euler's scheme."""
    return x + h * f(x, t)

#RK4 step
def rk4_step(f, x, t, h):
    """Perform a single step h using the 4th order Runge-Kutta scheme."""
    k1 = h * f(x, t)
    k2 = h * f(x + k1 / 2.0, t + h / 2.0)
    k3 = h * f(x + k2 / 2.0, t + h / 2.0)
    k4 = h * f(x + k3, t + h)
    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

#ODE solver
def ode_solve(step, f, x0, t0, h, nsteps):
    """Solve dx/dt = f(x,t) from t0 to t0 + nsteps*h with a fixed-step method."""
    t = t0 + h * np.arange(nsteps + 1)
    x = np.zeros((nsteps + 1,) + np.shape(x0))
    x[0] = x0
    for i in range(nsteps):
        x[i + 1] = step(f, x[i], t[i], h)
    return t, x

N = 20
h = t_max / N
f_n = lambda n, t: dndt(t, n, k1, k2)

t_euler, n_euler = ode_solve(euler_step, f_n, n0, 0.0, h, N)
t_rk4,   n_rk4   = ode_solve(rk4_step,   f_n, n0, 0.0, h, N)

#Comparison of the two methods with the scipy solution
def norm_pl(n):
    pl = n**2
    return pl / pl.max()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a in ax:
    a.plot(scipy_sol.t, pl_scipy_norm, "k", label="scipy")
    a.plot(t_euler, norm_pl(n_euler), "o--", ms=3, label=f"Euler (N={N})")
    a.plot(t_rk4, norm_pl(n_rk4), "s--", ms=3, label=f"RK4 (N={N})")
    a.set(xlabel="time", ylabel="normalised PL")
    a.legend()
ax[1].set_yscale("log")
plt.tight_layout()
plt.show()

# Task 4 - Rewrite RK4 using JAX's scan function
def jax_sol(step, f, x0, t0, h, nsteps):
    def body(x, t): # one time step, replaces the for loop in ode_solve
        x_new = step(f, x, t, h)
        return x_new, x_new

    t_steps = t0 + h * jnp.arange(nsteps)
    _, x_rest = jax.lax.scan(body, jnp.asarray(x0, dtype=float), t_steps)

    t = t0 + h * jnp.arange(nsteps + 1)
    x = jnp.concatenate([jnp.array([x0], dtype=float), x_rest])
    return t, x

t_jax, n_jax = jax_sol(rk4_step, f_n, n0, 0.0, h, N)

print(jnp.max(jnp.abs(n_jax - n_rk4) / n_rk4))     # should be around 1e-16

# Task 5 - differential decay time
def ln_pl(T, N=20):
    # RK4 from 0 to T in N steps (h depends on T, so JAX can differentiate with respect to T)
    t, n = jax_sol(rk4_step, f_n, n0, 0.0, T / N, N)
    return jnp.log(n[-1]**2 / n0**2)          # ln of normalised PL at time T

dlnpl_dt = jax.vmap(jax.grad(ln_pl))          # grad for one T, vmap for an array of T

T = jnp.linspace(1e-9, t_max, 200)            # start slightly above 0
tau_diff = -2.0 / dlnpl_dt(T)                 # tau_diff = -2 [d ln(PL)/dt]^-1

plt.plot(T * 1e9, tau_diff * 1e9)
plt.xlabel("time (ns)")
plt.ylabel(r"$\tau_{\rm diff}$ (ns)")
plt.show()

# Task 6 
# Task 7 

