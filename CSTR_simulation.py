import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# -------------------- CSTR model functions --------------------
def T_to_x2(T, x1):
    Ea, R, k0 = 718, 0.08206, 7.2e10
    return x1 * k0 * np.exp(-Ea / R / T)

def x2_to_T(x2, x1):
    Ea, R, k0 = 718, 0.08206, 7.2e10
    return (Ea / R) / np.log(k0 * x1 / x2)

def f1(x1):
    q, v = 100, 100
    C_Ai = 1
    return (q / v) * (C_Ai - x1)

def dx1_dt(x1, x2):
    g1 = -1
    return f1(x1) + g1 * x2

def f2(x1, x2):
    Ea, R, k0 = 718, 0.08206, 7.2e10
    rho, C, deltaH_R = 1000, 0.239, -2.0e4
    q, v, Ti = 100, 100, 350
    x1_sp = 0.1
    x2_sp = f1(x1_sp)
    T_sp = x2_to_T(x2_sp, x1_sp)
    m_sp = -(rho * q * C) * (Ti - T_sp) - (-deltaH_R) * k0 * np.exp(-Ea / R / T_sp) * x1_sp * v
    T = x2_to_T(x2, x1)
    dilution_term = (q / v) * (Ti - T)
    reaction_term = (-deltaH_R * x2) / (rho * C)
    base_term = m_sp / (v * rho * C)
    return x2 * (R / Ea) * (np.log(k0 * x1 / x2)) ** 2 * (dilution_term + reaction_term + base_term) + (x2 / x1) * dx1_dt(x1, x2)

def g2(x1, x2):
    Ea, R, k0 = 718, 0.08206, 7.2e10
    rho, C = 1000, 0.239
    v = 100
    return x2 * (R / Ea) * (np.log(k0 * x1 / x2)) ** 2 / (v * rho * C) * 1e6

def dx2_dt(x1, x2, m):
    return f2(x1, x2) + g2(x1, x2) * m

def x2star(x1, x1_star_val, e1_integral, k_p1, k_i1):
    c1_val = k_p1 * (x1_star_val - x1) + k_i1 * e1_integral
    g1 = -1
    return (c1_val - f1(x1)) / g1

def dx2star_dt(x1, x2, y, k_p1, k_i1):
    q, v = 100, 100
    df1dt = (-q / v) * dx1_dt(x1, x2)
    g1 = -1
    return (-k_p1 * dx1_dt(x1, x2) + k_i1 * (y - x1) - df1dt) / g1

def m(x1, x2, x1_star_val, e1_integral, e2_integral, k_p1, k_i1, k_p2, k_i2):
    x2_star_val = x2star(x1, x1_star_val, e1_integral, k_p1, k_i1)
    c2_val = k_p2 * (x2_star_val - x2) + k_i2 * e2_integral
    x2_star_dot = dx2star_dt(x1, x2, x1_star_val, k_p1, k_i1)
    f2_val = f2(x1, x2)
    return (c2_val + (-1)*(x1_star_val - x1) + x2_star_dot - f2_val) / g2(x1, x2)

def CSTR_statespace(states, x1_sp, controller_parameters):
    x1, x2, e1_integral, e2_integral = states
    k_p1, k_i1, k_p2, k_i2 = controller_parameters
    x2_star_val = x2star(x1, x1_sp, e1_integral, k_p1, k_i1)
    m_val = m(x1, x2, x1_sp, e1_integral, e2_integral, k_p1, k_i1, k_p2, k_i2)
    return np.array([
        dx1_dt(x1, x2),
        dx2_dt(x1, x2, m_val),
        x1_sp - x1,
        x2_star_val - x2
    ])

def simulate_CSTR(k_p1, k_i1, k_p2, k_i2, x1_sp=0.1, t_max=60.0, dt=0.1):
    t_points = int(t_max / dt)
    t = np.linspace(0.0, t_max, t_points)
    T_initial = 350
    x1_initial = 0.25
    x2_initial = T_to_x2(T_initial, x1_initial)

    x1_vals = np.full(t_points, x1_initial)
    x2_vals = np.full(t_points, x2_initial)
    T_vals = np.full(t_points, T_initial)
    m_vals = np.zeros(t_points)
    e1_integral, e2_integral = 0.0, 0.0

    controller_parameters = [k_p1, k_i1, k_p2, k_i2]

    def CSTR_simulator(states, t):
        return CSTR_statespace(states, x1_sp, controller_parameters)

    for i in range(1, t_points):
        m_vals[i-1] = m(x1_vals[i-1], x2_vals[i-1], x1_sp, e1_integral, e2_integral, k_p1, k_i1, k_p2, k_i2)
        sol = odeint(CSTR_simulator, np.array([x1_vals[i-1], x2_vals[i-1], e1_integral, e2_integral]), np.linspace(t[i-1], t[i], 5))
        x1_vals[i], x2_vals[i], e1_integral, e2_integral = sol[-1]
        T_vals[i] = x2_to_T(x2_vals[i], x1_vals[i])

    return t, x1_vals, x2_vals, m_vals

# -------------------- RSFG Method --------------------
def F_sum_func_for_control(x, controller_params):
    _, x1_vals, x2_vals, m_vals = simulate_CSTR(*controller_params)
    x1_sp = 0.1
    x2_sp = f1(x1_sp)
    m_sp = 5.0
    return np.mean(np.abs(x1_vals - x1_sp) + np.abs(x2_vals - x2_sp) + np.abs(m_vals - m_sp))

def G_mu(F_func, xi, mu):
    u = np.random.normal(0, 1, size=xi.shape)
    return (F_func(None, xi + mu*u) - F_func(None, xi)) * u / mu

def RSFG_iter(F_func, mu, xi0, gamma=0.2, num_iter=20):
    path = [xi0]
    xi = xi0.copy()
    for _ in range(num_iter):
        grad = G_mu(F_func, xi, mu)
        xi = xi - gamma * grad
        path.append(xi)
    return np.array(path)

# -------------------- Compare Controllers --------------------
controller_params_list = [
    [3.0, 0.1, 1.5, 0.1],
    [4.0, 0.2, 2.0, 0.2],
    [2.5, 0.05, 1.2, 0.15]
]

mu = 0.01

def evaluate_controller_performance(controller_params):
    F_wrapped = lambda x, xi: F_sum_func_for_control(x, xi)
    xi0 = np.random.uniform(2, 12, 4)
    rsfg_result = RSFG_iter(F_wrapped, mu, xi0, num_iter=60)
    return F_sum_func_for_control(None, controller_params)

# Plot bar chart for predefined controllers
performance = [evaluate_controller_performance(p) for p in controller_params_list]
labels = [f'P{i+1}' for i in range(len(controller_params_list))]

plt.figure(figsize=(8, 6))
plt.bar(labels, performance)
plt.ylabel('Final Error')
plt.title('Controller Performance Comparison')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# -------------------- RSFG Optimization of Controller --------------------
xi0 = np.random.uniform(0.1, 5.0, 4)
rsfg_control = RSFG_iter(F_sum_func_for_control, mu, xi0, num_iter=60)

# Plot cost over iteration
cost_vals = [F_sum_func_for_control(None, xi) for xi in rsfg_control]
plt.figure(figsize=(10, 6))
plt.plot(cost_vals, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('RSFG Optimization of Controller Parameters')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print final optimized parameters
kp1, ki1, kp2, ki2 = rsfg_control[-1]
print("✅ Optimized Controller Parameters:")
print(f"kp1 = {kp1:.4f}, ki1 = {ki1:.4f}, kp2 = {kp2:.4f}, ki2 = {ki2:.4f}")
