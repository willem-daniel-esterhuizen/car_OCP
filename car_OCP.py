import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import casadi as ca
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
# from IPython.display import HTML

def build_and_check_scaling(x_init, x_target, obstacles, constraints, T, h, warm_start, eps=1e-4, delta=1e-3):
    n = 3
    m = 2

    cost_scaling = get_arc_length(np.hstack((x_init[0:2], x_target[0:2])))

    no_steps = int(ca.floor(T/h))

    u1_max = constraints['u1_max']
    u1_min = constraints['u1_min']
    u2_max = constraints['u2_max']
    u2_min = constraints['u2_min']

    opti = ca.Opti()
    x = opti.variable(n, no_steps+1)
    u = opti.variable(m, no_steps)

    # Specify the initial condition
    opti.subject_to(x[:, 0] == x_init)

    cost = 0
    for k in range(no_steps):
        for obstacle in obstacles:
            opti.subject_to(( (x[0, k] - obstacle['centre'][0]) / (cost_scaling) )**2 + ( (x[1, k] - obstacle['centre'][1]) / (cost_scaling) )**2 >= (obstacle['radius'] / (cost_scaling) )**2)

        opti.subject_to(x[:, k+1] == get_x_next(x[:, k], u[:, k], h))

        cost += ( ca.sqrt( (x[0, k+1] - x[0, k])**2 + (x[1, k+1] - x[1, k])**2 + eps) + delta*(u[0, k]**2 + u[1, k]**2) ) / (cost_scaling)

    opti.subject_to(x[0, -1] == x_target[0, -1])
    opti.subject_to(x[1, -1] == x_target[1, -1])

    opti.subject_to(opti.bounded(u1_min, u[0,:], u1_max))
    opti.subject_to(opti.bounded(u2_min, u[1,:], u2_max))

    opti.minimize(cost)

    u_warm = np.vstack((warm_start['u1_warm'], 
                        warm_start['u2_warm']))
    x_warm = np.vstack((warm_start['x1_warm'],
                        warm_start['x2_warm'],
                        warm_start['x3_warm']))
    
    x0_vec = ca.vertcat(ca.vec(x_warm), ca.vec(u_warm))
    print_scaling_diagnostics(opti, x0_vec)

def solve_OCP(x_init, x_target, obstacles, constraints, T, h, warm_start=None, eps=1e-4, delta=1e-3):
    n = 3
    m = 2

    cost_scaling = get_arc_length(np.hstack((x_init[0:2], x_target[0:2])))

    no_steps = int(ca.floor(T/h))

    u1_max = constraints['u1_max']
    u1_min = constraints['u1_min']
    u2_max = constraints['u2_max']
    u2_min = constraints['u2_min']

    opti = ca.Opti()
    x = opti.variable(n, no_steps+1)
    u = opti.variable(m, no_steps)

    # Specify the initial condition
    opti.subject_to(x[:, 0] == x_init)

    cost = 0
    for k in range(no_steps):
        for obstacle in obstacles:
            opti.subject_to(( (x[0, k] - obstacle['centre'][0]) / (cost_scaling) )**2 + ( (x[1, k] - obstacle['centre'][1]) / (cost_scaling) )**2 >= (obstacle['radius'] / (cost_scaling) )**2)

        opti.subject_to(x[:, k+1] == get_x_next(x[:, k], u[:, k], h))

        cost += ( ca.sqrt( (x[0, k+1] - x[0, k])**2 + (x[1, k+1] - x[1, k])**2 + eps) + delta*(u[0, k]**2 + u[1, k]**2) ) / (cost_scaling)

    opti.subject_to(x[0, -1] == x_target[0, -1])
    opti.subject_to(x[1, -1] == x_target[1, -1])

    opti.subject_to(opti.bounded(u1_min, u[0,:], u1_max))
    opti.subject_to(opti.bounded(u2_min, u[1,:], u2_max))

    opts = {"ipopt.print_level": 0, 
            "print_time": 0, 
            "ipopt.sb": "yes",
            "ipopt.nlp_scaling_method": "gradient-based"}

    opti.minimize(cost)
    opti.solver("ipopt", opts)

    if warm_start is not None:
        u_warm = np.vstack((warm_start['u1_warm'], 
                            warm_start['u2_warm']))
        x_warm = np.vstack((warm_start['x1_warm'],
                            warm_start['x2_warm'],
                            warm_start['x3_warm']))

        opti.set_initial(x, x_warm)
        opti.set_initial(u, u_warm)

    solution = opti.solve()

    x_opt = solution.value(x)
    u_opt = solution.value(u)

    return x_opt, u_opt

def get_initial_warm_start(x_init, x_target, T, h):
    no_steps = int(ca.floor(T/h))

    x1_warm = np.linspace(x_init[0][0], x_target[0][0], no_steps + 1)
    x2_warm = np.linspace(x_init[1][0], x_target[1][0], no_steps + 1)
    x3_warm = np.linspace(0, 0, no_steps + 1)

    u1_warm = 0.1*np.ones(no_steps)
    u2_warm = np.zeros(no_steps)

    return {
        'x1_warm' : x1_warm,
        'x2_warm' : x2_warm,
        'x3_warm' : x3_warm,
        'u1_warm' : u1_warm,
        'u2_warm' : u2_warm
    }

def print_scaling_diagnostics(opti, x0_vec):
    f = opti.f       # objective (scalar)
    g = opti.g       # constraints (vector)
    x = opti.x       # decision variables

    grad_f = ca.jacobian(f, x)           # ∇f (row vector)
    jac_g  = ca.jacobian(g, x)           # Jacobian of constraints
    hess_f = ca.hessian(f, x)[0]         # Hessian of objective (Lagrangian approx if bounds)

    diag_fun = ca.Function('diag', [x], [f, g, grad_f, jac_g, hess_f])

    f0, g0, grad_f0, jac_g0, hess_f0 = diag_fun(x0_vec)

    print("=== SCALING DIAGNOSTICS ===")
    print(f"Objective J          : {float(f0):.3e}")
    # print(f"||∇J||_inf           : {float(ca.norm_inf(grad_f0)):.3e}")
    print(f"||∇J||_2             : {float(ca.norm_2(grad_f0)):.3e}")
    print(f"||Hess J||_Frob      : {float(ca.norm_fro(hess_f0)):.3e}")
    # print(f"||Hess J||_inf       : {float(ca.norm_inf(hess_f0)):.3e}")
    # print(f"||g||_inf            : {float(ca.norm_inf(g0)):.3e}")
    print(f"||g||_2              : {float(ca.norm_2(g0)):.3e}")
    print(f"||Jac g||_Frob       : {float(ca.norm_fro(jac_g0)):.3e}")
    # print(f"||Jac g||_inf        : {float(ca.norm_inf(jac_g0)):.3e}")
    print("============================")

def get_arc_length(x):

    arc_length = 0
    for k in range(1, x.shape[1]):
        arc_length += np.linalg.norm(x[:2, k] - x[:2, k - 1])
    return arc_length

def f(x,u):
    return ca.vertcat(u[0]*ca.cos(x[2]),
                      u[0]*ca.sin(x[2]),
                      u[1])

def get_x_next(x, u, h):
    # RK4 step for car
    k1 = f(x,u)
    k2 = f(x + h*(k1/2), u)
    k3 = f(x + h*(k2/2), u)
    k4 = f(x + h*k3, u)

    return x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def plot_constraints(ax, x_1_max, x_1_min, x2_init_min, x2_init_max):
    ax.plot([x_1_min, x_1_min], [x2_init_min, x2_init_max], 'k-')
    ax.plot([x_1_max, x_1_max], [x2_init_min, x2_init_max], 'k-')

def plot_solution_hold_on(ax, x_tot):
    x_1 = x_tot[0]
    x_2 = x_tot[1]

    x_1 = np.append(x_1[0], x_1)
    x_2 = np.append(x_2[0], x_2)

    ax.plot(x_1, x_2, 'b-')
    ax.plot([x_1[0], x_1[-1]], [x_2[0], x_2[-1]], 'k.')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.grid(True)

def plot_solution_versus_time(x_tot, u_tot):
    # this takes lists of numpy arrays...
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(5, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax5 = fig.add_subplot(gs[4], sharex=ax1)

    # Plot state versus time
    x1 = x_tot[0]
    x2 = x_tot[1]
    x3 = x_tot[2]

    # Duplicate the initial state so we can have a nice bar plots.
    x1 = np.append(x1[0], x1)
    x2 = np.append(x2[0], x2)
    x3 = np.append(x3[0], x3)
    time = np.arange(x_tot.shape[1] + 1) # because we duplicate the initial state

    ax1.set_title("x_1 vs time", fontsize=14)
    ax2.set_title("x_2 vs time", fontsize=14)
    ax3.set_title("x_3 vs time", fontsize=14)
    ax1.step(time, x1, 'b-', label='x1')
    ax2.step(time, x2, 'b-', label='x2')
    ax3.step(time, x3, 'b-', label='x3 (heading)')
    ax1.set_xlabel('Time step')
    ax2.set_xlabel('Time step')
    ax3.set_xlabel('Time step')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax1.set_ylabel('x1')
    ax1.legend()
    ax1.grid(True)
    # ax2.set_ylabel('x2')
    ax2.legend()
    ax2.grid(True)    
    # ax3.set_ylabel('x3 (heading)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot control versus time
    u1 = u_tot[0]
    u2 = u_tot[1]
    time = np.arange(u_tot.shape[1] + 1)
    ax4.step(time, np.append(u1[0], u1), 'g-', label='u1 (velocity)')
    ax4.set_title("u_1 vs time", fontsize=14)
    ax4.set_xlabel('Time step')
    # ax4.set_ylabel('Control input')
    ax4.legend()
    ax4.grid(True)
    ax5.step(time, np.append(u2[0], u2), 'g-', label='u2 (turning rate)')
    ax5.set_title("u_2 vs time", fontsize=14)
    ax5.set_xlabel('Time step')
    # ax5.set_ylabel('Control input')
    ax5.legend()
    ax5.grid(True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # plt.savefig("ocp-open-loop.svg", format="svg")
    plt.show()

def plot_solution_in_state_space(x_tot, obstacles, title, arc_length, obstacles_only=False, eps=None, delta=None):
    x1 = x_tot[0]
    x2 = x_tot[1]
    fig = plt.figure(figsize=(7, 7))
    gs_position_space = fig.add_gridspec(1, 1)
    ax6 = fig.add_subplot(gs_position_space[0])
    ax6.set_title(title, fontsize=14)
    ax6.set_xlabel('x1')
    ax6.set_ylabel('x2')
    ax6.grid(True)
    ax6.set_aspect('equal')

    if obstacles_only is False:
        ax6.plot(x1, x2, 'k-')

    # Plot the initial and final states
    ax6.plot([x1[0], x1[-1]], [x2[0], x2[-1]], 'k.')
    epsilon=0.05
    ax6.annotate(
        r'$\mathbf{x}^{\mathrm{ini}}$',
        xy=(x1[0], x2[0] + epsilon),
        xytext=(x1[0], x2[0] + epsilon)
    )
    ax6.annotate(
        r'$\mathbf{x}^{\mathrm{tar}}$',
        xy=(x1[-1] - 2*epsilon, x2[-1] + epsilon),
        xytext=(x1[-1] - 2*epsilon, x2[-1] + epsilon)
    )
    ax6.annotate(
        f'Arc length: {arc_length}',
        xy=(-2, 0.7),
        xytext=(-2, 0.7)
    )
    ax6.annotate(
        rf'$\varepsilon$: {eps}',
        xy=(-2, 0.6),
        xytext=(-2, 0.6)
    )
    ax6.annotate(
        rf'$\delta$: {delta}',
        xy=(-2, 0.5),
        xytext=(-2, 0.5)
    ) 

    # Plot the obstacles
    theta = np.linspace(0, 2*np.pi, 50)
    for obstacle in obstacles:
        x = obstacle['centre'][0] + obstacle['radius'] * np.cos(theta)
        y = obstacle['centre'][1] + obstacle['radius'] * np.sin(theta)
        ax6.plot(x, y, 'k-')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

def plot_control(u_tot):
    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot control versus time
    u1 = u_tot[0]
    u2 = u_tot[1]
    time = np.arange(u_tot.shape[1] + 1)
    ax1.step(time, np.append(u1[0], u1), 'g-', label='u1 (velocity)')
    ax1.set_title("u_1 vs time", fontsize=14)
    ax1.set_xlabel('Time step')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(top=1.1, bottom=-1.1)

    ax2.step(time, np.append(u2[0], u2), 'g-', label='u2 (turning rate)')
    ax2.set_title("u_2 vs time", fontsize=14)
    ax2.set_xlabel('Time step')
    ax2.legend()
    ax2.grid(True)
    ax1.set_ylim(top=1.1, bottom=-1.1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

def plot_solution_in_statespace_and_control(x_tot, u_tot, obstacles, arc_length, obstacles_only=False, eps=None, delta=None):
    x1 = x_tot[0]
    x2 = x_tot[1]
    fig = plt.figure(figsize=(7, 9))
    gs_position_space = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    ax6 = fig.add_subplot(gs_position_space[0])
    ax6.set_title('State space', fontsize=14)
    ax6.set_xlabel('x1')
    ax6.set_ylabel('x2')
    ax6.grid(True)
    ax6.set_aspect('equal')

    if obstacles_only is False:
        ax6.plot(x1, x2, 'k-')

    # Plot the initial and final states
    ax6.plot([x1[0], x1[-1]], [x2[0], x2[-1]], 'k.')
    epsilon=0.05
    ax6.annotate(
        r'$\mathbf{x}^{\mathrm{ini}}$',
        xy=(x1[0], x2[0] + epsilon),
        xytext=(x1[0], x2[0] + epsilon)
    )
    ax6.annotate(
        r'$\mathbf{x}^{\mathrm{tar}}$',
        xy=(x1[-1] - 2*epsilon, x2[-1] + epsilon),
        xytext=(x1[-1] - 2*epsilon, x2[-1] + epsilon)
    )
    ax6.annotate(
        f'Arc length: {arc_length}',
        xy=(-2, 0.7),
        xytext=(-2, 0.7)
    )
    ax6.annotate(
        rf'$\varepsilon$: {eps}',
        xy=(-2, 0.6),
        xytext=(-2, 0.6)
    )
    ax6.annotate(
        rf'$\delta$: {delta}',
        xy=(-2, 0.5),
        xytext=(-2, 0.5)
    ) 

    # Plot the obstacles
    theta = np.linspace(0, 2*np.pi, 50)
    for obstacle in obstacles:
        x = obstacle['centre'][0] + obstacle['radius'] * np.cos(theta)
        y = obstacle['centre'][1] + obstacle['radius'] * np.sin(theta)
        ax6.plot(x, y, 'k-')

    ax1 = fig.add_subplot(gs_position_space[1])
    ax2 = fig.add_subplot(gs_position_space[2], sharex=ax1)

    # Plot control versus time
    u1 = u_tot[0]
    u2 = u_tot[1]
    time = np.arange(u_tot.shape[1] + 1)
    ax1.step(time, np.append(u1[0], u1), 'g-', label='u1 (velocity)')
    ax1.set_title("u1 vs time", fontsize=14)
    ax1.set_xlabel('Time step')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(top=1.1, bottom=-1.1)

    ax2.step(time, np.append(u2[0], u2), 'g-', label='u2 (turning rate)')
    ax2.set_title("u2 vs time", fontsize=14)
    ax2.set_xlabel('Time step')
    ax2.legend()
    ax2.grid(True)
    ax1.set_ylim(top=1.1, bottom=-1.1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

def plot_arclength_vs_i(arc_lengths_with_homo, arc_lengths_no_homo):
    fig = plt.figure(figsize=(9, 5))

    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(arc_lengths_with_homo, color='blue', marker='o', label='With homotopy')
    ax1.plot(arc_lengths_no_homo, color='red', marker='o', label='Without homotopy')
    ax1.legend()
    ax1.grid(True)

    ax1.set_xticks(range(len(arc_lengths_no_homo)))
    xlabels = [rf'$\frac{{0.01}}{{2^{{{i}}}}}$' for i in range(len(arc_lengths_no_homo))]
    ax1.set_xticklabels(xlabels)

    ax1.set_title(r'Arc length of solution versus $\varepsilon$')
    ax1.set_xlabel(r'$\varepsilon$')
    ax1.set_ylabel('arc length')

    plt.show()