import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from casadi import *

def solve_OCP(x_hat, x_target, obstacles, K, h):
    n = 3 # state dimension
    m = 2 # control dimension

    no_steps = int(floor(K/h))

    # Constraints for all k
    u1_max = 1
    u1_min = -1
    u2_max = 1
    u2_min = -1

    opti = Opti()
    x = opti.variable(n, no_steps+1)  # State trajectory
    u = opti.variable(m, no_steps)    # Control trajectory
    # z = opti.variable(1, no_steps)

    # Specify the initial condition
    opti.subject_to(x[:, 0] == x_hat)

    cost = 0
    for k in range(no_steps):
        for obstacle in obstacles:
            opti.subject_to((x[0, k] - obstacle['centre'][0])**2 + (x[1, k] - obstacle['centre'][1])**2 >= obstacle['radius']**2)
        # cost += z[:, k] * h
        # opti.subject_to(z[:, k] >= u[0, k])
        # opti.subject_to(z[:, k] >= -u[0, k])

        cost += sqrt( (x[0, k+1] - x[0, k])**2 + (x[1, k+1] - x[1, k])**2 + 1e-6) # minimuse arc length

        x_next = get_x_next(x[:, k], u[:, k], h)
        opti.subject_to(x[:, k+1] == x_next)

    # cost += (x[0, -1] - x_target[0, -1])**2 + (x[1, -1] - x_target[1, -1])**2
    opti.subject_to(x[0, -1] == x_target[0, -1])
    opti.subject_to(x[1, -1] == x_target[1, -1])

    # constraints for every time step
    opti.subject_to(opti.bounded(u1_min, u[0,:], u1_max))
    opti.subject_to(opti.bounded(u2_min, u[1,:], u2_max))

    opts = {"ipopt.print_level": 5, "print_time": 0}
    # opts = {
    #     "ipopt.print_level": 5,
    #     "ipopt.tol": 1e-10,  # Tighter tolerance for precision
    #     "ipopt.mu_strategy": "adaptive",  # Better barrier handling
    #     "ipopt.hessian_approximation": "limited-memory",  # For large problems
    #     "ipopt.bound_relax_factor": 1e-6,  # Slight bound relaxation
    #     "expand": True
    # }
    opti.minimize(cost)
    opti.solver("ipopt", opts)
    
    solution = opti.solve()
    x_opt = solution.value(x)
    u_opt = solution.value(u)

    return x_opt, u_opt

def f(x,u):
    return vertcat(u[0]*cos(x[2]),
                   u[0]*sin(x[2]),
                   u[1])

def get_x_next(x, u, h):
    # RK4 step for car
    k1 = f(x,u)
    k2 = f(x + h*(k1/2), u)
    k3 = f(x + h*(k2/2), u)
    k4 = f(x + h*k3, u)

    return x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# def get_x_next(x, u, h):
#     # Euler
#     return x + h*f(x,u)

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

def plot_solution(x_tot, u_tot, obstacles):
    # this takes lists of numpy arrays...
    fig = plt.figure(figsize=(10, 8))
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

    ax1.step(time, x1, 'b-', label='x1')
    ax2.step(time, x2, 'b-', label='x2')
    ax3.step(time, x3, 'b-', label='x3 (heading)')
    ax1.set_xlabel('Time step')
    ax2.set_xlabel('Time step')
    ax3.set_xlabel('Time step')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylabel('x1')
    ax1.legend()
    ax1.grid(True)
    ax2.set_ylabel('x2')
    ax2.legend()
    ax2.grid(True)    
    ax3.set_ylabel('x3 (heading)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot control versus time
    u1 = u_tot[0]
    u2 = u_tot[1]
    time = np.arange(u_tot.shape[1] + 1)
    ax4.step(time, np.append(u1[0], u1), 'g-', label='u1 (velocity)')
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Control input')
    ax4.legend()
    ax4.grid(True)
    ax5.step(time, np.append(u2[0], u2), 'g-', label='u2 (turning rate)')
    ax5.set_xlabel('Time step')
    ax5.set_ylabel('Control input')
    ax5.legend()
    ax5.grid(True)

    # Plot the position in state space
    fig = plt.figure(figsize=(10, 8))
    gs_position_space = fig.add_gridspec(1, 1)
    ax6 = fig.add_subplot(gs_position_space[0])
    ax6.plot(x1, x2, 'k-', label='Position')
    ax6.set_xlabel('x1')
    ax6.set_ylabel('x2')
    ax6.legend()
    ax6.grid(True)
    ax6.set_aspect('equal')

    # Plot the initial and final states
    ax6.plot([x1[0], x1[-1]], [x2[0], x2[-1]], 'k.')

    # Plot the obstacles
    theta = np.linspace(0, 2*np.pi, 50)
    for obstacle in obstacles:
        x = obstacle['centre'][0] + obstacle['radius'] * np.cos(theta)
        y = obstacle['centre'][1] + obstacle['radius'] * np.sin(theta)
        ax6.plot(x, y, 'k-')

    plt.tight_layout()
    # plt.savefig("ocp-open-loop.svg", format="svg")
    plt.show()