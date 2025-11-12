import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import casadi as ca

def solve_OCP(x_init, x_target, obstacles, constraints, T, h, warm_start=None, eps_u=1e-6, eps_cost=1e-6, sqrt_cost_flag=True):
    n = 3
    m = 2

    no_steps = int(ca.floor(T/h))

    u1_max = constraints['u1_max']
    u1_min = constraints['u1_min']
    u2_max = constraints['u2_max']
    u2_min = constraints['u2_min']

    opti = ca.Opti()
    x = opti.variable(n, no_steps+1)
    u = opti.variable(m, no_steps)
    if sqrt_cost_flag is False:
        z = opti.variable(1, no_steps)    # If absolute value cost

    # Specify the initial condition
    opti.subject_to(x[:, 0] == x_init)

    cost = 0
    for k in range(no_steps):
        for obstacle in obstacles:
            opti.subject_to((x[0, k] - obstacle['centre'][0])**2 + (x[1, k] - obstacle['centre'][1])**2 >= obstacle['radius']**2)
        
        if sqrt_cost_flag is False:
            cost += z[:, k] + eps_u*(u[0, k]**2 + u[1, k]**2)
            opti.subject_to(z[:, k] >= np.sqrt(u[0, k]**2 + eps_cost) )
        else:
            cost += np.sqrt(eps_cost) * ( ca.sqrt( (x[0, k+1] - x[0, k])**2 + (x[1, k+1] - x[1, k])**2 + eps_cost) + eps_u*(u[0, k]**2 + u[1, k]**2) )
            # cost += ( ca.sqrt( (x[0, k+1] - x[0, k])**2 + (x[1, k+1] - x[1, k])**2 + eps_cost) + eps_u*(u[0, k]**2 + u[1, k]**2) )

        x_next = get_x_next(x[:, k], u[:, k], h)
        opti.subject_to(x[:, k+1] == x_next)
    # cost += 1e3*(x[0, -1] - x_target[0, -1])**2 + (x[1, -1] - x_target[1, -1])**2
    opti.subject_to(x[0, -1] == x_target[0, -1])
    opti.subject_to(x[1, -1] == x_target[1, -1])

    opti.subject_to(opti.bounded(u1_min, u[0,:], u1_max))
    opti.subject_to(opti.bounded(u2_min, u[1,:], u2_max))

    opts = {"ipopt.print_level": 5, "print_time": 1, "ipopt.sb": "yes"}
    # opts = {
    #     "ipopt.print_level": 5,
    #     "ipopt.tol": 1e-10,  # Tighter tolerance for precision
    #     "ipopt.mu_strategy": "adaptive",  # Better barrier handling
    #     "ipopt.hessian_approximation": "limited-memory",  # For large problems
    #     "ipopt.bound_relax_factor": 1e-6,  # Slight bound relaxation
    #     "expand": True,
    #     "ipopt.max_iter": 5000
    # }

    if warm_start is not None:
        opti.set_initial(u, np.vstack((warm_start['u1_warm'],
                                       warm_start['u2_warm'])))
        opti.set_initial(x, np.vstack((warm_start['x1_warm'],
                                       warm_start['x2_warm'],
                                       warm_start['x3_warm'])))


    opti.minimize(cost)
    opti.solver("ipopt", opts)

    # check_conditioning_sqrt_cost(opti, warm_start, no_steps, n, m)
    # check_conditioning_abs_cost(opti, warm_start, no_steps, n, m)

    solution = opti.solve()
    x_opt = solution.value(x)
    u_opt = solution.value(u)

    return x_opt, u_opt

def get_initial_warm_start(x_init, x_target, T, h):
    no_steps = int(ca.floor(T/h))
    x1_warm = np.linspace(0, x_target[0][0], no_steps + 1)
    x2_warm = np.linspace(0, x_target[1][0], no_steps + 1)
    x3_warm = np.linspace(0, 0, no_steps + 1)

    u1_warm = 0.1*np.ones(no_steps)
    u2_warm = 0.1**np.ones(no_steps)

    return {
        'x1_warm' : x1_warm,
        'x2_warm' : x2_warm,
        'x3_warm' : x3_warm,
        'u1_warm' : u1_warm,
        'u2_warm' : u2_warm
    }

def check_conditioning_abs_cost(opti, warm_start, no_steps, n, m):
    G = opti.g
    F = opti.f
    X = opti.x

    print("Number of decision variables:", X.shape)
    print("Number of constraints:", G.shape)

    # Convert to callable CasADi function
    J = ca.jacobian(G, X)
    H = ca.hessian(F, X)[0]

    nlp_fun = ca.Function(
        "nlp_fun",
        [X],  # inputs: decision vector
        [F, G, J, H]
    )
    u0 = np.vstack((warm_start['u1_warm'],
                    warm_start['u2_warm']))
    x0 = np.vstack((warm_start['x1_warm'],
                    warm_start['x2_warm'],
                    warm_start['x3_warm']))
    z0 = np.ones(no_steps)
    X0 = np.vstack((u0.reshape(m*no_steps,1), x0.reshape(n*(no_steps+1),1), z0.reshape(1*(no_steps),1)))
    f0, g0, J0, H0 = nlp_fun(X0)

    print("Objective (f) magnitude:", float(abs(f0)))
    print("Constraint (g) range: [", float(np.min(g0)), ",", float(np.max(g0)), "]")
    print("Jacobian (∂g/∂x) coefficient range: [", np.min(J0), ",", np.max(J0), "]")
    print("Hessian (∂²f/∂x²) coefficient range: [", np.min(H0), ",", np.max(H0), "]")

def check_conditioning_sqrt_cost(opti, warm_start, no_steps, n, m):
    G = opti.g
    F = opti.f
    X = opti.x

    print("Number of decision variables:", X.shape)
    print("Number of constraints:", G.shape)

    # Convert to callable CasADi function
    J = ca.jacobian(G, X)
    H = ca.hessian(F, X)[0]

    nlp_fun = ca.Function(
        "nlp_fun",
        [X],  # inputs: decision vector
        [F, G, J, H]
    )
    u0 = np.vstack((warm_start['u1_warm'],
                    warm_start['u2_warm']))
    x0= np.vstack((warm_start['x1_warm'],
                    warm_start['x2_warm'],
                    warm_start['x3_warm']))
    X0 = np.vstack((u0.reshape(m*no_steps,1),x0.reshape(n*(no_steps+1),1)))
    f0, g0, J0, H0 = nlp_fun(X0)

    print("Objective (f) magnitude:", float(abs(f0)))
    print("Constraint (g) range: [", float(np.min(g0)), ",", float(np.max(g0)), "]")
    print("Jacobian (∂g/∂x) coefficient range: [", np.min(J0), ",", np.max(J0), "]")
    print("Hessian (∂²f/∂x²) coefficient range: [", np.min(H0), ",", np.max(H0), "]")

def get_arc_length(x):
    arc_length = 0
    for k in range(x.shape[1]):
        arc_length += np.linalg.norm(x[:, k])
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

def plot_solution_in_state_space(x_tot, obstacles, title):
    x1 = x_tot[0]
    x2 = x_tot[1]
    x3 = x_tot[2]
    fig = plt.figure(figsize=(8, 8))
    gs_position_space = fig.add_gridspec(1, 1)
    ax6 = fig.add_subplot(gs_position_space[0])
    ax6.set_title(title, fontsize=14)
    ax6.plot(x1, x2, 'k-')
    ax6.set_xlabel('x1')
    ax6.set_ylabel('x2')
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

    fig.tight_layout(rect=[0, 0, 1, 0.95])

def plot_control(u_tot):
    fig = plt.figure(figsize=(8, 8))
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
    ax2.step(time, np.append(u2[0], u2), 'g-', label='u2 (turning rate)')
    ax2.set_title("u_2 vs time", fontsize=14)
    ax2.set_xlabel('Time step')
    ax2.legend()
    ax2.grid(True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

def plot_control_and_state_space(u_tot, x_tot, obstacles):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(3, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2], sharex=ax2)

    x1 = x_tot[0]
    x2 = x_tot[1]
    ax1.plot(x1, x2, 'k-')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.plot([x1[0], x1[-1]], [x2[0], x2[-1]], 'k.')
    ax1.grid(True)
    ax1.set_aspect('equal')
    ax1.set_title('State space')

    theta = np.linspace(0, 2*np.pi, 50)
    for obstacle in obstacles:
        x = obstacle['centre'][0] + obstacle['radius'] * np.cos(theta)
        y = obstacle['centre'][1] + obstacle['radius'] * np.sin(theta)
        ax1.plot(x, y, 'k-')

    u1 = u_tot[0]
    u2 = u_tot[1]
    time = np.arange(u_tot.shape[1] + 1)
    ax2.step(time, np.append(u1[0], u1), 'g-', label='u1 (velocity)')
    ax2.set_title("u_1 vs time", fontsize=14)
    ax2.set_xlabel('Time step')
    ax2.legend()
    ax2.grid(True)
    ax3.step(time, np.append(u2[0], u2), 'g-', label='u2 (turning rate)')
    ax3.set_title("u_2 vs time", fontsize=14)
    ax3.set_xlabel('Time step')
    ax3.legend()
    ax3.grid(True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()