import numpy as np
from car_OCP import solve_OCP, get_arc_length, plot_solution_in_statespace_and_control

if __name__ == "__main__":
    eps=1e-2
    delta=1e-2

    x_init = np.array([[2],[0],[np.pi/2]])
    x_target = np.array([[-2],[0]])

    T = 10 # Horizon length
    h = 0.1 # RK4 time step

    obstacles = [
        {'centre': (0,0), 'radius': 0.5},
        {'centre': (-0.5,-0.5), 'radius': 0.2},
        {'centre': (1,0.5), 'radius': 0.3},
        {'centre': (1,-0.35), 'radius': 0.3},
        {'centre': (-1.5,-0.2), 'radius': 0.2},
        {'centre': (1.5,-0.2), 'radius': 0.2},
        {'centre': (-0.75,0), 'radius': 0.2},
        {'centre': (-1.25,0.2), 'radius': 0.2}
    ]

    constraints = {
        'u1_min' : -1,
        'u1_max' : 1,
        'u2_min' : -1,
        'u2_max' : 1,
    }

    x_opt, u_opt = solve_OCP(x_init, x_target, obstacles, constraints, T, h, warm_start=None, eps=eps, delta=delta)

    plot_solution_in_statespace_and_control(x_opt, u_opt, obstacles, arc_length=np.round(get_arc_length(x_opt), 2), obstacles_only=False, eps=eps, delta=delta)

