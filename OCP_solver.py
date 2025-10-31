import numpy as np
from car_OCP import plot_solution, solve_OCP

if __name__ == "__main__":
    x_init = np.array([[2],[0],[np.pi/2]])

    T = 8 # Horizon length
    h = 0.1 # RK4 time step
    x_target = np.array([[-2],[0]])

    obstacles = [
        {'centre': (0,0), 'radius': 0.5},
        {'centre': (-0.5,-0.5), 'radius': 0.2},
        {'centre': (1,0.5), 'radius': 0.3},
        {'centre': (1,-0.2), 'radius': 0.2},
        {'centre': (-1.5,-0.2), 'radius': 0.2},
        {'centre': (1.5,0), 'radius': 0.2},
        {'centre': (-0.75,0), 'radius': 0.2}
    ]

    constraints = {
        'u1_min' : -1,
        'u1_max' : 1,
        'u2_min' : -1,
        'u2_max' : 1,
    }

    x_opt, u_opt = solve_OCP(x_init, x_target, obstacles, constraints, T, h)
    print('////////')
    print(f'x_final: {x_opt[:, -1]}, u_final: {u_opt[:, -1]}')
    plot_solution(x_opt, u_opt.reshape(2, int(np.floor(T/h))), obstacles)
    'stop'