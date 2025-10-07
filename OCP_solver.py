from casadi import *
import numpy as np
from car_OCP import plot_solution, get_x_next, solve_OCP

# Call it numerical experiments...

if __name__ == "__main__":
    x_init = np.array([[2],[0],[pi/2]]) # 2 x 1 vector
    # x_target = np.array([[1],[1],[0]])

    T = 10
    h = 0.01 # RK4 time step
    number_of_iterations = 50
    x_target = np.array([[-2],[0],[0]])

    u_cl = np.zeros((2, number_of_iterations))
    x_cl = np.zeros((3, number_of_iterations + 1))
    x_cl[:, 0] = x_init[:, 0]

    obstacles = [
        {'centre': (0,0), 'radius': 0.3}, # centre, radius pairs
        {'centre': (-0.5,-0.5), 'radius': 0.2},
        {'centre': (0.5,0.5), 'radius': 0.3}
    ]

    x_hat = x_init
    for i in range(number_of_iterations):
        x_opt, u_opt = solve_OCP(x_hat, x_target, obstacles, T, h)
        print('////////')
        print(f'x_final: {x_opt[:, -1]}, u_final: {u_opt[:, -1]}')
        plot_solution(x_opt, u_opt.reshape(2, int(floor(T/h))), obstacles)
        u_opt_first_element = u_opt[:, 0]

        # save closed loop x and u
        u_cl[:, i] = u_opt_first_element
        x_next = get_x_next(x_hat, u_opt_first_element, h)
        x_cl[:, i+1] = np.squeeze(x_next)

        # update initial state
        x_hat = x_next

    plot_solution(x_cl, u_cl)
    print(f'x_final: {x_cl[:, -1]}, u_final: {u_cl[:, -1]}')