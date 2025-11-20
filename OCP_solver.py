import numpy as np
from car_OCP import solve_OCP, get_initial_warm_start, get_arc_length, plot_control_and_state_space

if __name__ == "__main__":
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

    warm_start = get_initial_warm_start(x_init, x_target, T, h)
    for eps in [1e-6]:
        ** take smaller steps... **
        eps_cost=eps
        eps_u=0
        print('////////////////////////////')
        print(f'eps_cost: {eps_cost}. eps_u: {eps_u}')
        print('////////////////////////////')
        eps_cost=eps_cost
        eps_u=eps_u
        x_opt, u_opt = solve_OCP(x_init, x_target, obstacles, constraints, T, h, warm_start, eps_u=eps_u, eps_cost=eps_cost)

        # print(f'///////// arc length: {get_arc_length(x_opt)} //////////')
        # plot_control_and_state_space(u_opt, x_opt, obstacles)

        warm_start = {
            'x1_warm': x_opt[0,:],
            'x2_warm': x_opt[1,:],
            'x3_warm': x_opt[2,:],
            'u1_warm': u_opt[0,:],
            'u2_warm': u_opt[1,:]
        }

    print(f'///////// arc length: {get_arc_length(x_opt)} //////////')
    plot_control_and_state_space(u_opt, x_opt, obstacles)
    
    'stop'
