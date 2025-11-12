import numpy as np
from car_OCP import plot_solution_in_state_space, solve_OCP, get_initial_warm_start, get_arc_length, plot_solution_versus_time, plot_control, plot_control_and_state_space

if __name__ == "__main__":
    x_init = np.array([[2],[0],[np.pi/2]])
    x_target = np.array([[-2],[0]])

    T = 10 # Horizon length
    h = 0.1 # RK4 time step

    obstacles = [
        {'centre': (0,0), 'radius': 0.5}, # this one and... 
        {'centre': (-0.5,-0.5), 'radius': 0.2}, # this one for chattering, and         eps_cost=1e-5, eps_u=eps
        # {'centre': (1,0.5), 'radius': 0.3},
        # {'centre': (1,-0.2), 'radius': 0.2},
        # {'centre': (-1.5,-0.2), 'radius': 0.2},
        # {'centre': (1.5,0), 'radius': 0.2},
        # {'centre': (-0.75,0), 'radius': 0.2},
        # {'centre': (-1.25,0.2), 'radius': 0.2}
    ]

    constraints = {
        'u1_min' : -1,
        'u1_max' : 1,
        'u2_min' : -1,
        'u2_max' : 1,
    }

    # warm_start = get_initial_warm_start(x_init, x_target, T, h)
    warm_start=None
    for eps in [0]:
        eps_cost=1e-5
        eps_u=eps
        print('////////////////////////////')
        print(f'eps_cost: {eps_cost}. eps_u: {eps_u}')
        print('////////////////////////////')
        eps_cost=eps_cost
        eps_u=eps_u
        x_opt, u_opt = solve_OCP(x_init, x_target, obstacles, constraints, T, h, warm_start, eps_u=eps_u, eps_cost=eps_cost, sqrt_cost_flag=True)

        warm_start = {
            'x1_warm': x_opt[0,:],
            'x2_warm': x_opt[1,:],
            'x3_warm': x_opt[2,:],
            'u1_warm': u_opt[0,:],
            'u2_warm': u_opt[1,:]
        }
        # warm_start['x1_warm'] = x_opt[0,:]
        # warm_start['x2_warm'] = x_opt[1,:]
        # warm_start['x3_warm'] = x_opt[2,:]
        # warm_start['u1_warm'] = u_opt[0,:]
        # warm_start['u2_warm'] = u_opt[1,:]

        # plot_solution_in_state_space(x_opt, obstacles, title=fr'Solution with $\varepsilon$ = {eps_cost}')
        # plot_solution_versus_time(x_opt, u_opt)
        # 'stop'

    #     # arc_length = get_arc_length(x_opt)

    # plot_solution_in_state_space(x_opt, obstacles, title=fr'Solution with $\varepsilon$ = {eps_cost}')
    # plot_solution_versus_time(x_opt, u_opt)
    # plot_control(u_opt)
    print(f'///////// arc length: {get_arc_length(x_opt)} //////////')
    plot_control_and_state_space(u_opt, x_opt, obstacles)
    
    'stop'
