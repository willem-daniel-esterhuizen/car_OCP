from casadi import *
import numpy as np
from car_OCP import plot_solution, get_x_next
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == "__main__":
    x_init = np.array([[1],[0.1],[pi/2]])
    h = 0.1

    # drive straight
    u = np.vstack([np.ones(5), 
                   np.zeros(5)])
    # rotate CCW
    u = np.append(u, np.vstack([np.zeros(5), 
                                np.ones(5)]), axis=1)
    # drive backwards
    u = np.append(u, np.vstack([-np.ones(5), 
                                np.zeros(5)]), axis=1)
    # rotate CW
    u = np.append(u, np.vstack([np.zeros(10), 
                                -np.ones(10)]), axis=1)
    # wiggle wiggle
    u = np.append(u, np.vstack([0.5*np.ones(5), 
                                -np.ones(5)]), axis=1)
    # wiggle wiggle
    u = np.append(u, np.vstack([0.5*np.ones(5), 
                                np.ones(5)]), axis=1)
    # wiggle wiggle
    u = np.append(u, np.vstack([0.5*np.ones(5), 
                                -np.ones(5)]), axis=1)
    # tuuuuuurn CW
    u = np.append(u, np.vstack([0.2*np.ones(50), 
                                -2*np.ones(50)]), axis=1)

    x = x_init
    x_tot = x_init
    for k in range(u.shape[1]):
        x = get_x_next(x[:, 0], u[:, k], h)
        x_tot = np.append(x_tot, x, axis=1)

    # Animation setup
    fig, ax = plt.subplots()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectory Animation')
    ax.grid(True)

    line, = ax.plot(x_tot[0, :], x_tot[1, :], 'b-', alpha=0.3, label='Trajectory')

    point, = ax.plot(x_tot[0, 0], x_tot[1, 0], 'ro', label='Current Position')
    scaling = 0.1
    end_point_x = x_tot[0, 0] + scaling*cos(x_tot[2, 0])
    end_point_y = x_tot[1, 0] + scaling*sin(x_tot[2, 0])
    direction_arrow, = ax.plot([x_tot[0, 0], end_point_x] ,[x_tot[1, 0], end_point_y], 'g-', label='Direction Arrow')
    ax.legend()

    # Update function for animation
    def update(frame):
        point.set_data([x_tot[0, frame]], [x_tot[1, frame]])

        direction_arrow_x_start = x_tot[0, frame]
        direction_arrow_x_end = x_tot[0, frame] + scaling*cos(x_tot[2, frame])
        direction_arrow_y_start = x_tot[1, frame]
        direction_arrow_y_end = x_tot[1, frame] + scaling*sin(x_tot[2, frame])

        direction_arrow.set_data([direction_arrow_x_start, direction_arrow_x_end], [direction_arrow_y_start, direction_arrow_y_end])
        return point,

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        func=update,              # Function to update the plot
        frames=x_tot.shape[1],    # Number of frames (columns in x_tot)
        interval=50,              # Delay between frames in ms
        blit=True                 # Use blitting for performance
    )

    plt.show()
