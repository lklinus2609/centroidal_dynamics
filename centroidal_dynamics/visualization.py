# centroidal_dynamics/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_robot_state(time_array, com_array, momentum_array, support_polygon=None, cop=None):
    """
    Plot the robot state over time.
    
    Parameters:
    -----------
    time_array: array-like
        Time points
    com_array: array-like
        CoM positions over time
    momentum_array: array-like
        Momentum values over time (angular and linear)
    support_polygon: array-like, optional
        Vertices of the support polygon
    cop: array-like, optional
        Center of Pressure position
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot CoM trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(com_array[:, 0], com_array[:, 2], com_array[:, 1])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')
    ax1.set_title('CoM Trajectory')
    
    # Plot support polygon and CoP if provided
    if support_polygon is not None:
        ax1.plot(support_polygon[:, 0], support_polygon[:, 2], support_polygon[:, 1], 'r-')
    if cop is not None:
        ax1.plot([cop[0]], [cop[2]], [cop[1]], 'ro', markersize=10)
    
    # Plot CoM position over time
    ax2 = fig.add_subplot(222)
    ax2.plot(time_array, com_array[:, 0], 'r-', label='X')
    ax2.plot(time_array, com_array[:, 1], 'g-', label='Y')
    ax2.plot(time_array, com_array[:, 2], 'b-', label='Z')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('CoM Position')
    ax2.legend()
    ax2.grid(True)
    
    # Plot angular momentum
    ax3 = fig.add_subplot(223)
    ax3.plot(time_array, momentum_array[:, 0], 'r-', label='X')
    ax3.plot(time_array, momentum_array[:, 1], 'g-', label='Y')
    ax3.plot(time_array, momentum_array[:, 2], 'b-', label='Z')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angular Momentum (kg⋅m²/s)')
    ax3.set_title('Angular Momentum')
    ax3.legend()
    ax3.grid(True)
    
    # Plot linear momentum
    ax4 = fig.add_subplot(224)
    ax4.plot(time_array, momentum_array[:, 3], 'r-', label='X')
    ax4.plot(time_array, momentum_array[:, 4], 'g-', label='Y')
    ax4.plot(time_array, momentum_array[:, 5], 'b-', label='Z')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Linear Momentum (kg⋅m/s)')
    ax4.set_title('Linear Momentum')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()