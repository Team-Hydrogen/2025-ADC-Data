import numpy as np
import pandas as pd
from lamberthub import izzo2015
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

# Gravitational parameter for Earth (mu)
MU_EARTH = 398600.4418

def get_user_input():
    print("Enter the initial position vector (km) as x, y, z:")
    r1 = np.array(list(map(float, input().split(','))))
    print("Enter the final position vector (km) as x, y, z:")
    r2 = np.array(list(map(float, input().split(','))))
    print("Enter the time of flight (seconds):")
    tof = float(input())
    return r1, r2, tof

def two_body_equations(t, y, mu):
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r)
    a = -mu * r / r_norm**3
    return np.hstack((v, a))

def propagate_orbit(r0, v0, tof, mu):
    y0 = np.hstack((r0, v0))
    t_span = (0, tof)
    t_eval = np.arange(0, tof + 60, 60)  # Evaluate every 60 seconds (1 minute)
    sol = solve_ivp(two_body_equations, t_span, y0, args=(mu,), t_eval=t_eval, rtol=1e-9, atol=1e-9)
    return sol.t, sol.y.T

def main():
    r1, r2, tof = get_user_input()
    
    # Solve Lambert's problem to find initial and final velocities
    v1_lambert, v2_lambert = izzo2015(MU_EARTH, r1, r2, tof)
    
    # Propagate the orbit from the initial conditions
    times, states = propagate_orbit(r1, v1_lambert, tof, MU_EARTH)
    
    # Save the results to a CSV file
    df = pd.DataFrame(states, columns=['x (km)', 'y (km)', 'z (km)', 'vx (km/s)', 'vy (km/s)', 'vz (km/s)'])
    df.insert(0, 'Time (s)', times)
    df.to_csv('trajectory.csv', index=False)
    print("Trajectory data saved to 'trajectory.csv'.")
    #3d plot of trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')
    plt.show()


if __name__ == "__main__":
    main()
