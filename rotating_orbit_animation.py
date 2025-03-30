
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from skyfield.api import load, EarthSatellite

# Load timescale
ts = load.timescale()
time_range = ts.utc(2025, 3, 29, 0, range(0, 90))

# Simulated satellite TLE
sim_sat = EarthSatellite(
    "1 99999U 25000A   25088.00000000  .00000001  00000-0  00000-0 0  9990",
    "2 99999  28.5  80.0 0002000  90.0 270.0 15.00000000    01",
    "SimulatedSat", ts)

# Debris TLEs
tle_debris = [
    ("STARLINK-11629", "1 63104U 25039A   25088.23148322  .00013830  00000+0  82156-4 0  9991",
     "2 63104  42.9961 163.6559 0001528 269.0965  90.9730 15.77869528  5231"),
    ("STARLINK-11624", "1 63105U 25039B   25088.22660769  .00008135  00000+0  49565-4 0  9994",
     "2 63105  42.9989 163.6164 0001577 267.7708  92.2982 15.77856211  5237")
]
debris_sats = [EarthSatellite(l1, l2, name, ts) for name, l1, l2 in tle_debris]

# Satellite position
x_s, y_s, z_s = sim_sat.at(time_range).position.km

# Debris positions
debris_positions = [sat.at(time_range).position.km for sat in debris_sats]

# Setup plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth sphere
r_earth = 6371
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
x_e = r_earth * np.cos(u) * np.sin(v)
y_e = r_earth * np.sin(u) * np.sin(v)
z_e = r_earth * np.cos(v)
ax.plot_surface(x_e, y_e, z_e, color='lightblue', alpha=0.5, edgecolor='gray')

# Satellite and debris lines
sat_line, = ax.plot([], [], [], 'cyan', label="Simulated Satellite", linewidth=2.5)
debris_lines = [ax.plot([], [], [], '--', label=sat.name, alpha=0.6)[0] for sat in debris_sats]

# Limits and labels
max_range = 20000
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])
ax.set_title("Rotating 3D Orbit with Earth and Debris")
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.legend()

# Animation function
def update(frame):
    sat_line.set_data(x_s[:frame], y_s[:frame])
    sat_line.set_3d_properties(z_s[:frame])
    for i, (x_d, y_d, z_d) in enumerate(debris_positions):
        debris_lines[i].set_data(x_d[:frame], y_d[:frame])
        debris_lines[i].set_3d_properties(z_d[:frame])
    ax.view_init(elev=25, azim=frame * 2 % 360)
    return [sat_line] + debris_lines

ani = FuncAnimation(fig, update, frames=len(time_range), interval=100, blit=True)
plt.tight_layout()
plt.show()
