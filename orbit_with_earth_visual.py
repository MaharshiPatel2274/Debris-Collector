
from skyfield.api import load, EarthSatellite
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load timescale
ts = load.timescale()
start_time = ts.utc(2025, 3, 29, 0, 0, 0)
minutes_to_track = 90
time_range = ts.utc(2025, 3, 29, 0, range(0, minutes_to_track))

# Simulated satellite TLE from KSC
sim_sat_line1 = "1 99999U 25000A   25088.00000000  .00000001  00000-0  00000-0 0  9990"
sim_sat_line2 = "2 99999  28.5  80.0 0002000  90.0 270.0 15.00000000    01"
sim_sat = EarthSatellite(sim_sat_line1, sim_sat_line2, "SimulatedSat", ts)

# Debris TLEs
tle_debris = [
    ("STARLINK-11629",
     "1 63104U 25039A   25088.23148322  .00013830  00000+0  82156-4 0  9991",
     "2 63104  42.9961 163.6559 0001528 269.0965  90.9730 15.77869528  5231"),
    ("STARLINK-11624",
     "1 63105U 25039B   25088.22660769  .00008135  00000+0  49565-4 0  9994",
     "2 63105  42.9989 163.6164 0001577 267.7708  92.2982 15.77856211  5237")
]

debris_sats = [EarthSatellite(l1, l2, name, ts) for name, l1, l2 in tle_debris]

# Satellite position
sat_positions = sim_sat.at(time_range).position.km
x_s, y_s, z_s = sat_positions

# Setup 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth as a wireframe sphere
r_earth = 6371  # km
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
x_e = r_earth * np.cos(u) * np.sin(v)
y_e = r_earth * np.sin(u) * np.sin(v)
z_e = r_earth * np.cos(v)
ax.plot_surface(x_e, y_e, z_e, color='lightblue', alpha=0.5, edgecolor='gray')

# Plot satellite trajectory
ax.plot(x_s, y_s, z_s, label='Simulated Satellite', linewidth=2.5, color='cyan', zorder=10)

# Plot debris and calculate distances
for debris in debris_sats:
    pos = debris.at(time_range).position.km
    x_d, y_d, z_d = pos
    distances = np.sqrt((x_s - x_d)**2 + (y_s - y_d)**2 + (z_s - z_d)**2)
    if distances.size > 0:
        min_dist = np.min(distances)
        t_close = time_range[np.argmin(distances)].utc_iso()
        if min_dist < 10:
            print(f"[ALERT] Collision risk with {debris.name} at {t_close} - Distance: {min_dist:.2f} km")
        else:
            print(f"{debris.name}: Safe (closest approach {min_dist:.2f} km at {t_close})")
    ax.plot(x_d, y_d, z_d, label=debris.name, linestyle='--', alpha=0.6)

# Plot settings
ax.set_title("3D Orbital Simulation with Earth")
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.view_init(elev=25, azim=45)
ax.legend()
plt.tight_layout()
plt.show()
