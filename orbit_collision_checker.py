# Autonomous Debris Collector AI System
# Author: Maharshi Niraj Patel

import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sgp4.api import Satrec, jday
from pyorbital.orbital import Orbital
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting.static import StaticOrbitPlotter
from astropy import units as u
from astropy.time import Time

# -----------------------------
# Agent 1: Trajectory + Threat AI
# -----------------------------
def fetch_tle(group="debris", limit=50):
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
    lines = requests.get(url).text.strip().splitlines()
    return [(lines[i].strip(), lines[i+1].strip(), lines[i+2].strip()) 
            for i in range(0, len(lines)-2, 3)][:limit]

def estimate_size():
    return round(np.random.uniform(0.1, 5.0), 2)  # simulate size in meters

def compute_criticality(tle, launch_orbit):
    name, l1, l2 = tle
    jd, fr = jday(*datetime.utcnow().timetuple()[:6])
    sat = Satrec.twoline2rv(l1, l2)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None
    r_vec = np.array(r)
    dist = np.linalg.norm(r_vec - launch_orbit.r.to_value(u.km))
    size = estimate_size()
    proximity_score = max(0, 1 - dist / 1000)
    crit = 0.6 * proximity_score + 0.4 * (size / 5.0)
    return (name, r_vec, size, crit) if crit > 0 else None

def identify_critical_debris(tles, launch_orbit):
    threats = [compute_criticality(tle, launch_orbit) for tle in tles]
    return sorted(filter(None, threats), key=lambda x: -x[3])[:10]

# -----------------------------
# Agent 2: Navigation Planner
# -----------------------------
def get_iss_position():
    iss = Orbital("ISS (ZARYA)")
    return iss.get_position(datetime.utcnow(), normalize=True)[0]

def plan_navigation(iss_pos, debris_target):
    print(f"\n🛰️ Collector dispatched from ISS at {iss_pos}")
    for name, pos, size, crit in debris_target:
        print(f"\n➡️ Navigating to {name} | Size: {size}m | Score: {crit:.2f}")
        print("   ✓ Path optimized to avoid other debris")
        print("   ✓ Debris captured and returning to ISS")

# -----------------------------
# Agent 3: Execution Controller
# -----------------------------
def log_and_execute(debris_target):
    for name, _, size, _ in debris_target:
        print(f"🔧 Rock crusher activated for: {name} ({size}m)")

# -----------------------------
# Agent 4: Visualization & Communication
# -----------------------------
def visualize_orbits(launch_orbit, debris):
    fig, ax = plt.subplots(figsize=(8, 8))
    plotter = StaticOrbitPlotter(ax)
    plotter.plot(launch_orbit, label="Launch Path")
    for name, pos, size, _ in debris:
        try:
            debris_orbit = Orbit.from_vectors(Earth, pos * u.km, [0, 0, 0] * u.km / u.s)
            plotter.plot(debris_orbit, label=f"{name} ({size}m)")
        except: continue
    plt.title("🛰️ Critical Debris Near Earth Launch Corridor")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------
# Main Multi-Agent Simulation
# -----------------------------
def main():
    print("🔭 Initializing AI agents...")
    launch_orbit = Orbit.circular(Earth, 400 * u.km, inc=51.6 * u.deg, epoch=Time.now())
    tles = fetch_tle()
    critical_debris = identify_critical_debris(tles, launch_orbit)
    print("\n🚀 Launch Corridor Obstruction Report")
    for i, (n, _, s, c) in enumerate(critical_debris):
        print(f"{i+1}. {n} | Size: {s}m | Score: {c:.2f}")
    iss_pos = get_iss_position()
    plan_navigation(iss_pos, critical_debris)
    log_and_execute(critical_debris)
    visualize_orbits(launch_orbit, critical_debris)

if __name__ == "__main__":
    main()