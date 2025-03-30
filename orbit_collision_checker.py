import os
import math
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import requests

# --- Poliastro / Astropy Imports ---
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.iod import izzo
from astropy import units as u
from astropy.time import Time

# --- SGP4 Import ---
from sgp4.api import Satrec, jday

# =====================================================
# GLOBAL SETTINGS and API URLs (unchanged)
# =====================================================
DEBRIS_TLE_URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?CATNR=25544&FORMAT=TLE"
USE_TEXTURED_EARTH = True
EARTH_TEXTURE_FILE = "earth_texture.jpg"
MAX_OBJECTS = 500  # Limit number of debris objects for globe simulation

# =====================================================
# 1. AI Image-Processing for Debris Size Detection
# =====================================================
def ai_detect_debris_size(image_path):
    """
    Simulated AI image processing function.
    In production, load your pre-trained model to process an image and predict the debris size.
    Here we simulate by returning a random size between 0.5 and 2.0 meters.
    """
    print(f"[AI] Processing image '{image_path}' for debris size detection...")
    size = random.uniform(0.5, 2.0)
    return size

# =====================================================
# 2. API Calls for TLE Data (Debris & ISS)
# =====================================================
def fetch_tle_data(url):
    """
    Generic TLE fetcher; expects at least 3 lines: name, line1, line2.
    Used for ISS.
    """
    response = requests.get(url)
    lines = response.text.strip().splitlines()
    if len(lines) >= 3:
        return lines[0], lines[1], lines[2]
    return None, None, None

def fetch_debris_tle():
    """
    Fetches debris TLE data from Celestrak using DEBRIS_TLE_URL.
    Reads the file assuming every 3 lines form a TLE set: name, line1, line2.
    Returns a list of dictionaries.
    """
    print(f"[API] Fetching debris TLE data from: {DEBRIS_TLE_URL}")
    try:
        resp = requests.get(DEBRIS_TLE_URL, timeout=10)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        debris_list = []
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
                name = lines[i].strip()
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()
                debris_list.append({
                    "name": name,
                    "line1": line1,
                    "line2": line2
                })
        return debris_list[:MAX_OBJECTS]
    except Exception as e:
        print(f"[ERROR] Fetching debris TLE data: {e}")
        return []

# =====================================================
# 3. Simulated External Query for Debris Mass
# =====================================================
def fetch_debris_mass(debris):
    """
    Simulate retrieving debris mass (in kg) from an external database.
    Here we assign a random mass (e.g., between 0 and 20 kg).
    """
    mass = random.uniform(0, 20)
    debris["mass"] = mass
    return debris

# =====================================================
# 4. Assign Debris Size and Compute Criticality
# =====================================================
def assign_debris_size_and_criticality(debris_list):
    """
    For each debris object, simulate an image-based size determination via AI
    and query a simulated external database for mass.
    Then compute a criticality score:
         criticality = size (in m) + (mass in kg) / 1000.
    """
    for debris in debris_list:
        image_path = f"images/{debris['name'].replace(' ', '_')}.jpg"
        debris["size"] = ai_detect_debris_size(image_path)
        debris = fetch_debris_mass(debris)
        debris["criticality"] = debris["size"] + debris["mass"] / 1000.0
    return debris_list

# =====================================================
# 5. Convert TLE to an Orbit using sgp4 and poliastro
# =====================================================
def create_orbit_from_tle(debris, epoch):
    """
    Create a poliastro Orbit object from TLE lines using the sgp4 propagator.
    """
    try:
        sat = Satrec.twoline2rv(debris["line1"], debris["line2"])
        jd, fr = jday(epoch.datetime.year, epoch.datetime.month, epoch.datetime.day,
                      epoch.datetime.hour, epoch.datetime.minute,
                      epoch.datetime.second + epoch.datetime.microsecond * 1e-6)
        error, r, v = sat.sgp4(jd, fr)
        if error != 0:
            print(f"[WARN] sgp4 error for {debris['name']}: error code {error}")
            raise ValueError("Propagation error")
        r = np.array(r) * u.km
        v = np.array(v) * (u.km / u.s)
        orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch)
        return orbit
    except Exception as e:
        print(f"[ERROR] Creating orbit from TLE for {debris['name']}: {e}")
        return None

# =====================================================
# 6. Compute Transfer Orbit using Lambert's Problem
# =====================================================
def compute_transfer_orbit(iss_orbit, debris_orbit, departure_time, arrival_time):
    """
    Compute a transfer orbit from the ISS orbit to the debris orbit using Lambert's solution.
    Returns departure velocity, arrival velocity, and corresponding position vectors.
    """
    # Propagate to get position vectors at departure and arrival (using simplified propagation)
    r1 = iss_orbit.propagate(departure_time - iss_orbit.epoch).r
    r2 = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r
    (v1, v2), = izzo.lambert(Earth.k, r1, r2, (arrival_time - departure_time).to(u.s))
    return v1, v2, r1, r2

# =====================================================
# 7. Simulate the Debris Collector Mission (from ISS)
# =====================================================
def simulate_mission(debris_list):
    """
    Simulate a mission where the debris collector departs from a simulated ISS orbit,
    travels to the most critical debris object, computes a transfer orbit via Lambert's problem,
    estimates delta-v requirements and mission duration, and visualizes trajectories.
    """
    epoch = Time(datetime.utcnow())
    # Define a realistic ISS orbit (approx. 400 km altitude, 51.6° inclination)
    iss_orbit = Orbit.from_classical(
        attractor=Earth,
        a=(Earth.R + 400*u.km),
        ecc=0.001 * u.one,
        inc=51.6*u.deg,
        raan=0*u.deg,
        argp=0*u.deg,
        nu=0*u.deg,
        epoch=epoch
    )
    # For each debris object, try to create an orbit from its TLE. If that fails, simulate one.
    for debris in debris_list:
        if debris.get("line1") and debris.get("line2"):
            orbit = create_orbit_from_tle(debris, epoch)
            if orbit is None:
                altitude = random.uniform(350, 450) * u.km
                inclination = random.uniform(45, 55) * u.deg
                orbit = Orbit.from_classical(
                    attractor=Earth,
                    a=(Earth.R + altitude),
                    ecc=0.001 * u.one,
                    inc=inclination,
                    raan=random.uniform(0, 360)*u.deg,
                    argp=random.uniform(0, 360)*u.deg,
                    nu=random.uniform(0, 360)*u.deg,
                    epoch=epoch
                )
            debris["orbit"] = orbit
        else:
            altitude = random.uniform(350, 450) * u.km
            inclination = random.uniform(45, 55) * u.deg
            debris["orbit"] = Orbit.from_classical(
                attractor=Earth,
                a=(Earth.R + altitude),
                ecc=0.001 * u.one,
                inc=inclination,
                raan=random.uniform(0, 360)*u.deg,
                argp=random.uniform(0, 360)*u.deg,
                nu=random.uniform(0, 360)*u.deg,
                epoch=epoch
            )
    # Sort debris by descending criticality (largest first)
    debris_list.sort(key=lambda d: d["criticality"], reverse=True)
    target_debris = debris_list[0]
    print(f"\n[MISSION] Selected target debris: {target_debris['name']} (Size: {target_debris['size']:.2f} m, Mass: {target_debris['mass']:.2f} kg)")
    
    # Set departure (10 min after epoch) and arrival (transfer duration 40 min) times
    departure_time = epoch + 10*u.min
    arrival_time = departure_time + 40*u.min
    
    # Compute transfer orbit using Lambert's problem
    v_dep, v_arr, r_dep, r_arr = compute_transfer_orbit(iss_orbit, target_debris["orbit"], departure_time, arrival_time)
    transfer_time = (arrival_time - departure_time).to(u.s).value
    print(f"[MISSION] Transfer orbit computed; transfer duration = {transfer_time:.2f} seconds")
    
    # Estimate delta-v (approximated)
    iss_v = iss_orbit.v.to(u.km/u.s).value
    dep_delta_v = np.linalg.norm(v_dep.to(u.km/u.s).value - iss_v)
    debris_v = target_debris["orbit"].v.to(u.km/u.s).value
    arr_delta_v = np.linalg.norm(debris_v - v_arr.to(u.km/u.s).value)
    total_delta_v = dep_delta_v + arr_delta_v
    print(f"[MISSION] Delta-v requirements: Departure = {dep_delta_v:.2f} km/s, Arrival = {arr_delta_v:.2f} km/s, Total = {total_delta_v:.2f} km/s")
    
    print(f"[MISSION] Collecting debris {target_debris['name']} ...")
    print("[MISSION] Initiating rock crusher on ISS... Debris crushed to dust.\n")
    
    # Visualize the trajectories in 3D (ISS, debris, and transfer path)
    visualize_trajectories(iss_orbit, target_debris["orbit"], departure_time, arrival_time)
    
    return total_delta_v, transfer_time

# =====================================================
# 8. Visualize Trajectories in 3D
# =====================================================
def visualize_trajectories(iss_orbit, debris_orbit, departure_time, arrival_time):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Earth as a sphere (with optional texture if available)
    R_earth = Earth.R.to(u.km).value
    u_vals = np.linspace(0, 2*np.pi, 100)
    v_vals = np.linspace(0, np.pi, 100)
    x = R_earth * np.outer(np.cos(u_vals), np.sin(v_vals))
    y = R_earth * np.outer(np.sin(u_vals), np.sin(v_vals))
    z = R_earth * np.outer(np.ones_like(u_vals), np.cos(v_vals))
    if USE_TEXTURED_EARTH and os.path.isfile(EARTH_TEXTURE_FILE):
        img = plt.imread(EARTH_TEXTURE_FILE)
        facecolors = np.empty(x.shape + (4,), dtype=np.float32)
        img_h, img_w, _ = img.shape
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                s = u_vals[i] / (2*np.pi)
                t = (v_vals[j]) / np.pi
                x_img = int(s * (img_w - 1))
                y_img = int((1-t) * (img_h - 1))
                facecolors[i,j,:] = img[y_img, x_img, :4]
        ax.plot_surface(x, y, z, rcount=100, ccount=100, facecolors=facecolors, shade=False)
    else:
        ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    
    # Plot ISS orbit (sampled at 200 points)
    iss_points = iss_orbit.sample(200)
    ax.plot(iss_points.x.to(u.km).value,
            iss_points.y.to(u.km).value,
            iss_points.z.to(u.km).value,
            label="ISS Orbit", color="green")
    
    # Plot debris orbit (sampled at 200 points)
    debris_points = debris_orbit.sample(200)
    ax.plot(debris_points.x.to(u.km).value,
            debris_points.y.to(u.km).value,
            debris_points.z.to(u.km).value,
            label="Debris Orbit", color="red")
    
    # Compute a transfer trajectory by linear interpolation between departure and arrival positions
    t_vals = np.linspace(0, (arrival_time - departure_time).to(u.s).value, 100)
    transfer_points = []
    r_dep = iss_orbit.propagate(departure_time - iss_orbit.epoch).r.value
    r_arr = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r.value
    for t in t_vals:
        r = r_dep + (r_arr - r_dep) * (t / t_vals[-1])
        transfer_points.append(r)
    transfer_points = np.array(transfer_points)
    ax.plot(transfer_points[:,0], transfer_points[:,1], transfer_points[:,2],
            label="Transfer Trajectory", color="orange", linestyle="--")
    
    ax.set_title("Debris Collector Mission Trajectories")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    max_range = 1.5 * R_earth
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.show()

# =====================================================
# 9. Simulate Satellite Launch from Earth
# =====================================================
def simulate_satellite_launch():
    """
    Simulate a satellite launch from a specified Earth coordinate (e.g., Cape Canaveral).
    Here we create a simple LEO orbit (e.g., 500 km altitude, 28.5° inclination).
    """
    epoch = Time(datetime.utcnow())
    sat_orbit = Orbit.from_classical(
        attractor=Earth,
        a=(Earth.R + 500*u.km),
        ecc=0.001 * u.one,
        inc=28.5*u.deg,
        raan=0*u.deg,
        argp=0*u.deg,
        nu=0*u.deg,
        epoch=epoch
    )
    print(f"[SATELLITE] Simulated satellite launched from Earth (500 km altitude, 28.5° inc).")
    return sat_orbit

def visualize_satellite_and_mission(iss_orbit, debris_orbit, sat_orbit, departure_time, arrival_time):
    """
    Visualize in 3D the ISS orbit, debris orbit, satellite orbit, and the transfer trajectory.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    R_earth = Earth.R.to(u.km).value
    u_vals = np.linspace(0, 2*np.pi, 100)
    v_vals = np.linspace(0, np.pi, 100)
    x = R_earth * np.outer(np.cos(u_vals), np.sin(v_vals))
    y = R_earth * np.outer(np.sin(u_vals), np.sin(v_vals))
    z = R_earth * np.outer(np.ones_like(u_vals), np.cos(v_vals))
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    
    # Plot ISS orbit (green)
    iss_points = iss_orbit.sample(200)
    ax.plot(iss_points.x.to(u.km).value,
            iss_points.y.to(u.km).value,
            iss_points.z.to(u.km).value,
            label="ISS Orbit", color="green")
    
    # Plot debris orbit (red)
    debris_points = debris_orbit.sample(200)
    ax.plot(debris_points.x.to(u.km).value,
            debris_points.y.to(u.km).value,
            debris_points.z.to(u.km).value,
            label="Debris Orbit", color="red")
    
    # Plot satellite orbit (magenta)
    sat_points = sat_orbit.sample(200)
    ax.plot(sat_points.x.to(u.km).value,
            sat_points.y.to(u.km).value,
            sat_points.z.to(u.km).value,
            label="Satellite Orbit", color="magenta")
    
    # Plot transfer trajectory (orange dashed) from ISS to debris
    t_vals = np.linspace(0, (arrival_time - departure_time).to(u.s).value, 100)
    transfer_points = []
    r_dep = iss_orbit.propagate(departure_time - iss_orbit.epoch).r.value
    r_arr = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r.value
    for t in t_vals:
        r = r_dep + (r_arr - r_dep) * (t / t_vals[-1])
        transfer_points.append(r)
    transfer_points = np.array(transfer_points)
    ax.plot(transfer_points[:,0], transfer_points[:,1], transfer_points[:,2],
            label="Transfer Trajectory", color="orange", linestyle="--")
    
    ax.set_title("Combined Mission: Satellite Launch & Debris Collector Trajectories")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    max_range = 1.5 * R_earth
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.show()

# =====================================================
# 10. Plot Global Debris Map (Globe Simulation)
# =====================================================
def propagate_tles(tle_list):
    """
    For each TLE set (name, line1, line2) in tle_list, propagate it to current time
    using sgp4 and return a list of (name, x, y, z) positions (in km).
    """
    now = datetime.utcnow()
    jd_now, fr_now = jday(now.year, now.month, now.day,
                          now.hour, now.minute, now.second + now.microsecond*1e-6)
    positions = []
    for (name, line1, line2) in tle_list:
        try:
            sat = Satrec.twoline2rv(line1, line2)
            e, r, v = sat.sgp4(jd_now, fr_now)
            if e == 0:
                positions.append((name, r[0], r[1], r[2]))
        except Exception as ex:
            continue
    return positions

def plot_debris(positions):
    """
    Plot Earth as a sphere and scatter the debris positions on it.
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    R_earth = 6378.0
    u_vals = np.linspace(0, 2*np.pi, 60)
    v_vals = np.linspace(-np.pi/2, np.pi/2, 30)
    x = R_earth * np.outer(np.cos(u_vals), np.cos(v_vals))
    y = R_earth * np.outer(np.sin(u_vals), np.cos(v_vals))
    z = R_earth * np.outer(np.ones_like(u_vals), np.sin(v_vals))
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    zs = [p[3] for p in positions]
    ax.scatter(xs, ys, zs, s=2, color='red', alpha=0.6, label='Debris')
    
    ax.set_title("Global Debris Map from Celestrak Data")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    max_range = 1.2 * max(max(np.abs(xs)), max(np.abs(ys)), max(np.abs(zs)), R_earth)
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.legend()
    plt.show()

def plot_debris_globe():
    """
    Fetch debris TLE data from DEBRIS_TLE_URL, propagate to current time,
    and plot a global 3D debris map.
    """
    tle_data = fetch_debris_tle()
    if not tle_data:
        print("[ERROR] No debris TLE data found. Cannot plot globe simulation.")
        return
    positions = propagate_tles([(d["name"], d["line1"], d["line2"]) for d in tle_data])
    print(f"[INFO] Propagated {len(positions)} debris objects to current time.")
    plot_debris(positions)

# =====================================================
# 11. Main Function – Integrating Both Systems
# =====================================================
def main():
    print("=== Autonomous Debris Removal and Satellite Launch Simulation ===\n")
    
    # ----------------------------
    # Part A: Debris Collector Mission (ISS -> Target Debris)
    # ----------------------------
    # Fetch ISS TLE (for reference) and debris TLE data
    iss_name, iss_line1, iss_line2 = fetch_tle_data("https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE")
    print(f"[API] ISS TLE fetched: {iss_name}")
    
    debris_list = fetch_debris_tle()
    if not debris_list:
        print("[WARN] No debris TLE data fetched. Using simulated debris objects.")
        debris_list = []
        for i in range(5):
            debris_list.append({
                "name": f"Simulated_Debris_{i+1}",
                "line1": "",
                "line2": ""
            })
    debris_list = assign_debris_size_and_criticality(debris_list)
    
    print("\n[INFO] Detected debris objects (Size in m, Mass in kg, Criticality):")
    for d in debris_list:
        print(f"  {d['name']}: Size = {d['size']:.2f} m, Mass = {d['mass']:.2f} kg, Criticality = {d['criticality']:.2f}")
    
    total_delta_v, mission_duration = simulate_mission(debris_list)
    print(f"[RESULT] Debris collector mission complete. Total delta-v = {total_delta_v:.2f} km/s, Mission duration = {mission_duration:.2f} s\n")
    
    # ----------------------------
    # Part B: Simulate Satellite Launch from Earth
    # ----------------------------
    sat_orbit = simulate_satellite_launch()
    
    # For visualization, re-create ISS and target debris orbits (using same epoch as in simulate_mission)
    epoch = Time(datetime.utcnow())
    iss_orbit = Orbit.from_classical(
        attractor=Earth,
        a=(Earth.R + 400*u.km),
        ecc=0.001 * u.one,
        inc=51.6*u.deg,
        raan=0*u.deg,
        argp=0*u.deg,
        nu=0*u.deg,
        epoch=epoch
    )
    # Use the target debris from mission simulation (first in sorted list)
    target_debris = debris_list[0]
    if "orbit" not in target_debris:
        target_debris["orbit"] = create_orbit_from_tle(target_debris, epoch)
    debris_orbit = target_debris["orbit"]
    departure_time = epoch + 10*u.min
    arrival_time = departure_time + 40*u.min
    
    # Visualize combined orbits: ISS, debris, satellite and the transfer (ISS -> debris)
    visualize_satellite_and_mission(iss_orbit, debris_orbit, sat_orbit, departure_time, arrival_time)
    
    # ----------------------------
    # Part C: Global Debris Map Visualization
    # ----------------------------
    plot_debris_globe()

if __name__ == '__main__':
    main()
