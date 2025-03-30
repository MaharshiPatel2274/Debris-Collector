import random
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import requests
import json
import os

# Import poliastro and astropy for orbital mechanics
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.iod import izzo
from astropy import units as u
from astropy.time import Time

# Import sgp4 for TLE propagation
from sgp4.api import Satrec, jday

# =====================================================
# GLOBAL SETTINGS
# =====================================================
# Use a debris TLE file from Celestrak.
# For example, Fengyun-1C debris TLE:
DEBRIS_TLE_URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?CATNR=25544&FORMAT=TLE"

# Optional: Set to True to use an Earth texture image for the globe.
USE_TEXTURED_EARTH = True
EARTH_TEXTURE_FILE = "earth_texture.jpg"

# Limit number of TLE objects (if needed)
MAX_OBJECTS = 500

# =====================================================
# 1. AI Image-Processing for Debris Size Detection
# =====================================================
def ai_detect_debris_size(image_path):
    """
    Simulated AI image processing function.
    In production, load your pre-trained model (e.g., TensorFlow/Keras)
    to process an image and predict the debris size.
    Here we simulate by returning a random size between 0.5 and 5.0 meters.
    """
    print(f"Processing image '{image_path}' for debris size detection...")
    size = random.uniform(0.5, 2.0)
    return size

# =====================================================
# 2. Fetch TLE Data from Celestrak for Debris
# =====================================================
def fetch_debris_tle():
    """
    Attempt to fetch debris TLE data from Celestrak using the DEBRIS_TLE_URL.
    Reads the file assuming every 3 lines form a TLE set: name, line1, line2.
    Returns a list of dictionaries with keys: name, line1, line2.
    """
    print(f"Fetching TLE data from: {DEBRIS_TLE_URL}")
    try:
        resp = requests.get(DEBRIS_TLE_URL, timeout=10)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        debris_list = []
        for i in range(0, len(lines), 3):
            if i+2 < len(lines):
                name = lines[i].strip()
                line1 = lines[i+1].strip()
                line2 = lines[i+2].strip()
                debris_list.append({
                    "name": name,
                    "line1": line1,
                    "line2": line2
                })
        # Optionally limit the number of objects
        return debris_list[:MAX_OBJECTS]
    except Exception as e:
        print(f"Error fetching TLE data: {e}")
        return []

# =====================================================
# 3. Simulated External Query for Debris Mass
# =====================================================
def fetch_debris_mass(debris):
    """
    Simulate retrieving debris mass (in kg) from an external database (e.g., ESA DISCOS).
    Here we assign a random mass between 50 and 500 kg.
    """
    mass = random.uniform(0, 20)
    debris["mass"] = mass
    return debris

# =====================================================
# 4. Assign Debris Size and Compute Criticality
# =====================================================
def assign_debris_size_and_criticality(debris_list):
    """
    For each debris object, simulate an image-based size determination via AI,
    and simulate a mass query. Then compute a criticality score as:
         criticality = size (meters) + (mass in kg)/1000.
    """
    for debris in debris_list:
        image_path = f"images/{debris['name'].replace(' ', '_')}.jpg"
        debris["size"] = ai_detect_debris_size(image_path)
        debris = fetch_debris_mass(debris)
        debris["criticality"] = debris["size"] + debris["mass"] / 1000.0
    return debris_list

# =====================================================
# 5. Compute a Transfer Orbit using Lambert's Problem
# =====================================================
def compute_transfer_orbit(iss_orbit, debris_orbit, departure_time, arrival_time):
    """
    Compute a transfer orbit from the ISS orbit to the debris orbit using Lambert's solution.
    Returns departure velocity, arrival velocity, and the corresponding position vectors.
    """
    r1 = iss_orbit.propagate(departure_time - iss_orbit.epoch).r
    r2 = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r
    (v1, v2), = izzo.lambert(Earth.k, r1, r2, (arrival_time - departure_time).to(u.s))
    return v1, v2, r1, r2

# =====================================================
# 6. Convert TLE to an Orbit using sgp4 and poliastro
# =====================================================
def create_orbit_from_tle(debris, epoch):
    """
    Create a poliastro Orbit object from TLE lines using the sgp4 propagator.
    Propagates the TLE at the current epoch.
    """
    try:
        sat = Satrec.twoline2rv(debris["line1"], debris["line2"])
        jd, fr = jday(epoch.datetime.year, epoch.datetime.month, epoch.datetime.day,
                      epoch.datetime.hour, epoch.datetime.minute,
                      epoch.datetime.second + epoch.datetime.microsecond * 1e-6)
        error, r, v = sat.sgp4(jd, fr)
        if error != 0:
            print(f"sgp4 error for {debris['name']}: error code {error}")
            raise ValueError("Propagation error")
        r = np.array(r) * u.km
        v = np.array(v) * (u.km / u.s)
        orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch)
        return orbit
    except Exception as e:
        print(f"Error creating orbit from TLE for {debris['name']}: {e}")
        return None

# =====================================================
# 7. Simulate the Debris Collector Mission
# =====================================================
def simulate_mission(debris_list):
    """
    Simulate a mission where a debris collector departs from a simulated ISS orbit,
    travels to the most critical debris object, and returns.
    The function computes the transfer orbit via Lambert's problem and visualizes the trajectories.
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
    # For each debris object, if TLE data exists, try to create an orbit from it.
    # Otherwise, assign a simulated orbit.
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
    # Sort debris objects by descending criticality (largest score first)
    debris_list.sort(key=lambda d: d["criticality"], reverse=True)
    target_debris = debris_list[0]
    print(f"\nSelected target debris: {target_debris['name']} with size {target_debris['size']:.2f} m and mass {target_debris['mass']:.2f} kg")
    
    # Set departure and arrival times (e.g., departure 10 minutes after epoch, transfer duration 40 minutes)
    departure_time = epoch + 10*u.min
    arrival_time = departure_time + 40*u.min
    
    # Compute transfer orbit via Lambert's problem
    v_dep, v_arr, r_dep, r_arr = compute_transfer_orbit(iss_orbit, target_debris["orbit"], departure_time, arrival_time)
    transfer_time = (arrival_time - departure_time).to(u.s).value
    print(f"Transfer orbit computed with transfer time {transfer_time:.2f} seconds")
    
    # Compute delta-v requirements (approximated)
    iss_v = iss_orbit.v.to(u.km/u.s).value
    dep_delta_v = np.linalg.norm(v_dep.to(u.km/u.s).value - iss_v)
    debris_v = target_debris["orbit"].v.to(u.km/u.s).value
    arr_delta_v = np.linalg.norm(debris_v - v_arr.to(u.km/u.s).value)
    total_delta_v = dep_delta_v + arr_delta_v
    print(f"Delta-v required: Departure = {dep_delta_v:.2f} km/s, Arrival = {arr_delta_v:.2f} km/s, Total = {total_delta_v:.2f} km/s")
    print(f"Mission duration: {transfer_time:.2f} seconds")
    
    print(f"Collecting debris {target_debris['name']}...")
    print("Initiating rock crusher on ISS...")
    print(f"Debris {target_debris['name']} crushed to dust particles.\n")
    
    # Visualize trajectories (ISS orbit, debris orbit, and transfer trajectory)
    visualize_trajectories(iss_orbit, target_debris["orbit"], departure_time, arrival_time)
    
    return total_delta_v, transfer_time

# =====================================================
# 8. Visualize Mission Trajectories in 3D
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
        # Note: This is a naive cylindrical projection
        facecolors = np.empty(x.shape + (4,), dtype=np.float32)
        img_h, img_w, _ = img.shape
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                s = u_vals[i] / (2*np.pi)
                t = (v_vals[j]) / np.pi  # v from 0 to π maps to t from 0 to 1
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
    
    # Compute transfer trajectory via linear interpolation
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
    
    ax.set_title("Debris Collector Mission Trajectory")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    # Set axis limits to roughly encompass Earth and the orbits
    max_range = 1.5 * R_earth
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.show()

# =====================================================
# 9. Additional: Plot a Global Debris Map (Globe Simulation)
# =====================================================
def propagate_tles(tle_list):
    """
    For each TLE (name, line1, line2) in tle_list, propagate it to the current time
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
    
    # Earth sphere
    R_earth = 6378.0
    u_vals = np.linspace(0, 2*np.pi, 60)
    v_vals = np.linspace(-np.pi/2, np.pi/2, 30)
    x = R_earth * np.outer(np.cos(u_vals), np.cos(v_vals))
    y = R_earth * np.outer(np.sin(u_vals), np.cos(v_vals))
    z = R_earth * np.outer(np.ones_like(u_vals), np.sin(v_vals))
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    
    # Debris points
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
    Fetch debris TLE data from DEBRIS_TLE_URL, propagate them to current time,
    and plot a global 3D debris map.
    """
    tle_data = fetch_debris_tle()
    if not tle_data:
        print("No debris TLE data found. Cannot plot globe simulation.")
        return
    positions = propagate_tles(tle_data)
    print(f"Propagated {len(positions)} debris objects to current time.")
    plot_debris(positions)

# =====================================================
# 10. Main Function
# =====================================================
def main():
    # First, simulate the mission (ISS to target debris)
    debris_list = fetch_debris_tle()
    if not debris_list:
        print("No TLE data fetched. Using simulated debris objects.")
        debris_list = []
        for i in range(5):
            debris_list.append({
                "name": f"Simulated_Debris_{i+1}",
                "line1": "",
                "line2": ""
            })
    debris_list = assign_debris_size_and_criticality(debris_list)
    
    print("Detected debris objects (size in m, mass in kg, criticality):")
    for debris in debris_list:
        print(f"  {debris['name']}: Size = {debris['size']:.2f} m, Mass = {debris['mass']:.2f} kg, Criticality = {debris['criticality']:.2f}")
    
    total_delta_v, mission_duration = simulate_mission(debris_list)
    print("Mission simulation complete.")
    print(f"Total delta-v required: {total_delta_v:.2f} km/s")
    print(f"Mission duration: {mission_duration:.2f} seconds")
    
    # Next, show a full globe simulation with debris scattered around Earth
    plot_debris_globe()

# =====================================================
# Run the Script
# =====================================================
if __name__ == '__main__':
    main()
