import streamlit as st
import os
import math
import random
import numpy as np
import requests
from datetime import datetime

# Astropy / Poliastro imports
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.iod import izzo

# SGP4 imports
from sgp4.api import Satrec, jday

# Matplotlib imports
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt

# ============================
# GLOBAL SETTINGS and API URLs
# ============================
DEBRIS_TLE_URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?CATNR=25544&FORMAT=TLE"
ISS_TLE_URL    = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE"
MAX_OBJECTS = 500

# Specify your high-res Earth texture file here (ensure it's in the working directory)
USE_TEXTURED_EARTH = True
EARTH_TEXTURE_FILE = "earth_texture.jpg"  # Replace with a high-quality Earth image file

# ============================
# Helper Functions
# ============================

def ai_detect_debris_size(image_path):
    """
    Simulated AI image processing for debris size.
    The detection runs in the backend without logging per-segment messages.
    Returns a random size between 0.5 and 2.0 meters.
    """
    # No UI output here; processing happens quietly in the background.
    return random.uniform(0.5, 2.0)

def fetch_tle_data(url):
    try:
        resp = requests.get(url, timeout=10)
        lines = resp.text.strip().splitlines()
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
        else:
            return None, None, None
    except Exception as e:
        st.error(f"[API] Error fetching TLE data: {e}")
        return None, None, None

def fetch_debris_tle():
    debris_list = []
    try:
        st.write(f"**[API] Fetching debris TLE data from:** {DEBRIS_TLE_URL}")
        resp = requests.get(DEBRIS_TLE_URL, timeout=10)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
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
        debris_list = debris_list[:MAX_OBJECTS]
    except Exception as e:
        st.error(f"[API] Error fetching debris TLE: {e}")
    return debris_list

def fetch_debris_mass(debris):
    # Simulated external query: assign random mass between 0 and 20 kg
    debris["mass"] = random.uniform(0, 20)
    return debris

def assign_debris_size_and_criticality(debris_list):
    for d in debris_list:
        # The AI detection runs quietly in the backend.
        image_path = f"images/{d['name'].replace(' ', '_')}.jpg"
        d["size"] = ai_detect_debris_size(image_path)
        d = fetch_debris_mass(d)
        d["criticality"] = d["size"] + d["mass"] / 1000.0
    return debris_list

def create_orbit_from_tle(debris, epoch):
    try:
        sat = Satrec.twoline2rv(debris["line1"], debris["line2"])
        jd, fr = jday(epoch.datetime.year, epoch.datetime.month, epoch.datetime.day,
                      epoch.datetime.hour, epoch.datetime.minute,
                      epoch.datetime.second + epoch.datetime.microsecond*1e-6)
        error, r, v = sat.sgp4(jd, fr)
        if error != 0:
            st.warning(f"[WARN] sgp4 error for {debris['name']}: code {error}")
            raise ValueError("Propagation error")
        r = np.array(r) * u.km
        v = np.array(v) * (u.km / u.s)
        orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch)
        return orbit
    except Exception as e:
        st.error(f"[ERROR] Creating orbit for {debris['name']}: {e}")
        return None

def compute_transfer_orbit(iss_orbit, debris_orbit, departure_time, arrival_time):
    r1 = iss_orbit.propagate(departure_time - iss_orbit.epoch).r
    r2 = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r
    (v1, v2), = izzo.lambert(Earth.k, r1, r2, (arrival_time - departure_time).to(u.s))
    return v1, v2, r1, r2

def simulate_mission(debris_list, log_func=st.write):
    """
    Simulates the debris collector mission.
    Logs detailed calculation steps.
    Also assigns the computed departure velocity to the debris collector.
    """
    epoch = Time(datetime.utcnow())

    # Create a realistic ISS orbit (approx. 400 km altitude, 51.6° inclination)
    iss_orbit = Orbit.from_classical(
        attractor=Earth,
        a=(Earth.R + 400*u.km),
        ecc=0.001*u.one,
        inc=51.6*u.deg,
        raan=0*u.deg,
        argp=0*u.deg,
        nu=0*u.deg,
        epoch=epoch
    )

    # Convert TLE to orbit for each debris or simulate if TLE fails
    for d in debris_list:
        if d.get("line1") and d.get("line2"):
            orb = create_orbit_from_tle(d, epoch)
            if orb is None:
                alt = random.uniform(350,450)*u.km
                inc = random.uniform(45,55)*u.deg
                orb = Orbit.from_classical(Earth, Earth.R+alt, 0.001*u.one, inc,
                                           0*u.deg, 0*u.deg, 0*u.deg, epoch=epoch)
            d["orbit"] = orb
        else:
            alt = random.uniform(350,450)*u.km
            inc = random.uniform(45,55)*u.deg
            d["orbit"] = Orbit.from_classical(Earth, Earth.R+alt, 0.001*u.one, inc,
                                              0*u.deg, 0*u.deg, 0*u.deg, epoch=epoch)
    # Sort debris by criticality (highest first)
    debris_list.sort(key=lambda x: x["criticality"], reverse=True)
    target = debris_list[0]
    log_func(f"**[MISSION] Selected target debris:** {target['name']} (Size: {target['size']:.2f} m, Mass: {target['mass']:.2f} kg)")

    departure_time = epoch + 10*u.min
    arrival_time = departure_time + 40*u.min

    v_dep, v_arr, r_dep, r_arr = compute_transfer_orbit(iss_orbit, target["orbit"], departure_time, arrival_time)
    transfer_duration = (arrival_time - departure_time).to(u.s).value

    # Calculate delta-v requirements
    iss_v = iss_orbit.v.to(u.km/u.s).value
    dep_dv = np.linalg.norm(v_dep.to(u.km/u.s).value - iss_v)
    debris_v = target["orbit"].v.to(u.km/u.s).value
    arr_dv = np.linalg.norm(debris_v - v_arr.to(u.km/u.s).value)
    total_dv = dep_dv + arr_dv

    # Assign the departure velocity to the debris collector (for simulation)
    collector_velocity = v_dep.to(u.km/u.s).value

    log_func(f"**[MISSION] Transfer orbit computed; transfer duration = {transfer_duration:.2f} seconds**")
    log_func(f"**[MISSION] Delta-v requirements:** Departure = {dep_dv:.2f} km/s, Arrival = {arr_dv:.2f} km/s, Total = {total_dv:.2f} km/s")
    log_func(f"**[MISSION] Assigned collector velocity:** {collector_velocity[0]:.2f} km/s (x-component) [simulated]")
    log_func(f"**[MISSION] Collecting debris {target['name']} ...**")
    log_func(f"**[MISSION] Initiating rock crusher on ISS... Debris crushed to dust.**")
    log_func(f"**[RESULT] Debris collector mission complete. Total delta-v = {total_dv:.2f} km/s, Mission duration = {transfer_duration:.2f} s**")

    return iss_orbit, target["orbit"], departure_time, arrival_time

def plot_earth(ax):
    """
    Plots Earth on the given 3D axis.
    If USE_TEXTURED_EARTH is True and a valid texture file exists,
    maps the image onto the sphere for a realistic appearance.
    """
    R_earth = Earth.R.to(u.km).value
    u_vals = np.linspace(0, 2*np.pi, 100)
    v_vals = np.linspace(0, np.pi, 100)
    X = R_earth * np.outer(np.cos(u_vals), np.sin(v_vals))
    Y = R_earth * np.outer(np.sin(u_vals), np.sin(v_vals))
    Z = R_earth * np.outer(np.ones_like(u_vals), np.cos(v_vals))
    if USE_TEXTURED_EARTH and os.path.isfile(EARTH_TEXTURE_FILE):
        img = plt.imread(EARTH_TEXTURE_FILE)
        facecolors = np.empty(X.shape + (4,), dtype=np.float32)
        img_h, img_w, _ = img.shape
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                s = u_vals[i] / (2*np.pi)
                t = v_vals[j] / np.pi
                x_img = int(s * (img_w - 1))
                y_img = int((1-t) * (img_h - 1))
                facecolors[i, j, :] = img[y_img, x_img, :4]
        ax.plot_surface(X, Y, Z, rcount=X.shape[0], ccount=X.shape[1],
                        facecolors=facecolors, shade=False)
    else:
        ax.plot_surface(X, Y, Z, color='b', alpha=0.3)
    return ax

def plot_mission_3d(iss_orbit, debris_orbit, departure_time, arrival_time):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax = plot_earth(ax)
    # Plot ISS orbit
    iss_points = iss_orbit.sample(200)
    ax.plot(iss_points.x.to(u.km).value,
            iss_points.y.to(u.km).value,
            iss_points.z.to(u.km).value,
            color='green', label="ISS Orbit")
    # Plot debris orbit
    debris_points = debris_orbit.sample(200)
    ax.plot(debris_points.x.to(u.km).value,
            debris_points.y.to(u.km).value,
            debris_points.z.to(u.km).value,
            color='red', label="Debris Orbit")
    # Plot transfer trajectory (linear interpolation)
    t_vals = np.linspace(0, (arrival_time - departure_time).to(u.s).value, 100)
    r_dep = iss_orbit.propagate(departure_time - iss_orbit.epoch).r.value
    r_arr = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r.value
    transfer_points = [r_dep + (r_arr - r_dep)*(t/t_vals[-1]) for t in t_vals]
    transfer_points = np.array(transfer_points)
    ax.plot(transfer_points[:,0], transfer_points[:,1], transfer_points[:,2],
            color='orange', linestyle='--', label="Transfer Trajectory")
    ax.set_title("Debris Collector Mission")
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
    max_r = 1.5 * Earth.R.to(u.km).value
    ax.set_xlim([-max_r, max_r]); ax.set_ylim([-max_r, max_r]); ax.set_zlim([-max_r, max_r])
    ax.legend()
    return fig

def simulate_satellite_launch(lat_deg, lon_deg, altitude_km, log_func=st.write):
    epoch = Time(datetime.utcnow())
    # Naively use the absolute value of latitude as the inclination.
    inc = abs(lat_deg)*u.deg
    if inc > 90*u.deg:
        inc = 90*u.deg
    sat_orbit = Orbit.from_classical(
        attractor=Earth,
        a=(Earth.R + altitude_km*u.km),
        ecc=0.001*u.one,
        inc=inc,
        raan=0*u.deg,
        argp=0*u.deg,
        nu=0*u.deg,
        epoch=epoch
    )
    log_func(f"**[SATELLITE] Simulated satellite launched from Earth (altitude: {altitude_km} km, inc: {inc})**")
    return sat_orbit

def plot_satellite_orbit_3d(sat_orbit, title="Satellite Orbit"):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax = plot_earth(ax)
    pts = sat_orbit.sample(200)
    ax.plot(pts.x.to(u.km).value,
            pts.y.to(u.km).value,
            pts.z.to(u.km).value,
            color='magenta', label="Satellite Orbit")
    ax.set_title(title)
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
    max_r = 1.5 * Earth.R.to(u.km).value
    ax.set_xlim([-max_r, max_r]); ax.set_ylim([-max_r, max_r]); ax.set_zlim([-max_r, max_r])
    ax.legend()
    return fig

def propagate_tles(tle_list):
    now = datetime.utcnow()
    jd_now, fr_now = jday(now.year, now.month, now.day, now.hour, now.minute, now.second+now.microsecond*1e-6)
    pos_list = []
    for (nm, l1, l2) in tle_list:
        try:
            s = Satrec.twoline2rv(l1, l2)
            e, r, v = s.sgp4(jd_now, fr_now)
            if e == 0:
                pos_list.append((nm, r[0], r[1], r[2]))
        except Exception:
            pass
    return pos_list

def plot_debris_globe(debris_positions):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    R_earth = 6378.0
    u_vals = np.linspace(0, 2*np.pi, 60)
    v_vals = np.linspace(-np.pi/2, np.pi/2, 30)
    X = R_earth * np.outer(np.cos(u_vals), np.cos(v_vals))
    Y = R_earth * np.outer(np.sin(u_vals), np.cos(v_vals))
    Z = R_earth * np.outer(np.ones_like(u_vals), np.sin(v_vals))
    ax.plot_surface(X, Y, Z, color='blue', alpha=0.3)
    xs = [p[1] for p in debris_positions]
    ys = [p[2] for p in debris_positions]
    zs = [p[3] for p in debris_positions]
    ax.scatter(xs, ys, zs, s=2, color='red', alpha=0.5, label='Debris')
    ax.set_title("Global Debris Map")
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
    mr = 1.2 * max(max(np.abs(xs)) if xs else R_earth, max(np.abs(ys)) if ys else R_earth, max(np.abs(zs)) if zs else R_earth, R_earth)
    ax.set_xlim([-mr, mr]); ax.set_ylim([-mr, mr]); ax.set_zlim([-mr, mr])
    ax.legend()
    return fig

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="Debris & Satellite Simulation", layout="wide")
st.title("🚀 Autonomous Debris Removal & Satellite Launch Simulation")

# Use Streamlit's session_state to store data across interactions
if 'debris_list' not in st.session_state:
    st.session_state.debris_list = None
if 'iss_orbit' not in st.session_state:
    st.session_state.iss_orbit = None
if 'debris_orbit' not in st.session_state:
    st.session_state.debris_orbit = None
if 'dep_time' not in st.session_state:
    st.session_state.dep_time = None
if 'arr_time' not in st.session_state:
    st.session_state.arr_time = None
if 'sat_orbit' not in st.session_state:
    st.session_state.sat_orbit = None

# Sidebar for Satellite Launch Inputs
st.sidebar.header("Satellite Launch Parameters")
lat_input = st.sidebar.text_input("Launch Latitude (°)", "28.5")
lon_input = st.sidebar.text_input("Launch Longitude (°)", "-80.6")
alt_input = st.sidebar.text_input("Launch Altitude (km)", "500")

# Sidebar Buttons
if st.sidebar.button("Fetch Debris TLE"):
    st.markdown("**[API] Fetching debris TLE data...**")
    debris = fetch_debris_tle()
    if debris:
        st.session_state.debris_list = assign_debris_size_and_criticality(debris)
        st.markdown(f"**[API] Fetched {len(st.session_state.debris_list)} debris objects.**")
        for d in st.session_state.debris_list:
            st.write(f"* {d['name']}: size = {d['size']:.2f} m, mass = {d['mass']:.2f} kg, crit = {d['criticality']:.2f}*")
    else:
        st.error("[API] No debris TLE data fetched.")

if st.sidebar.button("Simulate Debris Mission"):
    if st.session_state.debris_list is None:
        st.error("[ERROR] No debris data. Please fetch debris TLE first!")
    else:
        st.markdown("**[MISSION] Simulating debris-collector mission from ISS...**")
        iss_orbit, debris_orbit, dep_time, arr_time = simulate_mission(st.session_state.debris_list, log_func=st.markdown)
        st.session_state.iss_orbit = iss_orbit
        st.session_state.debris_orbit = debris_orbit
        st.session_state.dep_time = dep_time
        st.session_state.arr_time = arr_time
        fig = plot_mission_3d(iss_orbit, debris_orbit, dep_time, arr_time)
        st.pyplot(fig)

if st.sidebar.button("Simulate Satellite Launch"):
    try:
        lat_val = float(lat_input)
        lon_val = float(lon_input)
        alt_val = float(alt_input)
    except Exception as e:
        st.error("[ERROR] Invalid input for latitude, longitude, or altitude.")
    else:
        st.markdown("**[SATELLITE] Simulating satellite launch...**")
        sat_orbit = simulate_satellite_launch(lat_val, lon_val, alt_val, log_func=st.markdown)
        st.session_state.sat_orbit = sat_orbit
        fig_sat = plot_satellite_orbit_3d(sat_orbit, title="Satellite Orbit from Launch Site")
        st.pyplot(fig_sat)

if st.sidebar.button("Plot Global Debris Map"):
    if st.session_state.debris_list is None:
        st.error("[ERROR] No debris data. Please fetch debris TLE first!")
    else:
        tle_list = [(d["name"], d["line1"], d["line2"]) for d in st.session_state.debris_list if d["line1"] and d["line2"]]
        positions = propagate_tles(tle_list)
        st.markdown(f"**[INFO] Propagated {len(positions)} debris objects to current time.**")
        fig_globe = plot_debris_globe(positions)
        st.pyplot(fig_globe)

st.markdown("### Instructions")
st.markdown(
    """
    1. Use the **Sidebar** to set your satellite launch parameters.
    2. Click **Fetch Debris TLE** to retrieve debris data.
    3. Click **Simulate Debris Mission** to compute the ISS-to-debris transfer trajectory and see detailed calculations.
    4. Click **Simulate Satellite Launch** to view a satellite orbit based on your launch site.
    5. Click **Plot Global Debris Map** to view a 3D map of debris around Earth.
    """
)
