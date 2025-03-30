import streamlit as st
import sys
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
matplotlib.use("Agg")  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt

# ============================
# GLOBAL SETTINGS and API URLs
# ============================
DEBRIS_TLE_URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?CATNR=25544&FORMAT=TLE"
ISS_TLE_URL    = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE"
MAX_OBJECTS = 500

# ============================
# Helper Functions (Same as Before)
# ============================

def ai_detect_debris_size(image_path):
    # Simulate image processing: return random size in meters
    st.info(f"Simulating AI detection for image: {image_path}")
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
        st.error(f"Error fetching TLE data: {e}")
        return None, None, None

def fetch_debris_tle():
    debris_list = []
    try:
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
        st.error(f"Error fetching debris TLE: {e}")
    return debris_list

def fetch_debris_mass(debris):
    # Simulate external mass query: return random mass (kg)
    debris["mass"] = random.uniform(0, 20)
    return debris

def assign_debris_size_and_criticality(debris_list):
    for d in debris_list:
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
            st.warning(f"sgp4 error for {debris['name']}: code {error}")
            raise ValueError("Propagation error")
        r = np.array(r) * u.km
        v = np.array(v) * (u.km / u.s)
        orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch)
        return orbit
    except Exception as e:
        st.error(f"Error creating orbit for {debris['name']}: {e}")
        return None

def compute_transfer_orbit(iss_orbit, debris_orbit, departure_time, arrival_time):
    r1 = iss_orbit.propagate(departure_time - iss_orbit.epoch).r
    r2 = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r
    (v1, v2), = izzo.lambert(Earth.k, r1, r2, (arrival_time - departure_time).to(u.s))
    return v1, v2, r1, r2

def simulate_mission(debris_list, log_func=st.write):
    epoch = Time(datetime.utcnow())
    # Define a realistic ISS orbit (approx. 400 km altitude, 51.6° inclination)
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
    # For each debris, try to create an orbit; otherwise, simulate one.
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
    # Sort by descending criticality
    debris_list.sort(key=lambda x: x["criticality"], reverse=True)
    target = debris_list[0]
    log_func(f"**[MISSION] Target debris:** {target['name']} (Size={target['size']:.2f} m, Mass={target['mass']:.2f} kg)")
    departure_time = epoch + 10*u.min
    arrival_time = departure_time + 40*u.min
    v_dep, v_arr, r_dep, r_arr = compute_transfer_orbit(iss_orbit, target["orbit"], departure_time, arrival_time)
    ttime = (arrival_time - departure_time).to(u.s).value
    iss_v = iss_orbit.v.to(u.km/u.s).value
    dep_dv = np.linalg.norm(v_dep.to(u.km/u.s).value - iss_v)
    debris_v = target["orbit"].v.to(u.km/u.s).value
    arr_dv = np.linalg.norm(debris_v - v_arr.to(u.km/u.s).value)
    total_dv = dep_dv + arr_dv
    log_func(f"**[MISSION] Transfer time:** {ttime:.2f} s")
    log_func(f"**[MISSION] Delta-v:** Dep = {dep_dv:.2f} km/s, Arr = {arr_dv:.2f} km/s, Total = {total_dv:.2f} km/s")
    log_func("**[MISSION] Debris collected and crushed.**")
    return iss_orbit, target["orbit"], departure_time, arrival_time

def plot_mission_3d(iss_orbit, debris_orbit, departure_time, arrival_time):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    R_earth = Earth.R.to(u.km).value
    u_vals = np.linspace(0, 2*np.pi, 60)
    v_vals = np.linspace(0, np.pi, 30)
    X = R_earth * np.outer(np.cos(u_vals), np.sin(v_vals))
    Y = R_earth * np.outer(np.sin(u_vals), np.sin(v_vals))
    Z = R_earth * np.outer(np.ones_like(u_vals), np.cos(v_vals))
    ax.plot_surface(X, Y, Z, color='b', alpha=0.3)
    # ISS orbit
    iss_points = iss_orbit.sample(200)
    ax.plot(iss_points.x.to(u.km).value,
            iss_points.y.to(u.km).value,
            iss_points.z.to(u.km).value,
            color='green', label="ISS Orbit")
    # Debris orbit
    debris_points = debris_orbit.sample(200)
    ax.plot(debris_points.x.to(u.km).value,
            debris_points.y.to(u.km).value,
            debris_points.z.to(u.km).value,
            color='red', label="Debris Orbit")
    # Transfer trajectory (linear interpolation)
    t_vals = np.linspace(0, (arrival_time - departure_time).to(u.s).value, 100)
    r_dep = iss_orbit.propagate(departure_time - iss_orbit.epoch).r.value
    r_arr = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r.value
    transfer_points = []
    for t in t_vals:
        r = r_dep + (r_arr - r_dep) * (t / t_vals[-1])
        transfer_points.append(r)
    transfer_points = np.array(transfer_points)
    ax.plot(transfer_points[:,0], transfer_points[:,1], transfer_points[:,2],
            color='orange', linestyle='--', label="Transfer")
    ax.set_title("Debris Collector Mission")
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
    max_r = 1.5*R_earth
    ax.set_xlim([-max_r, max_r]); ax.set_ylim([-max_r, max_r]); ax.set_zlim([-max_r, max_r])
    ax.legend()
    return fig

def simulate_satellite_launch(lat_deg, lon_deg, altitude_km, log_func=st.write):
    epoch = Time(datetime.utcnow())
    # Naively set inclination = |lat| (for demonstration)
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
    log_func(f"**[SATELLITE] Launched:** lat={lat_deg:.2f}°, lon={lon_deg:.2f}°, alt={altitude_km} km → inc={inc}")
    return sat_orbit

def plot_satellite_orbit_3d(sat_orbit, title="Satellite Orbit"):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    R_earth = Earth.R.to(u.km).value
    u_vals = np.linspace(0, 2*np.pi, 60)
    v_vals = np.linspace(0, np.pi, 30)
    X = R_earth * np.outer(np.cos(u_vals), np.sin(v_vals))
    Y = R_earth * np.outer(np.sin(u_vals), np.sin(v_vals))
    Z = R_earth * np.outer(np.ones_like(u_vals), np.cos(v_vals))
    ax.plot_surface(X, Y, Z, color='b', alpha=0.3)
    pts = sat_orbit.sample(200)
    ax.plot(pts.x.to(u.km).value,
            pts.y.to(u.km).value,
            pts.z.to(u.km).value,
            color='magenta', label="Satellite Orbit")
    ax.set_title(title)
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
    max_r = 1.5*R_earth
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
    mr = 1.2*max(max(np.abs(xs)) if xs else R_earth, max(np.abs(ys)) if ys else R_earth, max(np.abs(zs)) if zs else R_earth, R_earth)
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
    st.write("**Fetching debris TLE data...**")
    debris = fetch_debris_tle()
    if debris:
        st.session_state.debris_list = assign_debris_size_and_criticality(debris)
        st.write(f"Fetched {len(st.session_state.debris_list)} debris objects.")
        for d in st.session_state.debris_list:
            st.write(f"*{d['name']}*: size={d['size']:.2f} m, mass={d['mass']:.2f} kg, crit={d['criticality']:.2f}")
    else:
        st.error("No debris TLE data fetched.")

if st.sidebar.button("Simulate Debris Mission"):
    if st.session_state.debris_list is None:
        st.error("Please fetch debris data first!")
    else:
        st.write("**Simulating debris-collector mission from ISS...**")
        iss_orbit, debris_orbit, dep_time, arr_time = simulate_mission(st.session_state.debris_list, log_func=st.write)
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
        st.error("Invalid input for latitude, longitude, or altitude.")
    else:
        st.write("**Simulating satellite launch...**")
        sat_orbit = simulate_satellite_launch(lat_val, lon_val, alt_val, log_func=st.write)
        st.session_state.sat_orbit = sat_orbit
        fig_sat = plot_satellite_orbit_3d(sat_orbit, title="Satellite Orbit from Launch Site")
        st.pyplot(fig_sat)

if st.sidebar.button("Plot Global Debris Map"):
    if st.session_state.debris_list is None:
        st.error("Fetch debris data first!")
    else:
        tle_list = [(d["name"], d["line1"], d["line2"]) for d in st.session_state.debris_list if d["line1"] and d["line2"]]
        positions = propagate_tles(tle_list)
        st.write(f"Propagated {len(positions)} debris objects to current time.")
        fig_globe = plot_debris_globe(positions)
        st.pyplot(fig_globe)

# Main area: you can also show logs or instructions
st.markdown("### Instructions")
st.markdown(
    """
    1. Use the **Sidebar** to input your launch parameters.
    2. Click **Fetch Debris TLE** to retrieve and process debris data.
    3. Click **Simulate Debris Mission** to compute the ISS-to-debris transfer trajectory.
    4. Click **Simulate Satellite Launch** to see the orbit of a satellite launched from your chosen site.
    5. Click **Plot Global Debris Map** to view a 3D map of debris.
    """
)
