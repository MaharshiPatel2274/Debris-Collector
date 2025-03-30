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

# Plotly for interactive 3D
import plotly.graph_objects as go

# ============================
# GLOBAL SETTINGS and API URLs
# ============================
DEBRIS_TLE_URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?CATNR=25544&FORMAT=TLE"
ISS_TLE_URL    = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE"
MAX_OBJECTS = 500

# If you have a texture file, you can still do a color-coded sphere in Plotly. 
# We'll keep it simple by coloring Earth with a built-in colorscale for now.

# ============================
# Helper Functions
# ============================

def ai_detect_debris_size(image_path):
    # Simulated AI detection. Quietly returns random size [0.5..2.0].
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
            if i + 2 < len(lines):
                name = lines[i].strip()
                line1 = lines[i+1].strip()
                line2 = lines[i+2].strip()
                debris_list.append({"name": name, "line1": line1, "line2": line2})
        debris_list = debris_list[:MAX_OBJECTS]
    except Exception as e:
        st.error(f"[API] Error fetching debris TLE: {e}")
    return debris_list

def fetch_debris_mass(debris):
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
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            st.warning(f"[WARN] sgp4 error for {debris['name']}: code {e}")
            raise ValueError("Propagation error")
        r = np.array(r) * u.km
        v = np.array(v) * (u.km / u.s)
        orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch)
        return orbit
    except Exception as ex:
        st.error(f"[ERROR] Creating orbit from TLE for {debris['name']}: {ex}")
        return None

def compute_transfer_orbit(iss_orbit, debris_orbit, departure_time, arrival_time):
    r1 = iss_orbit.propagate(departure_time - iss_orbit.epoch).r
    r2 = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r
    (v1, v2), = izzo.lambert(Earth.k, r1, r2, (arrival_time - departure_time).to(u.s))
    return v1, v2, r1, r2

def simulate_mission(debris_list, log_func=st.write):
    epoch = Time(datetime.utcnow())
    # ISS orbit
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
    # Create orbits for debris
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
    debris_list.sort(key=lambda x: x["criticality"], reverse=True)
    target = debris_list[0]
    log_func(f"**[MISSION] Selected target debris:** {target['name']} (Size={target['size']:.2f} m, Mass={target['mass']:.2f} kg)")

    departure_time = epoch + 10*u.min
    arrival_time = departure_time + 40*u.min

    v_dep, v_arr, r_dep, r_arr = compute_transfer_orbit(iss_orbit, target["orbit"], departure_time, arrival_time)
    ttime = (arrival_time - departure_time).to(u.s).value

    iss_v = iss_orbit.v.to(u.km/u.s).value
    dep_dv = np.linalg.norm(v_dep.to(u.km/u.s).value - iss_v)
    debris_v = target["orbit"].v.to(u.km/u.s).value
    arr_dv = np.linalg.norm(debris_v - v_arr.to(u.km/u.s).value)
    total_dv = dep_dv + arr_dv

    log_func(f"**[MISSION] Transfer orbit computed; transfer duration = {ttime:.2f} seconds**")
    log_func(f"**[MISSION] Delta-v:** Dep = {dep_dv:.2f} km/s, Arr = {arr_dv:.2f} km/s, Total = {total_dv:.2f} km/s")
    log_func(f"**[MISSION] Debris {target['name']} collected & crushed.**")
    log_func(f"**[RESULT] Mission complete. Δv={total_dv:.2f} km/s, time={ttime:.2f} s**")

    return iss_orbit, target["orbit"], departure_time, arrival_time

# ------------------------------
# Plotly 3D Earth + Orbits
# ------------------------------
def plot_mission_3d_interactive(iss_orbit, debris_orbit, departure_time, arrival_time):
    """
    Uses Plotly to create an interactive 3D scene with:
      - Earth as a sphere
      - ISS orbit (green)
      - Debris orbit (red)
      - Transfer path (orange dashed)
    You can rotate and zoom this figure in the Streamlit UI.
    """
    # 1) Create Earth sphere with param grids
    R_earth = Earth.R.to(u.km).value
    n_u, n_v = 50, 50
    u_vals = np.linspace(0, 2*np.pi, n_u)
    v_vals = np.linspace(0, np.pi, n_v)
    xs = R_earth * np.outer(np.cos(u_vals), np.sin(v_vals))
    ys = R_earth * np.outer(np.sin(u_vals), np.sin(v_vals))
    zs = R_earth * np.outer(np.ones_like(u_vals), np.cos(v_vals))

    earth_surface = go.Surface(
        x=xs, y=ys, z=zs,
        colorscale="Blues",
        opacity=0.3,
        name="Earth"
    )

    # 2) Sample orbits
    iss_pts = iss_orbit.sample(200)
    iss_x = iss_pts.x.to(u.km).value
    iss_y = iss_pts.y.to(u.km).value
    iss_z = iss_pts.z.to(u.km).value

    deb_pts = debris_orbit.sample(200)
    deb_x = deb_pts.x.to(u.km).value
    deb_y = deb_pts.y.to(u.km).value
    deb_z = deb_pts.z.to(u.km).value

    # 3) Transfer path by linear interpolation
    t_vals = np.linspace(0, (arrival_time - departure_time).to(u.s).value, 100)
    r_dep = iss_orbit.propagate(departure_time - iss_orbit.epoch).r.value
    r_arr = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r.value
    transfer_xyz = []
    for t in t_vals:
        frac = t / t_vals[-1]
        r = r_dep + frac * (r_arr - r_dep)
        transfer_xyz.append(r)
    transfer_xyz = np.array(transfer_xyz)
    tr_x = transfer_xyz[:,0]
    tr_y = transfer_xyz[:,1]
    tr_z = transfer_xyz[:,2]

    # 4) Create Scatter3d traces
    iss_trace = go.Scatter3d(
        x=iss_x, y=iss_y, z=iss_z,
        mode='lines',
        line=dict(color='green', width=3),
        name="ISS Orbit"
    )
    debris_trace = go.Scatter3d(
        x=deb_x, y=deb_y, z=deb_z,
        mode='lines',
        line=dict(color='red', width=3),
        name="Debris Orbit"
    )
    transfer_trace = go.Scatter3d(
        x=tr_x, y=tr_y, z=tr_z,
        mode='lines',
        line=dict(color='orange', width=3, dash='dot'),
        name="Transfer"
    )

    # 5) Build the figure
    fig = go.Figure(data=[earth_surface, iss_trace, debris_trace, transfer_trace])
    fig.update_layout(
        title="Debris Collector Mission (Interactive)",
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data'
        )
    )
    return fig

# ------------------------------
# Satellite Launch
# ------------------------------
def simulate_satellite_launch(lat_deg, lon_deg, alt_km, log_func=st.write):
    epoch = Time(datetime.utcnow())
    inc = abs(lat_deg)*u.deg
    if inc > 90*u.deg:
        inc = 90*u.deg
    sat_orb = Orbit.from_classical(
        attractor=Earth,
        a=(Earth.R + alt_km*u.km),
        ecc=0.001*u.one,
        inc=inc,
        raan=0*u.deg,
        argp=0*u.deg,
        nu=0*u.deg,
        epoch=epoch
    )
    log_func(f"**[SATELLITE] Launched from lat={lat_deg:.2f}, lon={lon_deg:.2f}, alt={alt_km} km => inc={inc}**")
    return sat_orb

def plot_satellite_orbit_3d_interactive(sat_orbit):
    """
    Similar to the mission plot, but only for the satellite orbit + Earth.
    """
    # Earth sphere
    R_earth = Earth.R.to(u.km).value
    n_u, n_v = 50, 50
    u_vals = np.linspace(0, 2*np.pi, n_u)
    v_vals = np.linspace(0, np.pi, n_v)
    xs = R_earth * np.outer(np.cos(u_vals), np.sin(v_vals))
    ys = R_earth * np.outer(np.sin(u_vals), np.sin(v_vals))
    zs = R_earth * np.outer(np.ones_like(u_vals), np.cos(v_vals))

    earth_surface = go.Surface(
        x=xs, y=ys, z=zs,
        colorscale="Blues",
        opacity=0.3,
        name="Earth"
    )

    # Satellite orbit
    pts = sat_orbit.sample(200)
    sx = pts.x.to(u.km).value
    sy = pts.y.to(u.km).value
    sz = pts.z.to(u.km).value
    sat_trace = go.Scatter3d(
        x=sx, y=sy, z=sz,
        mode='lines',
        line=dict(color='magenta', width=3),
        name="Satellite Orbit"
    )

    fig = go.Figure(data=[earth_surface, sat_trace])
    fig.update_layout(
        title="Satellite Orbit (Interactive)",
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data'
        )
    )
    return fig

# ------------------------------
# Debris Map
# ------------------------------
def propagate_tles(tle_list):
    now = datetime.utcnow()
    jd_now, fr_now = jday(now.year, now.month, now.day,
                          now.hour, now.minute, now.second + now.microsecond*1e-6)
    positions = []
    for (nm, line1, line2) in tle_list:
        try:
            sat = Satrec.twoline2rv(line1, line2)
            e, r, v = sat.sgp4(jd_now, fr_now)
            if e == 0:
                positions.append((nm, r[0], r[1], r[2]))
        except:
            pass
    return positions

def plot_debris_map_3d_plotly(debris_positions):
    R_earth = 6378.0
    n_u, n_v = 50, 50
    u_vals = np.linspace(0, 2*np.pi, n_u)
    v_vals = np.linspace(-np.pi/2, np.pi/2, n_v)
    xs = R_earth * np.outer(np.cos(u_vals), np.cos(v_vals))
    ys = R_earth * np.outer(np.sin(u_vals), np.cos(v_vals))
    zs = R_earth * np.outer(np.ones_like(u_vals), np.sin(v_vals))

    earth_surf = go.Surface(
        x=xs, y=ys, z=zs,
        colorscale="Blues",
        opacity=0.3,
        name="Earth"
    )
    xpts = [p[1] for p in debris_positions]
    ypts = [p[2] for p in debris_positions]
    zpts = [p[3] for p in debris_positions]
    debris_trace = go.Scatter3d(
        x=xpts, y=ypts, z=zpts,
        mode='markers',
        marker=dict(size=2, color='red'),
        name='Debris'
    )
    fig = go.Figure(data=[earth_surf, debris_trace])
    fig.update_layout(
        title="Global Debris Map (Interactive)",
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data'
        )
    )
    return fig

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="Debris & Satellite Simulation (Plotly)", layout="wide")
st.title("🚀 Autonomous Debris Removal & Satellite Launch (Interactive Plotly)")

# Use session_state to store data
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

# Buttons
if st.sidebar.button("Fetch Debris TLE"):
    st.write("**[API] Fetching debris TLE data...**")
    debris = fetch_debris_tle()
    if debris:
        st.session_state.debris_list = assign_debris_size_and_criticality(debris)
        st.write(f"**[API] Fetched {len(st.session_state.debris_list)} debris objects.**")
        for d in st.session_state.debris_list:
            st.write(f"* {d['name']}: size={d['size']:.2f} m, mass={d['mass']:.2f} kg, crit={d['criticality']:.2f}")
    else:
        st.error("[API] No debris TLE data fetched.")

if st.sidebar.button("Simulate Debris Mission"):
    if st.session_state.debris_list is None:
        st.error("No debris data. Please fetch debris TLE first!")
    else:
        st.write("**[MISSION] Simulating debris-collector mission from ISS...**")
        iss_orb, deb_orb, dep_t, arr_t = simulate_mission(st.session_state.debris_list, log_func=st.write)
        st.session_state.iss_orbit = iss_orb
        st.session_state.debris_orbit = deb_orb
        st.session_state.dep_time = dep_t
        st.session_state.arr_time = arr_t
        # Use Plotly figure
        fig = plot_mission_3d_interactive(iss_orb, deb_orb, dep_t, arr_t)
        st.plotly_chart(fig, use_container_width=True)

if st.sidebar.button("Simulate Satellite Launch"):
    try:
        lat_val = float(lat_input)
        lon_val = float(lon_input)
        alt_val = float(alt_input)
    except:
        st.error("Invalid lat/lon/alt.")
    else:
        st.write("**[SATELLITE] Simulating satellite launch...**")
        sat_orb = simulate_satellite_launch(lat_val, lon_val, alt_val, log_func=st.write)
        st.session_state.sat_orbit = sat_orb
        fig_sat = plot_satellite_orbit_3d_interactive(sat_orb)
        st.plotly_chart(fig_sat, use_container_width=True)

if st.sidebar.button("Plot Global Debris Map"):
    if st.session_state.debris_list is None:
        st.error("No debris data. Please fetch debris TLE first!")
    else:
        tle_list = [(d["name"], d["line1"], d["line2"]) for d in st.session_state.debris_list if d["line1"] and d["line2"]]
        positions = propagate_tles(tle_list)
        st.write(f"Propagated {len(positions)} debris objects to current time.")
        fig_globe = plot_debris_map_3d_plotly(positions)
        st.plotly_chart(fig_globe, use_container_width=True)

st.markdown("### Instructions")
st.markdown("""
1. **Fetch Debris TLE**: Retrieves debris data and assigns size/mass.
2. **Simulate Debris Mission**: Computes the ISS→debris transfer trajectory and displays an **interactive 3D** Plotly figure you can rotate.
3. **Simulate Satellite Launch**: Creates a naive orbit from lat/lon/alt, also displayed in a 3D Plotly figure.
4. **Plot Global Debris Map**: 3D Plotly globe with debris points.
5. **Rotate**: Click and drag in each figure to rotate/zoom in real time.
""")
