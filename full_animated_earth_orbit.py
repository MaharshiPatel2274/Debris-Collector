import streamlit as st
from datetime import datetime
import os
import math
import random
import numpy as np
import requests, json

# Poliastro / Astropy imports
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.iod import izzo

# SGP4 imports
from sgp4.api import Satrec, jday

# Plotly for interactive 3D figures
import plotly.graph_objects as go

# -------------------------------------------------------------------
# 1. Streamlit Page Config
# -------------------------------------------------------------------
st.set_page_config(page_title="Debris & Satellite Simulation (Interactive)", layout="wide")

# -------------------------------------------------------------------
# 2. Main Header Title (Centered)
# -------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align: left;'> Autonomous Debris Removal & Satellite Launch </h1>", 
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# 3. Sidebar: Logo at Top (Medium Size) and Satellite Launch Parameters
# -------------------------------------------------------------------
# Make sure logo_main.png is in the same directory as this script
logo_path = os.path.join(os.path.dirname(__file__), "logo_main.png")

# Remove deprecated use_column_width and specify a fixed width for medium size
st.sidebar.image(logo_path, width=270)

st.sidebar.header("Satellite Launch Parameters")
lat_input = st.sidebar.text_input("Launch Latitude (°)", "28.5")
lon_input = st.sidebar.text_input("Launch Longitude (°)", "-80.6")
alt_input = st.sidebar.text_input("Launch Altitude (km)", "500")

# -------------------------------------------------------------------
# 4. Global URLs and Constants
# -------------------------------------------------------------------
DEBRIS_TLE_URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?CATNR=25544&FORMAT=TLE"
ISS_TLE_URL    = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE"
MAX_OBJECTS = 500

USE_TEXTURED_EARTH = True
EARTH_TEXTURE_FILE = "earth_texture.jpg"  # Replace if needed

# -------------------------------------------------------------------
# 5. Helper Functions
# -------------------------------------------------------------------
def ai_detect_debris_size(image_path):
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
        jd, fr = jday(
            epoch.datetime.year, epoch.datetime.month, epoch.datetime.day,
            epoch.datetime.hour, epoch.datetime.minute,
            epoch.datetime.second + epoch.datetime.microsecond*1e-6
        )
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            st.warning(f"[WARN] sgp4 error for {debris['name']}: code {e}")
            raise ValueError("Propagation error")
        r = np.array(r) * u.km
        v = np.array(v) * (u.km / u.s)
        orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch)
        return orbit
    except Exception as ex:
        st.error(f"[ERROR] Creating orbit for {debris['name']}: {ex}")
        return None

def compute_transfer_orbit(iss_orbit, debris_orbit, departure_time, arrival_time):
    r1 = iss_orbit.propagate(departure_time - iss_orbit.epoch).r
    r2 = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r
    (v1, v2), = izzo.lambert(Earth.k, r1, r2, (arrival_time - departure_time).to(u.s))
    return v1, v2, r1, r2

def simulate_mission(debris_list, log_func=st.write):
    epoch = Time(datetime.utcnow())
    # Approximate ISS orbit
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
    # Build orbits for each piece of debris
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
    log_func(f"**[MISSION] Selected target debris:** {target['name']} (Size: {target['size']:.2f} m, Mass: {target['mass']:.2f} kg)")

    departure_time = epoch + 10*u.min
    arrival_time = departure_time + 40*u.min

    v_dep, v_arr, r_dep, r_arr = compute_transfer_orbit(iss_orbit, target["orbit"], departure_time, arrival_time)
    ttime = (arrival_time - departure_time).to(u.s).value

    iss_v = iss_orbit.v.to(u.km/u.s).value
    dep_dv = np.linalg.norm(v_dep.to(u.km/u.s).value - iss_v)
    debris_v = target["orbit"].v.to(u.km/u.s).value
    arr_dv = np.linalg.norm(debris_v - v_arr.to(u.km/u.s).value)
    total_dv = dep_dv + arr_dv

    log_func(f"**[MISSION] Transfer orbit computed; duration = {ttime:.2f} s**")
    log_func(f"**[MISSION] Delta-v:** Dep = {dep_dv:.2f} km/s, Arr = {arr_dv:.2f} km/s, Total = {total_dv:.2f} km/s")
    log_func("**[MISSION] Debris collected & crushed.**")
    log_func(f"**[RESULT] Mission complete. Δv={total_dv:.2f} km/s, time={ttime:.2f} s**")

    return iss_orbit, target["orbit"], departure_time, arrival_time

def simulate_satellite_launch(lat_deg, lon_deg, alt_km, log_func=st.write):
    epoch = Time(datetime.utcnow())
    inc = abs(lat_deg)*u.deg
    if inc > 90*u.deg:
        inc = 90*u.deg
    sat_orbit = Orbit.from_classical(
        attractor=Earth,
        a=(Earth.R + alt_km*u.km),
        ecc=0.001*u.one,
        inc=inc,
        raan=0*u.deg,
        argp=0*u.deg,
        nu=0*u.deg,
        epoch=epoch
    )
    log_func(f"**[SATELLITE] Launched from Earth (lat: {lat_deg}°, lon: {lon_deg}°, alt: {alt_km} km, inc: {inc})**")
    return sat_orbit

# -------------------------------------------------------------------
# 6. Plotly 3D Interactive Visualization
# -------------------------------------------------------------------
def plot_mission_3d_interactive(iss_orbit, debris_orbit, departure_time, arrival_time):
    R = Earth.R.to(u.km).value
    n_u, n_v = 50, 50
    u_vals = np.linspace(0, 2*np.pi, n_u)
    v_vals = np.linspace(0, np.pi, n_v)
    X = R * np.outer(np.cos(u_vals), np.sin(v_vals))
    Y = R * np.outer(np.sin(u_vals), np.sin(v_vals))
    Z = R * np.outer(np.ones_like(u_vals), np.cos(v_vals))

    earth_surface = go.Surface(
        x=X, y=Y, z=Z,
        colorscale="Blues",
        opacity=0.3,
        name="Earth"
    )

    # ISS orbit
    iss_pts = iss_orbit.sample(200)
    iss_trace = go.Scatter3d(
        x=iss_pts.x.to(u.km).value,
        y=iss_pts.y.to(u.km).value,
        z=iss_pts.z.to(u.km).value,
        mode="lines",
        line=dict(color="green", width=3),
        name="ISS Orbit"
    )

    # Debris orbit
    debris_pts = debris_orbit.sample(200)
    debris_trace = go.Scatter3d(
        x=debris_pts.x.to(u.km).value,
        y=debris_pts.y.to(u.km).value,
        z=debris_pts.z.to(u.km).value,
        mode="lines",
        line=dict(color="red", width=3),
        name="Debris Orbit"
    )

    # Transfer trajectory (linear interpolation)
    t_vals = np.linspace(0, (arrival_time - departure_time).to(u.s).value, 100)
    r_dep = iss_orbit.propagate(departure_time - iss_orbit.epoch).r.value
    r_arr = debris_orbit.propagate(arrival_time - debris_orbit.epoch).r.value
    transfer_xyz = []
    for t in t_vals:
        frac = t / t_vals[-1]
        r = r_dep + frac * (r_arr - r_dep)
        transfer_xyz.append(r)
    transfer_xyz = np.array(transfer_xyz)
    transfer_trace = go.Scatter3d(
        x=transfer_xyz[:,0],
        y=transfer_xyz[:,1],
        z=transfer_xyz[:,2],
        mode="lines",
        line=dict(color="orange", width=3, dash="dot"),
        name="Transfer Trajectory"
    )

    fig = go.Figure(data=[earth_surface, iss_trace, debris_trace, transfer_trace])
    fig.update_layout(
        title="Debris Collector Mission ",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data"
        )
    )
    return fig

def plot_satellite_orbit_3d_interactive(sat_orbit):
    R = Earth.R.to(u.km).value
    n_u, n_v = 50, 50
    u_vals = np.linspace(0, 2*np.pi, n_u)
    v_vals = np.linspace(0, np.pi, n_v)
    X = R * np.outer(np.cos(u_vals), np.sin(v_vals))
    Y = R * np.outer(np.sin(u_vals), np.sin(v_vals))
    Z = R * np.outer(np.ones_like(u_vals), np.cos(v_vals))

    earth_surface = go.Surface(
        x=X, y=Y, z=Z,
        colorscale="Blues",
        opacity=0.3,
        name="Earth"
    )
    sat_pts = sat_orbit.sample(200)
    sat_trace = go.Scatter3d(
        x=sat_pts.x.to(u.km).value,
        y=sat_pts.y.to(u.km).value,
        z=sat_pts.z.to(u.km).value,
        mode="lines",
        line=dict(color="magenta", width=3),
        name="Satellite Orbit"
    )

    fig = go.Figure(data=[earth_surface, sat_trace])
    fig.update_layout(
        title="Satellite Orbit ",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data"
        )
    )
    return fig

def propagate_tles(tle_list):
    now = datetime.utcnow()
    jd_now, fr_now = jday(
        now.year, now.month, now.day,
        now.hour, now.minute,
        now.second + now.microsecond*1e-6
    )
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

def plot_debris_map_3d_plotly(debris_positions):
    R = 6378.0
    n_u, n_v = 50, 50
    u_vals = np.linspace(0, 2*np.pi, n_u)
    v_vals = np.linspace(-np.pi/2, np.pi/2, n_v)
    X = R * np.outer(np.cos(u_vals), np.cos(v_vals))
    Y = R * np.outer(np.sin(u_vals), np.cos(v_vals))
    Z = R * np.outer(np.ones_like(u_vals), np.sin(v_vals))

    earth_surf = go.Surface(
        x=X, y=Y, z=Z,
        colorscale="Blues",
        opacity=0.3,
        name="Earth"
    )
    xs = [p[1] for p in debris_positions]
    ys = [p[2] for p in debris_positions]
    zs = [p[3] for p in debris_positions]

    debris_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=2, color="red"),
        name="Debris"
    )
    fig = go.Figure(data=[earth_surf, debris_trace])
    fig.update_layout(
        title="Global Debris Map ",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data"
        )
    )
    return fig

# -------------------------------------------------------------------
# 7. Main UI: Buttons and Plotting
# -------------------------------------------------------------------
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

if st.sidebar.button("Fetch Debris TLE"):
    st.markdown("**[API] Fetching debris TLE data...**")
    debris = fetch_debris_tle()
    if debris:
        st.session_state.debris_list = assign_debris_size_and_criticality(debris)
        st.markdown(f"**[API] Fetched {len(st.session_state.debris_list)} debris objects.**")
        for d in st.session_state.debris_list:
            st.write(f"* {d['name']}: size={d['size']:.2f} m, mass={d['mass']:.2f} kg, crit={d['criticality']:.2f}")
    else:
        st.error("[API] No debris TLE data fetched.")

if st.sidebar.button("Simulate Debris Mission"):
    if st.session_state.debris_list is None:
        st.error("No debris data. Please fetch debris TLE first!")
    else:
        st.markdown("**[MISSION] Simulating debris-collector mission from ISS...**")
        iss_orbit, debris_orbit, dep_time, arr_time = simulate_mission(
            st.session_state.debris_list, 
            log_func=st.markdown
        )
        st.session_state.iss_orbit = iss_orbit
        st.session_state.debris_orbit = debris_orbit
        st.session_state.dep_time = dep_time
        st.session_state.arr_time = arr_time

        fig = plot_mission_3d_interactive(iss_orbit, debris_orbit, dep_time, arr_time)
        st.plotly_chart(fig, use_container_width=True)

if st.sidebar.button("Simulate Satellite Launch"):
    try:
        lat_val = float(lat_input)
        lon_val = float(lon_input)
        alt_val = float(alt_input)
    except Exception:
        st.error("Invalid input for latitude, longitude, or altitude.")
    else:
        st.markdown("**[SATELLITE] Simulating satellite launch...**")
        sat_orbit = simulate_satellite_launch(lat_val, lon_val, alt_val, log_func=st.markdown)
        st.session_state.sat_orbit = sat_orbit

        fig_sat = plot_satellite_orbit_3d_interactive(sat_orbit)
        st.plotly_chart(fig_sat, use_container_width=True)

if st.sidebar.button("Plot Global Debris Map"):
    if st.session_state.debris_list is None:
        st.error("No debris data. Please fetch debris TLE first!")
    else:
        tle_list = [
            (d["name"], d["line1"], d["line2"]) 
            for d in st.session_state.debris_list 
            if d["line1"] and d["line2"]
        ]
        positions = propagate_tles(tle_list)
        st.markdown(f"**[INFO] Propagated {len(positions)} debris objects to current time.**")
        fig_deb = plot_debris_map_3d_plotly(positions)
        st.plotly_chart(fig_deb, use_container_width=True)

# -------------------------------------------------------------------
# 8. Instructions / Footer
# -------------------------------------------------------------------
st.markdown("### Instructions")
st.markdown("""
1. **Fetch Debris TLE**: Retrieves debris data and assigns size/mass.  
2. **Simulate Debris Mission**: Computes the ISS→debris transfer trajectory and displays an interactive 3D Plotly figure you can rotate.  
3. **Simulate Satellite Launch**: Generates a satellite orbit based on your launch parameters and displays it interactively.  
4. **Plot Global Debris Map**: Shows a 3D interactive globe with debris scatter points.  
5. **Rotate**: Use your mouse to click and drag in each figure to rotate/zoom in real time.
""")
