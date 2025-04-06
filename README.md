# 🌍 COSMiQ: AI-Powered Orbital Debris Collector

**COSMiQ** is a real-time simulation platform built to visualize autonomous orbital debris collection missions. It uses real satellite tracking data and physics-based models to compute ISS-to-debris transfer trajectories, simulate satellite launches, and map global debris in 3D—all through an interactive web interface.

🏆 **Winner at DevHacks 2025 Hackathon!**

---

## 🚀 Live Features

- Fetches **live TLE data** from [CelesTrak](https://celestrak.org/)
- Simulates debris orbits and ISS-to-debris **Lambert transfers**
- Generates **autonomous satellite launch orbits**
- Interactive **3D Plotly visualizations** of orbits and trajectories
- Transfer path toggle, orbit color legend, and clean sidebar UI
- Fully built in **Python using Streamlit**

---

## 🛠 Tech Stack

- **Python 3.8+**
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)
- [Poliastro](https://docs.poliastro.space/)
- [SGP4](https://pypi.org/project/sgp4/)
- [Astropy](https://www.astropy.org/)
- CelesTrak TLE APIs

---

## 🧑‍💻 Getting Started

### 🔧 Prerequisites

- Python 3.8 or above
- pip package manager

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cosmiq-debris-collector.git
cd cosmiq-debris-collector

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Running Locally

Start the app with Streamlit:

```bash
streamlit run full_animated_earth_orbit.py
```

Once the server starts, open your browser to:\
📍 `http://localhost:8501`

---

## 📖 Usage Guide

1. **Sidebar Configuration**

   - Upload your logo (optional)
   - Enter launch latitude, longitude, and altitude
   - Click "Fetch Debris TLE" to pull live orbital data

2. **Mission Simulation**

   - Click "Simulate Debris Mission" to compute an optimal transfer from ISS to the most critical debris
   - Toggle the transfer trajectory line visibility with the button in the 3D plot

3. **Satellite Launch**

   - Simulate a new satellite based on custom launch site and altitude

4. **Debris Map**

   - Visualize all tracked debris in a global 3D scatter plot

5. **Rotate & Zoom**

   - All plots are interactive—click and drag to orbit, scroll to zoom

---

## 📁 Folder Structure

```
cosmiq-debris-collector/
├── debris_collector.py   # Main Streamlit app
├── logo_main.png                  # UI branding
├── earth_texture.jpg              # Optional Earth texture
├── requirements.txt               # Python dependencies
├── README.md                      # This file

```

---

## 👨‍💻 Team & Credits

Built by:

- Maharshi Niraj Patel (Backend + UI/UX + Simulation Logic)

Special thanks to **DevHacks 2025** for hosting an incredible space-tech hackathon!

---

## 📜 License

This project is for educational and research use.\
Please attribute authors if using or modifying.

---

## 🌐 Demo

Want to try it?\
Clone the repo → Run locally → Simulate space missions in your browser.

```bash
streamlit run full_animated_earth_orbit.py
```

---

