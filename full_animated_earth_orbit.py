import requests
import numpy as np
import random
import time
from datetime import datetime
from sgp4.api import Satrec, jday
import matplotlib.pyplot as plt

# ---------------------------
# Data Structures and Classes
# ---------------------------
class Debris:
    """
    Represents a space debris object with its TLE data.
    A simulated "size" is assigned (as a proxy for its collision risk).
    """
    def __init__(self, name, tle_line1, tle_line2):
        self.name = name
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        self.satrec = Satrec.twoline2rv(tle_line1, tle_line2)
        self.size = None            # To be set by the AI (in arbitrary units)
        self.criticality_score = None

    def propagate(self, dt_minutes):
        """
        Propagates the orbit using SGP4 for a given dt (in minutes).
        Returns position and velocity (in km and km/s).
        """
        now = datetime.utcnow()
        jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
        error_code, pos, vel = self.satrec.sgp4(jd, fr)
        return pos, vel


class TrajectoryOptimizer:
    """
    AI agent to evaluate debris criticality.
    Given a launch point (e.g. Cape Canaveral) and a list of debris,
    it calculates a criticality score for each debris object based on its size and estimated proximity.
    """
    def __init__(self, launch_coords):
        # launch_coords as a tuple: (latitude, longitude)
        self.launch_coords = launch_coords

    def compute_criticality(self, debris, launch_coords):
        # For simulation: assign a random current location to the debris.
        debris_location = (random.uniform(-90, 90), random.uniform(-180, 180))
        lat1, lon1 = launch_coords
        lat2, lon2 = debris_location
        # Simple (Euclidean) distance (note: for demonstration only)
        distance = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
        # If size not set, assign a random size (scale 1 to 10)
        if debris.size is None:
            debris.size = random.uniform(1, 10)
        # Higher score means higher risk: large debris closer to the launch path are prioritized.
        score = debris.size / (distance + 1)  # add 1 to avoid division by zero
        return score, debris_location

    def identify_critical_debris(self, debris_list):
        prioritized = []
        for d in debris_list:
            score, location = self.compute_criticality(d, self.launch_coords)
            d.criticality_score = score
            prioritized.append((d, score, location))
        # Sort debris in descending order (most critical first)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        return prioritized


class NavigationAgent:
    """
    Plans a simple trajectory as a series of waypoints from a start to a target.
    In a real-world system, this would use orbital mechanics and collision avoidance algorithms.
    """
    def plan_trajectory(self, start_coords, target_coords):
        # Both coordinates are tuples: (latitude, longitude, altitude)
        lat1, lon1, alt1 = start_coords
        lat2, lon2, alt2 = target_coords
        waypoints = []
        steps = 10
        for i in range(steps + 1):
            lat = lat1 + (lat2 - lat1) * i / steps
            lon = lon1 + (lon2 - lon1) * i / steps
            alt = alt1 + (alt2 - alt1) * i / steps
            waypoints.append((lat, lon, alt))
        return waypoints


class ExecutionAgent:
    """
    Controls the physical dispatch of the debris collector.
    Simulates movement along the planned waypoints.
    """
    def execute_mission(self, trajectory):
        for wp in trajectory:
            print(f"Debris collector moving to waypoint: {wp}")
            time.sleep(0.1)  # Simulate transit delay
        print("Debris collector reached target. Commencing debris collection...")
        time.sleep(1)  # Simulate debris collection process
        print("Debris collected. Initiating return trajectory...")
        time.sleep(1)
        print("Debris collector returned to ISS successfully.")


class CommunicationAgent:
    """
    Handles mission logging and status communications.
    """
    def send_message(self, message):
        print(f"[COMMUNICATION] {message}")


# ---------------------------
# Helper Functions
# ---------------------------
def fetch_tle_data(url):
    """
    Fetches TLE data from a given URL.
    Expects at least 3 lines: name, line1, line2.
    """
    response = requests.get(url)
    lines = response.text.strip().splitlines()
    if len(lines) >= 3:
        return lines[0], lines[1], lines[2]
    return None, None, None


def fetch_debris_list(url, num_objects=5):
    """
    Fetches debris TLEs from a given URL.
    Returns a list of Debris objects; defaults to a few objects for simulation.
    """
    response = requests.get(url)
    lines = response.text.strip().splitlines()
    debris_objects = []
    # Each debris object is represented by 3 lines (name + 2 TLE lines)
    for i in range(0, min(num_objects * 3, len(lines)), 3):
        if i + 2 < len(lines):
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
            debris_objects.append(Debris(name, line1, line2))
    return debris_objects


def visualize_trajectory(traj_to, traj_return):
    """
    Plots a simple 2D visualization (latitude vs. longitude) of the outbound and return trajectories.
    """
    to_lats = [wp[0] for wp in traj_to]
    to_lons = [wp[1] for wp in traj_to]
    ret_lats = [wp[0] for wp in traj_return]
    ret_lons = [wp[1] for wp in traj_return]

    plt.figure(figsize=(8, 6))
    plt.plot(to_lons, to_lats, 'bo-', label="To Target")
    plt.plot(ret_lons, ret_lats, 'ro-', label="Return")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.title("Simulated Trajectory of Debris Collector")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------
# Main System Flow
# ---------------------------
def main():
    # Initialize Communication
    comms = CommunicationAgent()
    comms.send_message("Mission initiated: Autonomous Debris Removal System online.")

    # Step 1: Retrieve ISS TLE data (starting point for the debris collector)
    iss_tle_url = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE"
    iss_name, iss_line1, iss_line2 = fetch_tle_data(iss_tle_url)
    comms.send_message(f"ISS TLE data fetched: {iss_name}")
    # For simulation, set ISS coordinates (latitude, longitude, altitude in km)
    iss_coords = (0.0, 0.0, 408)  # Example: 408 km altitude

    # Step 2: Fetch debris data from Celestrak
    debris_tle_url = "https://celestrak.org/NORAD/elements/iridium-33-debris.txt"
    debris_list = fetch_debris_list(debris_tle_url, num_objects=5)
    comms.send_message(f"Fetched {len(debris_list)} debris objects from Celestrak.")

    # Step 3: Run the AI agent to prioritize debris based on a criticality score.
    # Assume a launch mission from Cape Canaveral (28.5° N, -80.6° W)
    launch_coords = (28.5, -80.6)
    trajectory_optimizer = TrajectoryOptimizer(launch_coords)
    prioritized_debris = trajectory_optimizer.identify_critical_debris(debris_list)

    comms.send_message("Prioritizing debris based on criticality score...")
    for d, score, location in prioritized_debris:
        comms.send_message(
            f"Debris: {d.name}, Size: {d.size:.2f}, Score: {score:.2f}, Estimated Location: {location}"
        )

    # Choose the highest-priority debris object as the target.
    target_debris, target_score, debris_location = prioritized_debris[0]
    comms.send_message(
        f"Selected target debris: {target_debris.name} with criticality score: {target_score:.2f}"
    )

    # For simulation, assume the debris altitude is 800 km.
    target_coords = (debris_location[0], debris_location[1], 800)

    # Step 4: Plan the trajectory from ISS to the target debris and back.
    nav_agent = NavigationAgent()
    trajectory_to_target = nav_agent.plan_trajectory(iss_coords, target_coords)
    trajectory_return = nav_agent.plan_trajectory(target_coords, iss_coords)
    comms.send_message("Trajectory planned from ISS to target debris and back.")

    # Step 5: Dispatch the debris collector along the planned trajectories.
    exec_agent = ExecutionAgent()
    comms.send_message("Dispatching debris collector to target debris...")
    exec_agent.execute_mission(trajectory_to_target)
    exec_agent.execute_mission(trajectory_return)
    comms.send_message("Mission completed: Debris collected and returned to ISS.")

    # Optional: Visualize the planned trajectories.
    visualize_trajectory(trajectory_to_target, trajectory_return)


if __name__ == '__main__':
    main()
