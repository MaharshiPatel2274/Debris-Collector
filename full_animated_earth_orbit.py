import requests
import random
import numpy as np
import math
from sgp4.api import Satrec, jday
import matplotlib.pyplot as plt

# ------------------------------
# 1. Data Retrieval from Celestrak
# ------------------------------
def fetch_debris_tle():
    """
    Fetch TLE data for debris objects from Celestrak.
    In a real implementation, this URL would point to a debris TLE file.
    Here we use a placeholder URL. If the API call fails, we return an empty list.
    """
    url = "https://celestrak.com/NORAD/elements/debris.txt"  # placeholder URL
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        tle_data = response.text.splitlines()
        debris_list = []
        # TLE format: 3 lines per object (name, line1, line2)
        for i in range(0, len(tle_data), 3):
            if i + 2 < len(tle_data):
                name = tle_data[i].strip()
                line1 = tle_data[i+1].strip()
                line2 = tle_data[i+2].strip()
                debris_list.append({
                    "name": name,
                    "line1": line1,
                    "line2": line2
                })
        return debris_list
    except Exception as e:
        print(f"Error fetching TLE data: {e}")
        return []  # fallback to simulated debris

# ------------------------------
# 2. AI-based Debris Size Detection
# ------------------------------
def detect_debris_size(debris):
    """
    Simulate an AI size-detection module.
    In a real system, this function would process sensor/image data.
    Here we simulate by assigning a random size (in meters) to the debris.
    """
    size = random.uniform(0.1, 5.0)  # size between 0.1 and 5.0 meters
    return size

def compute_criticality(debris):
    """
    Compute a criticality score for debris.
    For this simulation, we assume that larger debris are more critical.
    You could add other factors (e.g., trajectory intersections, collision risk).
    """
    size = debris.get("size", 0)
    # Example: criticality could be directly proportional to size.
    return size

# ------------------------------
# 3. Route Planning & Navigation
# ------------------------------
def plan_route(start_pos, target_pos):
    """
    Plan a route from the start position to the target position.
    For simplicity, we calculate the Euclidean distance.
    In real orbital mechanics, you would use Hohmann transfers or solve Lambert's problem.
    
    Returns:
      travel_time (seconds): Estimated travel time given a nominal speed.
      distance (km): Euclidean distance between start and target.
    """
    start = np.array(start_pos)
    target = np.array(target_pos)
    distance = np.linalg.norm(start - target)
    # Assume a nominal travel speed in LEO (e.g., ~7.66 km/s)
    speed = 7.66  # km/s
    travel_time = distance / speed
    return travel_time, distance

# ------------------------------
# 4. Debris Collector Execution
# ------------------------------
def dispatch_collector(debris, iss_position, debris_position):
    """
    Simulate dispatching the debris collector:
      - Calculate route from ISS to debris and back.
      - Simulate travel times and distances.
      - Invoke the rock crusher (simulate crushing debris to dust).
    """
    outbound_time, outbound_distance = plan_route(iss_position, debris_position)
    return_time, return_distance = plan_route(debris_position, iss_position)
    total_time = outbound_time + return_time

    print(f"\nDispatching debris collector to target: {debris['name']}")
    print(f"Outbound: {outbound_distance:.2f} km in {outbound_time:.2f} seconds")
    print(f"Return: {return_distance:.2f} km in {return_time:.2f} seconds")
    print(f"Total mission duration: {total_time:.2f} seconds")
    
    # Simulate debris capture and crushing process
    print(f"Collecting debris {debris['name']}...")
    print("Initiating rock crusher on ISS...")
    print(f"Debris {debris['name']} crushed to dust particles.\n")
    
    return total_time

# ------------------------------
# 5. Main Mission Logic
# ------------------------------
def main():
    # Step 1: Fetch debris data (TLEs) from Celestrak
    debris_list = fetch_debris_tle()
    if not debris_list:
        print("No debris data fetched. Using simulated debris objects.")
        # Generate simulated debris objects if API call fails
        debris_list = []
        for i in range(5):
            debris_list.append({
                "name": f"Simulated_Debris_{i+1}",
                "line1": "",
                "line2": ""
            })
    
    # Step 2: AI-based size detection and criticality scoring
    for debris in debris_list:
        debris["size"] = detect_debris_size(debris)
        debris["criticality"] = compute_criticality(debris)
    
    # Sort debris objects by descending criticality (largest first)
    debris_list.sort(key=lambda d: d["criticality"], reverse=True)
    
    print("Detected debris objects (size in meters, criticality score):")
    for debris in debris_list:
        print(f"  {debris['name']}: Size = {debris['size']:.2f} m, Criticality = {debris['criticality']:.2f}")
    
    # Step 3: Simulate positions for ISS and each debris object.
    # For simplicity, we assume the ISS is at a fixed position.
    # In practice, you would use ISS TLE data and propagate its orbit.
    iss_position = [0, 0, 400]  # (x, y, z) in km (roughly 400 km altitude)
    
    # Assign simulated positions to debris objects (random positions in LEO)
    for debris in debris_list:
        debris["position"] = [
            random.uniform(-7000, 7000),
            random.uniform(-7000, 7000),
            random.uniform(-7000, 7000)
        ]
    
    # Step 4: AI selects the top-priority debris to remove
    target_debris = debris_list[0]
    print(f"\nSelected target debris: {target_debris['name']} with size {target_debris['size']:.2f} m")
    
    # Step 5: Dispatch the debris collector mission
    mission_time = dispatch_collector(target_debris, iss_position, target_debris["position"])
    
    # (Optional) Visualization of the mission path (ISS -> Debris -> ISS)
    iss = np.array(iss_position)
    target = np.array(target_debris["position"])
    path = np.vstack([iss, target, iss])
    
    plt.figure(figsize=(6, 5))
    plt.plot(path[:, 0], path[:, 1], marker='o', linestyle='--')
    plt.title("Simulated Debris Collector Route")
    plt.xlabel("X Position (km)")
    plt.ylabel("Y Position (km)")
    plt.grid(True)
    plt.show()
    
    print("Mission simulation complete.")

# ------------------------------
# Run the Simulation
# ------------------------------
if __name__ == '__main__':
    main()
