#start for the development code here
!pip install astroquery
!pip install gradio
print('done installing')
#now the actual code
import numpy as np
import pandas as pd
import datetime
from astroquery.jplhorizons import Horizons
import plotly.graph_objects as go
import torch
import torch.nn as nn
import gradio as gr
import os

#######################################
# 1. Constants & Unit Conversions
#######################################
GM_sun = 1.32712440018e20     # Sun's GM [m^3/s^2]
AU_TO_M = 1.496e11            # 1 AU in meters
DAY_TO_S = 86400.0            # Seconds per day

# Normalized units: positions in AU, velocities in AU/day, time in days.
#GM_sun_norm = GM_sun / (AU_TO_M*3) * (DAY_TO_S*2)
# Although not used in this Euler-based simulation, we define G_norm for planetary terms:
#G_norm = (6.67430e-11) / (AU_TO_M*3) * (DAY_TO_S*2)

#changed code on 30-05-2025 by vishu -> issue : straight line for orbit
# Sun’s gravitational parameter in AU^3/day^2
GM_sun_norm = GM_sun / (AU_TO_M**3) * (DAY_TO_S**2)

# Universal G in AU^3/(kg·day^2)
G_norm      = 6.67430e-11 / (AU_TO_M**3) * (DAY_TO_S**2)
##till here changes only rest at bottom is orignal

# Additional force constants (normalized)
k_SRP_norm = 4.56e-6 * (DAY_TO_S**2) / AU_TO_M   # Solar radiation pressure [AU/day^2]
k_drag_norm = 1e-3  # Atmospheric drag constant (placeholder)

#######################################
# 2. External Planet Data (Orbital Elements & IDs)
#######################################
all_planet_ids = {
    "mercury": 199,
    "venus":   299,
    "earth":   399,
    "mars":    499,
    "jupiter": 599,
    "saturn":  699,
    "uranus":  799,
    "neptune": 899
}
planet_masses = {
    "mercury": 3.3011e23,
    "venus":   4.8675e24,
    "earth":   5.972e24,
    "mars":    6.4171e23,
    "jupiter": 1.8982e27,
    "saturn":  5.6834e26,
    "uranus":  8.6810e25,
    "neptune": 1.02409e26
}
planet_list = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]

#######################################
# 3. Keplerian-to-Cartesian Conversion (Normalized)
#######################################
def orbital_elements_to_state(a, e, i_deg, w_deg, Omega_deg, M_deg, t):
    """
    Converts orbital elements (a in AU, angles in degrees) and time offset t (in seconds)
    into a Cartesian state vector [x, y, z, vx, vy, vz] in normalized units.
    (Positions in AU, velocities in AU/day)
    """
    a_m = a * AU_TO_M
    i = np.radians(i_deg)
    w = np.radians(w_deg)
    Omega = np.radians(Omega_deg)
    M0 = np.radians(M_deg)

    n = np.sqrt(GM_sun / a_m**3)
    M = M0 + n * t
    E = M
    for _ in range(10):
        E = M + e * np.sin(E)
    f = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r = a_m * (1 - e * np.cos(E))
    x_perifocal = r * np.cos(f)
    y_perifocal = r * np.sin(f)
    z_perifocal = 0.0
    h = np.sqrt(GM_sun * a_m * (1-e**2))
    vx_perifocal = - (GM_sun/h) * np.sin(E)
    vy_perifocal = (GM_sun/h) * np.sqrt(1-e**2) * np.cos(E)
    vz_perifocal = 0.0

    cos_O = np.cos(Omega)
    sin_O = np.sin(Omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_w = np.cos(w)
    sin_w = np.sin(w)

    R = np.array([
        [cos_O*cos_w - sin_O*sin_w*cos_i, -cos_O*sin_w - sin_O*cos_w*cos_i, sin_O*sin_i],
        [sin_O*cos_w + cos_O*sin_w*cos_i, -sin_O*sin_w + cos_O*cos_w*cos_i, -cos_O*sin_i],
        [sin_w*sin_i, cos_w*sin_i, cos_i]
    ])

    pos_SI = R @ np.array([x_perifocal, y_perifocal, z_perifocal])
    vel_SI = R @ np.array([vx_perifocal, vy_perifocal, vz_perifocal])

    pos_norm = pos_SI / AU_TO_M           # in AU
    vel_norm = vel_SI * DAY_TO_S / AU_TO_M  # in AU/day
    return np.hstack((pos_norm, vel_norm))

#######################################
# 4. Dynamic Fetching of Planet Positions via Horizons
#######################################
def date_to_seconds_since(ref_date_str, date_str):
    fmt = "%Y-%m-%d %H:%M:%S"
    ref = datetime.datetime.strptime(ref_date_str, fmt)
    current = datetime.datetime.strptime(date_str, fmt)
    return (current - ref).total_seconds()

def fetch_planet_positions(planet_id, start_date, end_date, step):
    obj = Horizons(id=planet_id, location='@sun',
                   epochs={'start': start_date, 'stop': end_date, 'step': step})
    vectors = obj.vectors().to_pandas()
    vectors["x_m"] = vectors["x"].astype(float) * AU_TO_M
    vectors["y_m"] = vectors["y"].astype(float) * AU_TO_M
    vectors["z_m"] = vectors["z"].astype(float) * AU_TO_M
    times = []
    for ds in vectors["datetime_str"]:
        ds_clean = ds.replace("A.D. ", "").strip()
        dt_parsed = datetime.datetime.strptime(ds_clean, "%Y-%b-%d %H:%M:%S.%f")
        ds_formatted = dt_parsed.strftime("%Y-%m-%d %H:%M:%S")
        t_s = date_to_seconds_since(start_date, ds_formatted)
        times.append(t_s)
    vectors["time_s"] = times
    return vectors

#######################################
# 5. Get Target's Initial State and Orbital Elements Using its ID
#######################################
def get_target_state_and_elements(target_id, start_date_str):
    """
    Queries Horizons for the target object using its ID.
    Returns:
      - init_state: normalized state [x, y, z, vx, vy, vz] (positions in AU, velocities in AU/day)
      - elements: dictionary with keys "a", "e", "incl", "w", "Omega", "M"
    """
    stop_date = (datetime.datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S") +
                 datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    obj = Horizons(id=target_id, id_type="smallbody", location='@sun',
                   epochs={'start': start_date_str, 'stop': stop_date, 'step': '1h'})
    df_vec = obj.vectors().to_pandas()
    state_row = df_vec.iloc[0]
    # Get state in SI units then convert to normalized (positions in AU, velocities in AU/day)
    x = state_row["x"] * AU_TO_M
    y = state_row["y"] * AU_TO_M
    z = state_row["z"] * AU_TO_M
    vx = state_row["vx"] * AU_TO_M / DAY_TO_S
    vy = state_row["vy"] * AU_TO_M / DAY_TO_S
    vz = state_row["vz"] * AU_TO_M / DAY_TO_S
    state_SI = np.array([x, y, z, vx, vy, vz])
    state_norm = np.hstack((state_SI[:3] / AU_TO_M, state_SI[3:] * DAY_TO_S / AU_TO_M))

    df_elem = obj.elements().to_pandas()
    row_elem = df_elem.iloc[0]
    elements = {
        "a": float(row_elem["a"]),
        "e": float(row_elem["e"]),
        "incl": float(row_elem["incl"]),
        "w": float(row_elem["w"]),
        "Omega": float(row_elem["Omega"]),
        "M": float(row_elem["M"])
    }
    return state_norm, elements

#######################################
# 6. Compute External Planet Trajectories (Normalized to AU)
#######################################
def compute_planet_trajectory(planet_key, simulation_days, start_date_str):
    end_date_dt = datetime.datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(days=simulation_days)
    end_date_str = end_date_dt.strftime("%Y-%m-%d %H:%M:%S")
    df_planet = fetch_planet_positions(all_planet_ids[planet_key], start_date_str, end_date_str, step="1d")
    traj = df_planet[['x_m','y_m','z_m']].values  # in meters
    traj_AU = traj / AU_TO_M
    return torch.tensor(traj_AU, dtype=torch.float32)

def compute_all_planets_trajectories(simulation_days, start_date_str):
    trajs = []
    for p in planet_list:
        traj = compute_planet_trajectory(p, simulation_days, start_date_str)
        trajs.append(traj.unsqueeze(-1))
    trajs = torch.cat(trajs, dim=-1)  # Shape: (num_steps, 3, 8)
    return trajs

#######################################
# 6.1 Linear Interpolation for External Positions (Time in days)
#######################################
def interp_ext_positions(external_positions, t, dt_day):
    # external_positions: (batch, num_steps, 3, 8); t in days; dt_day typically 1 day
    batch_size, num_steps, _, _ = external_positions.shape
    i = int(t // dt_day)
    if i >= num_steps - 1:
        return external_positions[:, -1, :, :]
    t_i = i * dt_day
    t_ip1 = (i + 1) * dt_day
    ratio = (t - t_i) / (t_ip1 - t_i)
    return (1 - ratio) * external_positions[:, i, :, :] + ratio * external_positions[:, i+1, :, :]

#######################################
# 7. Euler Simulator (Physics-Based, Fixed Coefficients)
#######################################
class EulerSimulator(nn.Module):
    def __init__(self, dt, num_steps):
        """
        dt: time step in days (here dt = 1)
        num_steps: number of daily steps.
        Uses fixed coefficients (all = 1) for:
          - Sun gravity, planetary gravity (8 planets), SRP, and atmospheric drag.
        (Total: 11 coefficients)
        """
        super(EulerSimulator, self).__init__()
        self.dt = dt
        self.num_steps = num_steps
        self.fixed_coeffs = torch.ones(11, dtype=torch.float32)

    def acceleration(self, t, state, external_positions, dt_day, coeffs):
        eps = 1e-6
        x, y, z, vx, vy, vz = torch.unbind(state, dim=1)
        r_vec = torch.stack([x, y, z], dim=1)  # (batch, 3)
        r_norm = torch.norm(r_vec, dim=1, keepdim=True) + eps

        # Sun Gravity
        a_sun = - coeffs[0] * GM_sun_norm * r_vec / (r_norm**3)

        # Planetary Gravity
        interp_ext = interp_ext_positions(external_positions, t, dt_day)  # (batch, 3, 8)
        r_vec_exp = r_vec.unsqueeze(-1)  # (batch, 3, 1)
        d_vec = interp_ext - r_vec_exp   # (batch, 3, 8)
        d_norm = torch.norm(d_vec, dim=1, keepdim=True) + eps  # (batch, 1, 8)
        masses = torch.tensor([planet_masses[p] for p in planet_list],
                              dtype=torch.float32, device=state.device).view(1, 1, -1)
        coeffs_planets = coeffs[1:9].view(1, 1, -1)
        a_ext = coeffs_planets * G_norm * masses * d_vec / (d_norm**3)
        a_ext = torch.sum(a_ext, dim=2)  # (batch, 3)

        # Solar Radiation Pressure
        a_srp = coeffs[9] * (k_SRP_norm / (r_norm**2)) * (r_vec / r_norm)

        # Atmospheric Drag
        v_vec = torch.stack([vx, vy, vz], dim=1)
        v_norm = torch.norm(v_vec, dim=1, keepdim=True) + eps
        a_drag = - coeffs[10] * k_drag_norm * v_norm * (v_vec / v_norm)

        a_total = a_sun + a_ext + a_srp + a_drag
        derivative = torch.cat([v_vec, a_total], dim=1)  # (batch, 6)
        return derivative

    def forward(self, init_state, external_positions, coeffs=None):
        if coeffs is None:
            coeffs = self.fixed_coeffs.to(init_state.device)
        dt_day = self.dt
        state = init_state  # (batch, 6); already normalized
        traj_list = []
        t = 0.0  # time in days
        for step in range(self.num_steps):
            state = state + dt_day * self.acceleration(t, state, external_positions, dt_day, coeffs)
            t += dt_day
            traj_list.append(state[:, :3])
        traj = torch.stack(traj_list, dim=1)  # (batch, num_steps, 3)
        return traj

#######################################
# 8. Gradio Simulation Function
#######################################
def simulate_app(simulation_days=300, object_index=100, start_date_str="2025-01-01 00:00:00"):
    """
    Runs the simulation for a given target object.

    Parameters:
      simulation_days: Number of days to simulate.
      object_index: Row index in the CSV file of target objects.
      start_date_str: Start date string in "YYYY-MM-DD HH:MM:SS" format.

    Returns:
      Two Plotly figures (trajectory comparison and full solar system view) and text error metrics.
    """
    # Load target objects from CSV (if not available, use a dummy DataFrame)
    csv_filename = "/content/object_ids_and_names.csv"
    if os.path.exists(csv_filename):
        df_objects = pd.read_csv(csv_filename)
    else:
        df_objects = pd.DataFrame({"id": [1], "name": ["TestObject"]})

    object_index = int(object_index)
    if object_index >= len(df_objects):
        object_index = 0
    obj_row = df_objects.iloc[object_index]
    target_id = int(obj_row['id'])
    target_name = obj_row['name']
    print(f"Simulating object {target_name} (ID: {target_id})")

    # Get target's initial state and orbital elements from Horizons.
    try:
        init_state_np, elements = get_target_state_and_elements(target_id, start_date_str)
    except Exception as e:
        return None, None, f"Error fetching target data: {e}"

    init_state = torch.tensor(init_state_np, dtype=torch.float32)

    # Ground truth trajectory via classical orbital propagation:
    true_traj_list = []
    time_steps = np.arange(0, simulation_days+1, 1)  # in days; convert days to seconds for orbital_elements_to_state
    for t_day in time_steps:
        s = orbital_elements_to_state(elements["a"], elements["e"],
                                      elements["incl"], elements["w"],
                                      elements["Omega"], elements["M"], t_day * DAY_TO_S)
        true_traj_list.append(s[:3])
    true_traj = torch.tensor(np.stack(true_traj_list, axis=0), dtype=torch.float32)

    # Fetch external planetary trajectories (normalized to AU)
    external_trajs = {}
    end_date_dt = datetime.datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(days=simulation_days)
    end_date_str = end_date_dt.strftime("%Y-%m-%d %H:%M:%S")
    for p in planet_list:
        try:
            ext_traj = fetch_planet_positions(all_planet_ids[p], start_date_str, end_date_str, step="1d")
        except Exception as e:
            return None, None, f"Error fetching positions for {p}: {e}"
        traj_arr = ext_traj[['x_m', 'y_m', 'z_m']].values / AU_TO_M
        external_trajs[p] = torch.tensor(traj_arr, dtype=torch.float32)
    ext_list = [external_trajs[p].unsqueeze(-1) for p in planet_list]
    external_positions = torch.cat(ext_list, dim=-1)  # shape: (num_steps, 3, 8)

    # Prepare batch dimensions for target state and external trajectories.
    init_state = init_state.unsqueeze(0)       # (1, 6)
    batch_ext = external_positions.unsqueeze(0)  # (1, num_steps, 3, 8)

    # Instantiate the Euler Simulator (physics-based, fixed coefficients).
    num_steps = simulation_days + 1
    simulator = EulerSimulator(dt=1.0, num_steps=num_steps)
    fixed_coeffs = torch.ones(11, dtype=torch.float32)

    # Predict the target's trajectory.
    predicted_traj = simulator(init_state, batch_ext, fixed_coeffs)

    # Visualization 1: Compare predicted vs. ground-truth trajectories.
    fig1 = go.Figure()
    true_np = true_traj.cpu().numpy()
    pred_np = predicted_traj[0].cpu().numpy()
    fig1.add_trace(go.Scatter3d(
        x=true_np[:,0],
        y=true_np[:,1],
        z=true_np[:,2],
        mode='lines+markers',
        name='True Trajectory',
        line=dict(color='blue', width=4)
    ))
    fig1.add_trace(go.Scatter3d(
        x=pred_np[:,0],
        y=pred_np[:,1],
        z=pred_np[:,2],
        mode='lines+markers',
        name='Predicted Trajectory',
        line=dict(color='magenta', width=4)
    ))
    fig1.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        name='Sun',
        marker=dict(size=12, color='yellow')
    ))
    fig1.update_layout(
        title=f"Predicted vs. True Trajectory for {target_name} (in AU)",
        scene=dict(xaxis_title="X (AU)", yaxis_title="Y (AU)", zaxis_title="Z (AU)", aspectmode="data")
    )

    # Visualization 2: Full view of external planetary trajectories and target predicted trajectory.
    fig2 = go.Figure()
    colors = {
        "mercury": "gray",
        "venus": "orange",
        "earth": "blue",
        "mars": "red",
        "jupiter": "brown",
        "saturn": "goldenrod",
        "uranus": "lightblue",
        "neptune": "purple"
    }
    for p in planet_list:
        traj_p = external_trajs[p].cpu().numpy()  # (num_steps, 3)
        fig2.add_trace(go.Scatter3d(
            x=traj_p[:,0],
            y=traj_p[:,1],
            z=traj_p[:,2],
            mode='lines',
            name=f"{p.capitalize()} Actual",
            line=dict(color=colors.get(p, "black"), width=2)
        ))
    fig2.add_trace(go.Scatter3d(
        x=pred_np[:,0],
        y=pred_np[:,1],
        z=pred_np[:,2],
        mode='lines+markers',
        name=f"{target_name} Predicted",
        line=dict(color='magenta', width=4)
    ))
    fig2.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        name='Sun',
        marker=dict(size=12, color='yellow')
    ))
    fig2.update_layout(
        title="External Planets vs. Target Predicted Trajectory (in AU)",
        scene=dict(xaxis_title="X (AU)", yaxis_title="Y (AU)", zaxis_title="Z (AU)", aspectmode="data")
    )

    # Compute error metrics.
    mse = np.mean((true_np - pred_np)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_np - pred_np))
    err_str = f"Error Metrics (in AU):\nMSE = {mse:.4e}\nRMSE = {rmse:.4e}\nMAE = {mae:.4e}"

    return fig1, fig2, err_str

#######################################
# 9. Launch Gradio Interface
#######################################
iface = gr.Interface(
    fn=simulate_app,
    inputs=[
        gr.Slider(minimum=30, maximum=1000, step=10, value=300, label="Simulation Days"),
        gr.Number(value=100, label="Target Object Row Index (from CSV)"),
        gr.Textbox(value="2025-01-01 00:00:00", label="Start Date (YYYY-MM-DD HH:MM:SS)")
    ],
    outputs=[
        gr.Plot(label="Trajectory Comparison"),
        gr.Plot(label="Planets vs. Target Trajectory"),
        gr.Textbox(label="Error Metrics")
    ],
    title="Orbital Trajectory Simulation",
    description="Simulate a celestial target's trajectory along with external planetary positions using a physics-based Euler simulator."
)

if __name__ == "__main__":
    iface.launch()
