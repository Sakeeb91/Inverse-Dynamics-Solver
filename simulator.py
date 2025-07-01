"""
Physics simulation engine for trebuchet projectile motion.
Implements differential equation solving with numerical integration.
"""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, List
import matplotlib.pyplot as plt

# Physical Constants
G = 9.81  # Gravitational acceleration (m/s^2)
RHO = 1.225  # Air density at sea level (kg/m^3)
C_D = 0.47  # Drag coefficient for a sphere
PROJ_RADIUS = 0.05  # Projectile radius (m)
PROJ_MASS = 1.0  # Projectile mass (kg)
TREBUCHET_HEIGHT = 2.0  # Initial height (m)


class ProjectileSimulator:
    """
    Physics-based projectile motion simulator with air resistance and wind effects.
    """
    
    def __init__(self, wind_speed: float = 0.0):
        """
        Initialize the simulator.
        
        Args:
            wind_speed: Horizontal wind speed (m/s). Positive is tailwind.
        """
        self.wind_speed = wind_speed
        self.proj_area = np.pi * PROJ_RADIUS**2
        
    def _projectile_ode(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Defines the system of ODEs for projectile motion.
        
        Args:
            state: Current state [x, y, vx, vy]
            t: Time (not used but required by odeint)
            
        Returns:
            Derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt]
        """
        x, y, vx, vy = state
        
        # Relative velocity (accounting for wind)
        v_rel_x = vx - self.wind_speed
        v_rel_y = vy
        v_rel_mag = np.sqrt(v_rel_x**2 + v_rel_y**2)
        
        # Avoid division by zero
        if v_rel_mag < 1e-10:
            v_rel_mag = 1e-10
            
        # Drag force magnitude
        drag_force_mag = 0.5 * RHO * v_rel_mag**2 * C_D * self.proj_area
        
        # Drag force components (opposite to relative velocity)
        drag_force_x = -drag_force_mag * (v_rel_x / v_rel_mag)
        drag_force_y = -drag_force_mag * (v_rel_y / v_rel_mag)
        
        # Equations of motion
        dx_dt = vx
        dy_dt = vy
        dvx_dt = drag_force_x / PROJ_MASS
        dvy_dt = (drag_force_y / PROJ_MASS) - G
        
        return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])
    
    def simulate_trajectory(self, 
                          counterweight_mass: float, 
                          release_angle_deg: float, 
                          t_max: float = 15.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Simulate the complete projectile trajectory.
        
        Args:
            counterweight_mass: Mass of trebuchet counterweight (kg)
            release_angle_deg: Release angle in degrees
            t_max: Maximum simulation time (s)
            
        Returns:
            Tuple of (time_points, trajectory_states, final_distance)
        """
        # Convert angle to radians
        release_angle_rad = np.deg2rad(release_angle_deg)
        
        # Calculate initial velocity from trebuchet mechanics
        # Simplified model: v0 = sqrt(2 * g * h * M_cw / M_proj)
        # Where h is the effective height drop of counterweight
        effective_height = 2.0  # meters
        initial_speed = np.sqrt(2 * G * effective_height * counterweight_mass / PROJ_MASS)
        
        # Initial velocity components
        v0_x = initial_speed * np.cos(release_angle_rad)
        v0_y = initial_speed * np.sin(release_angle_rad)
        
        # Initial state: [x, y, vx, vy]
        initial_state = np.array([0.0, TREBUCHET_HEIGHT, v0_x, v0_y])
        
        # Time points for integration
        t_points = np.linspace(0, t_max, int(t_max * 100))  # 100 points per second
        
        # Solve ODE
        trajectory = odeint(self._projectile_ode, initial_state, t_points)
        
        # Find landing point (when y <= 0)
        landing_indices = np.where(trajectory[:, 1] <= 0)[0]
        
        if len(landing_indices) > 0:
            # Interpolate to find exact landing point
            landing_idx = landing_indices[0]
            if landing_idx > 0:
                # Linear interpolation between last positive and first negative y
                y_prev = trajectory[landing_idx - 1, 1]
                y_curr = trajectory[landing_idx, 1]
                x_prev = trajectory[landing_idx - 1, 0]
                x_curr = trajectory[landing_idx, 0]
                
                # Interpolate x coordinate at y = 0
                final_distance = x_prev + (x_curr - x_prev) * (-y_prev) / (y_curr - y_prev)
                
                # Truncate trajectory at landing
                trajectory = trajectory[:landing_idx + 1]
                t_points = t_points[:landing_idx + 1]
            else:
                final_distance = trajectory[0, 0]
        else:
            # Projectile didn't land within simulation time
            final_distance = trajectory[-1, 0]
        
        return t_points, trajectory, max(0, final_distance)
    
    def calculate_distance_only(self, 
                              counterweight_mass: float, 
                              release_angle_deg: float) -> float:
        """
        Calculate only the final distance (optimized for training).
        
        Args:
            counterweight_mass: Mass of trebuchet counterweight (kg)
            release_angle_deg: Release angle in degrees
            
        Returns:
            Final horizontal distance (m)
        """
        _, _, distance = self.simulate_trajectory(counterweight_mass, release_angle_deg)
        return distance
    
    def generate_trajectory_plot(self, 
                               counterweight_mass: float, 
                               release_angle_deg: float,
                               save_path: str = None) -> plt.Figure:
        """
        Generate a matplotlib plot of the trajectory.
        
        Args:
            counterweight_mass: Mass of trebuchet counterweight (kg)
            release_angle_deg: Release angle in degrees
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        t_points, trajectory, final_distance = self.simulate_trajectory(
            counterweight_mass, release_angle_deg
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
        
        # Mark start and end points
        ax.plot(0, TREBUCHET_HEIGHT, 'go', markersize=8, label='Launch Point')
        ax.plot(final_distance, 0, 'ro', markersize=8, label=f'Landing Point ({final_distance:.1f}m)')
        
        # Ground line
        ax.axhline(y=0, color='brown', linestyle='-', alpha=0.3, label='Ground')
        
        # Formatting
        ax.set_xlabel('Horizontal Distance (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title(f'Trebuchet Trajectory\n'
                    f'Counterweight: {counterweight_mass:.1f}kg, '
                    f'Angle: {release_angle_deg:.1f}°, '
                    f'Wind: {self.wind_speed:.1f}m/s')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(bottom=-0.5)
        
        # Add parameter text box
        params_text = f'Parameters:\n' \
                     f'Mass: {counterweight_mass:.1f} kg\n' \
                     f'Angle: {release_angle_deg:.1f}°\n' \
                     f'Wind: {self.wind_speed:.1f} m/s\n' \
                     f'Distance: {final_distance:.1f} m'
        ax.text(0.02, 0.98, params_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def calculate_optimal_parameters_analytical(target_distance: float, 
                                          wind_speed: float = 0.0) -> Tuple[float, float]:
    """
    Analytical approximation for optimal parameters (for comparison).
    This is a simplified calculation that ignores air resistance.
    
    Args:
        target_distance: Desired distance (m)
        wind_speed: Wind speed (m/s)
        
    Returns:
        Tuple of (estimated_counterweight_mass, estimated_release_angle)
    """
    # Optimal angle for maximum range without air resistance is 45°
    optimal_angle = 45.0
    
    # Estimate required initial velocity for target distance
    # Range formula: R = v0^2 * sin(2θ) / g
    required_v0_squared = target_distance * G / np.sin(2 * np.deg2rad(optimal_angle))
    
    # Estimate counterweight mass from initial velocity
    # v0 = sqrt(2 * g * h * M_cw / M_proj)
    effective_height = 2.0
    estimated_mass = required_v0_squared * PROJ_MASS / (2 * G * effective_height)
    
    # Ensure reasonable bounds
    estimated_mass = np.clip(estimated_mass, 50, 500)
    
    return estimated_mass, optimal_angle


if __name__ == "__main__":
    # Test the simulator
    simulator = ProjectileSimulator(wind_speed=2.0)
    
    # Test trajectory
    mass = 200.0
    angle = 45.0
    
    print(f"Testing with mass={mass}kg, angle={angle}°, wind={simulator.wind_speed}m/s")
    
    distance = simulator.calculate_distance_only(mass, angle)
    print(f"Calculated distance: {distance:.2f}m")
    
    # Generate plot
    fig = simulator.generate_trajectory_plot(mass, angle)
    plt.show()