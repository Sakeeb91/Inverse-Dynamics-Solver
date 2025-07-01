"""
Unit tests for the physics simulator module.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import ProjectileSimulator, calculate_optimal_parameters_analytical


class TestProjectileSimulator(unittest.TestCase):
    """Test cases for ProjectileSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = ProjectileSimulator(wind_speed=0.0)
    
    def test_initialization(self):
        """Test simulator initialization."""
        self.assertEqual(self.simulator.wind_speed, 0.0)
        self.assertAlmostEqual(self.simulator.proj_area, np.pi * 0.05**2)
    
    def test_trajectory_basic(self):
        """Test basic trajectory calculation."""
        # Test with reasonable parameters
        mass = 200.0
        angle = 45.0
        
        t_points, trajectory, distance = self.simulator.simulate_trajectory(mass, angle)
        
        # Basic sanity checks
        self.assertGreater(len(trajectory), 0)
        self.assertGreater(distance, 0)
        self.assertLess(distance, 1000)  # Reasonable upper bound
        
        # Check trajectory shape
        self.assertEqual(trajectory.shape[1], 4)  # [x, y, vx, vy]
        
        # Initial conditions
        self.assertAlmostEqual(trajectory[0, 0], 0.0)  # Initial x
        self.assertAlmostEqual(trajectory[0, 1], 2.0)  # Initial y (trebuchet height)
        
        # Final condition
        self.assertLessEqual(trajectory[-1, 1], 0.1)  # Should land near ground
    
    def test_wind_effect(self):
        """Test wind effect on trajectory."""
        mass = 200.0
        angle = 45.0
        
        # No wind
        self.simulator.wind_speed = 0.0
        _, _, distance_no_wind = self.simulator.simulate_trajectory(mass, angle)
        
        # Tailwind
        self.simulator.wind_speed = 5.0
        _, _, distance_tailwind = self.simulator.simulate_trajectory(mass, angle)
        
        # Headwind
        self.simulator.wind_speed = -5.0
        _, _, distance_headwind = self.simulator.simulate_trajectory(mass, angle)
        
        # Tailwind should increase distance, headwind should decrease
        self.assertGreater(distance_tailwind, distance_no_wind)
        self.assertLess(distance_headwind, distance_no_wind)
    
    def test_parameter_effects(self):
        """Test effect of different parameters."""
        base_mass = 200.0
        base_angle = 45.0
        
        # Test mass effect
        _, _, distance_light = self.simulator.simulate_trajectory(100.0, base_angle)
        _, _, distance_heavy = self.simulator.simulate_trajectory(400.0, base_angle)
        
        # Heavier counterweight should give more distance
        self.assertGreater(distance_heavy, distance_light)
        
        # Test angle effect (45° should be close to optimal)
        _, _, distance_30 = self.simulator.simulate_trajectory(base_mass, 30.0)
        _, _, distance_45 = self.simulator.simulate_trajectory(base_mass, 45.0)
        _, _, distance_60 = self.simulator.simulate_trajectory(base_mass, 60.0)
        
        # 45° should be better than extreme angles
        self.assertGreater(distance_45, distance_30)
        self.assertGreater(distance_45, distance_60)
    
    def test_calculate_distance_only(self):
        """Test the optimized distance calculation."""
        mass = 200.0
        angle = 45.0
        
        distance_full = self.simulator.simulate_trajectory(mass, angle)[2]
        distance_only = self.simulator.calculate_distance_only(mass, angle)
        
        # Should give same result
        self.assertAlmostEqual(distance_full, distance_only, places=1)
    
    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases."""
        # Minimum mass
        distance_min = self.simulator.calculate_distance_only(50.0, 45.0)
        self.assertGreater(distance_min, 0)
        
        # Maximum mass
        distance_max = self.simulator.calculate_distance_only(500.0, 45.0)
        self.assertGreater(distance_max, distance_min)
        
        # Extreme angles
        distance_low = self.simulator.calculate_distance_only(200.0, 30.0)
        distance_high = self.simulator.calculate_distance_only(200.0, 60.0)
        
        self.assertGreater(distance_low, 0)
        self.assertGreater(distance_high, 0)


class TestAnalyticalFunctions(unittest.TestCase):
    """Test analytical helper functions."""
    
    def test_analytical_estimation(self):
        """Test analytical parameter estimation."""
        target_distance = 150.0
        wind_speed = 0.0
        
        estimated_mass, estimated_angle = calculate_optimal_parameters_analytical(
            target_distance, wind_speed
        )
        
        # Check reasonable bounds
        self.assertGreaterEqual(estimated_mass, 50.0)
        self.assertLessEqual(estimated_mass, 500.0)
        self.assertGreaterEqual(estimated_angle, 30.0)
        self.assertLessEqual(estimated_angle, 60.0)
    
    def test_analytical_different_distances(self):
        """Test analytical estimation for different distances."""
        distances = [50.0, 100.0, 200.0, 300.0]
        
        for distance in distances:
            mass, angle = calculate_optimal_parameters_analytical(distance, 0.0)
            
            # Should give reasonable parameters
            self.assertGreater(mass, 0)
            self.assertGreater(angle, 0)
            
            # Longer distances should generally need more mass
            if distance > 50.0:
                mass_short, _ = calculate_optimal_parameters_analytical(50.0, 0.0)
                # Allow some flexibility due to simplifications in analytical model
                self.assertGreaterEqual(mass, mass_short * 0.8)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)