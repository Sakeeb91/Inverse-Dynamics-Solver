"""
Unit tests for the neural network model module.
"""

import unittest
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TrebuchetController, DataGenerator
from simulator import ProjectileSimulator


class TestTrebuchetController(unittest.TestCase):
    """Test cases for TrebuchetController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = TrebuchetController(
            hidden_layer_sizes=(32, 32),
            max_iter=50,  # Small for testing
            random_state=42
        )
        
        # Generate small test dataset
        np.random.seed(42)
        self.X_test = np.random.uniform([50, -5], [300, 5], (50, 2))
        self.y_test = np.random.uniform([50, 30], [500, 60], (50, 2))
    
    def test_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.hidden_layer_sizes, (32, 32))
        self.assertEqual(self.controller.max_iter, 50)
        self.assertEqual(self.controller.random_state, 42)
        self.assertFalse(self.controller.is_fitted)
    
    def test_input_validation(self):
        """Test input validation."""
        # Wrong input dimensions
        X_wrong = np.random.randn(10, 3)  # Should be 2 features
        y_wrong = np.random.randn(10, 3)  # Should be 2 outputs
        
        with self.assertRaises(ValueError):
            self.controller.fit(X_wrong, self.y_test)
        
        with self.assertRaises(ValueError):
            self.controller.fit(self.X_test, y_wrong)
    
    def test_fit_predict_cycle(self):
        """Test complete fit-predict cycle."""
        # Fit model
        self.controller.fit(self.X_test, self.y_test)
        self.assertTrue(self.controller.is_fitted)
        
        # Test prediction
        predictions = self.controller.predict(self.X_test[:5])
        
        # Check prediction shape and bounds
        self.assertEqual(predictions.shape, (5, 2))
        
        # Check physical bounds
        masses = predictions[:, 0]
        angles = predictions[:, 1]
        
        self.assertTrue(np.all(masses >= 50.0))
        self.assertTrue(np.all(masses <= 500.0))
        self.assertTrue(np.all(angles >= 30.0))
        self.assertTrue(np.all(angles <= 60.0))
    
    def test_predict_single(self):
        """Test single prediction interface."""
        self.controller.fit(self.X_test, self.y_test)
        
        mass, angle = self.controller.predict_single(150.0, 2.0)
        
        # Check types and bounds
        self.assertIsInstance(mass, (float, np.floating))
        self.assertIsInstance(angle, (float, np.floating))
        self.assertGreaterEqual(mass, 50.0)
        self.assertLessEqual(mass, 500.0)
        self.assertGreaterEqual(angle, 30.0)
        self.assertLessEqual(angle, 60.0)
    
    def test_predict_before_fit(self):
        """Test prediction before fitting raises error."""
        with self.assertRaises(ValueError):
            self.controller.predict(self.X_test)
        
        with self.assertRaises(ValueError):
            self.controller.predict_single(150.0, 2.0)
    
    def test_score(self):
        """Test scoring functionality."""
        self.controller.fit(self.X_test, self.y_test)
        
        score = self.controller.score(self.X_test, self.y_test)
        
        # R² score should be between -∞ and 1, but typically positive for fitted data
        self.assertIsInstance(score, (float, np.floating))
        self.assertGreater(score, -10)  # Reasonable lower bound
        self.assertLessEqual(score, 1.0)
    
    def test_training_history(self):
        """Test training history retrieval."""
        # Before fitting
        history = self.controller.get_training_history()
        self.assertEqual(history, {})
        
        # After fitting
        self.controller.fit(self.X_test, self.y_test)
        history = self.controller.get_training_history()
        
        # Should have some training information
        self.assertIsInstance(history, dict)
        self.assertIn('n_iter', history)
    
    def test_save_load_model(self):
        """Test model persistence."""
        # Train model
        self.controller.fit(self.X_test, self.y_test)
        original_prediction = self.controller.predict_single(150.0, 2.0)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            self.controller.save_model(model_path)
            
            # Create new controller and load
            new_controller = TrebuchetController()
            new_controller.load_model(model_path)
            
            # Test loaded model
            self.assertTrue(new_controller.is_fitted)
            loaded_prediction = new_controller.predict_single(150.0, 2.0)
            
            # Predictions should match
            self.assertAlmostEqual(original_prediction[0], loaded_prediction[0], places=3)
            self.assertAlmostEqual(original_prediction[1], loaded_prediction[1], places=3)
        
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_save_before_fit(self):
        """Test saving before fitting raises error."""
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            with self.assertRaises(ValueError):
                self.controller.save_model(f.name)


class TestDataGenerator(unittest.TestCase):
    """Test cases for DataGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = ProjectileSimulator(wind_speed=0.0)
        self.data_gen = DataGenerator(self.simulator)
    
    def test_random_dataset_generation(self):
        """Test random dataset generation."""
        n_samples = 20
        X, y, distances = self.data_gen.generate_random_dataset(n_samples=n_samples)
        
        # Check shapes
        self.assertEqual(X.shape, (n_samples, 2))
        self.assertEqual(y.shape, (n_samples, 2))
        self.assertEqual(len(distances), n_samples)
        
        # Check data ranges
        target_distances = X[:, 0]
        wind_speeds = X[:, 1]
        masses = y[:, 0]
        angles = y[:, 1]
        
        self.assertTrue(np.all(target_distances >= 50))
        self.assertTrue(np.all(target_distances <= 300))
        self.assertTrue(np.all(wind_speeds >= -5))
        self.assertTrue(np.all(wind_speeds <= 5))
        self.assertTrue(np.all(masses >= 50))
        self.assertTrue(np.all(masses <= 500))
        self.assertTrue(np.all(angles >= 30))
        self.assertTrue(np.all(angles <= 60))
    
    def test_random_dataset_consistency(self):
        """Test that generated data is physically consistent."""
        X, y, distances = self.data_gen.generate_random_dataset(n_samples=10)
        
        # Verify that the parameters actually achieve the distances
        for i in range(len(X)):
            wind_speed = X[i, 1]
            mass = y[i, 0]
            angle = y[i, 1]
            expected_distance = distances[i]
            
            # Simulate with the same parameters
            self.simulator.wind_speed = wind_speed
            actual_distance = self.simulator.calculate_distance_only(mass, angle)
            
            # Should match (allowing for numerical precision)
            self.assertAlmostEqual(actual_distance, expected_distance, places=1)
    
    def test_targeted_dataset_generation(self):
        """Test targeted dataset generation."""
        target_distances = [100.0, 200.0]
        wind_speeds = [0.0, 2.0]
        
        X, y = self.data_gen.generate_targeted_dataset(
            target_distances, wind_speeds, n_samples_per_condition=2
        )
        
        # Should have some samples (exact number depends on optimization success)
        self.assertGreater(len(X), 0)
        self.assertEqual(X.shape[1], 2)
        self.assertEqual(y.shape[1], 2)
        self.assertEqual(len(X), len(y))
        
        # Check that inputs are from our target conditions
        unique_distances = np.unique(X[:, 0])
        unique_winds = np.unique(X[:, 1])
        
        for dist in unique_distances:
            self.assertIn(dist, target_distances)
        
        for wind in unique_winds:
            self.assertIn(wind, wind_speeds)
    
    def test_custom_ranges(self):
        """Test custom parameter ranges."""
        custom_ranges = {
            'distance_range': (80, 120),
            'wind_range': (-2, 2),
            'mass_range': (100, 200),
            'angle_range': (40, 50)
        }
        
        X, y, _ = self.data_gen.generate_random_dataset(
            n_samples=20, **custom_ranges
        )
        
        # Check custom ranges are respected
        self.assertTrue(np.all(X[:, 0] >= 80))
        self.assertTrue(np.all(X[:, 0] <= 120))
        self.assertTrue(np.all(X[:, 1] >= -2))
        self.assertTrue(np.all(X[:, 1] <= 2))
        self.assertTrue(np.all(y[:, 0] >= 100))
        self.assertTrue(np.all(y[:, 0] <= 200))
        self.assertTrue(np.all(y[:, 1] >= 40))
        self.assertTrue(np.all(y[:, 1] <= 50))


if __name__ == '__main__':
    unittest.main(verbosity=2)