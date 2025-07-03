"""
Neural network controller using scikit-learn for trebuchet parameter optimization.
Implements a Multi-Layer Perceptron that learns to predict optimal parameters.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Tuple, List, Optional
import joblib
import os

# Import explainability components
try:
    from explainable_ai.explainer_engine import get_global_explainer
    from explainable_ai.audit_logger import audit_log, AuditLevel
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False


class TrebuchetController(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible neural network controller for trebuchet optimization.
    
    This model learns to map from target conditions (distance, wind) to
    optimal trebuchet parameters (counterweight mass, release angle).
    """
    
    def __init__(self, 
                 hidden_layer_sizes: Tuple[int, ...] = (64, 64),
                 activation: str = 'relu',
                 learning_rate_init: float = 0.001,
                 max_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize the trebuchet controller.
        
        Args:
            hidden_layer_sizes: Size of hidden layers
            activation: Activation function ('relu', 'tanh', 'logistic')
            learning_rate_init: Initial learning rate
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Initialize scalers and model
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
        
        # Parameter bounds for physical constraints
        self.mass_bounds = (50.0, 500.0)  # kg
        self.angle_bounds = (30.0, 60.0)  # degrees
        
    def _create_model(self):
        """Create the MLPRegressor model."""
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50
        )
    
    def _normalize_inputs(self, X: np.ndarray) -> np.ndarray:
        """Normalize input features (target_distance, wind_speed)."""
        return self.input_scaler.transform(X)
    
    def _denormalize_outputs(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalized outputs back to physical parameters."""
        y_scaled = self.output_scaler.inverse_transform(y_norm)
        
        # Apply physical constraints
        mass = np.clip(y_scaled[:, 0], self.mass_bounds[0], self.mass_bounds[1])
        angle = np.clip(y_scaled[:, 1], self.angle_bounds[0], self.angle_bounds[1])
        
        return np.column_stack([mass, angle])
    
    def _normalize_outputs(self, y: np.ndarray) -> np.ndarray:
        """Normalize output parameters for training."""
        return self.output_scaler.transform(y)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the controller on input-output pairs.
        
        Args:
            X: Input features, shape (n_samples, 2) [target_distance, wind_speed]
            y: Target outputs, shape (n_samples, 2) [counterweight_mass, release_angle]
            
        Returns:
            self: Returns the fitted estimator
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[1] != 2:
            raise ValueError("Input X must have exactly 2 features: [target_distance, wind_speed]")
        if y.shape[1] != 2:
            raise ValueError("Output y must have exactly 2 features: [counterweight_mass, release_angle]")
        
        # Fit scalers
        self.input_scaler.fit(X)
        self.output_scaler.fit(y)
        
        # Normalize data
        X_norm = self.input_scaler.transform(X)
        y_norm = self.output_scaler.transform(y)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_norm, y_norm)
        
        self.is_fitted = True
        
        # Register with explainability engine
        if EXPLAINABILITY_AVAILABLE:
            try:
                explainer = get_global_explainer()
                explainer.register_ml_model(
                    model_name="trebuchet_controller",
                    model=self.model,
                    feature_names=["target_distance", "wind_speed"]
                )
                
                # Log model training
                audit_log(
                    event_type="model_training",
                    actor="TrebuchetController",
                    action="fit_model",
                    resource="trebuchet_controller",
                    outcome="success",
                    details={
                        "training_samples": len(X),
                        "hidden_layers": self.hidden_layer_sizes,
                        "activation": self.activation,
                        "max_iter": self.max_iter
                    },
                    audit_level=AuditLevel.COMPLIANCE
                )
            except Exception as e:
                print(f"Warning: Could not register model with explainability engine: {e}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict optimal parameters for given target conditions.
        
        Args:
            X: Input features, shape (n_samples, 2) [target_distance, wind_speed]
            
        Returns:
            Predicted parameters, shape (n_samples, 2) [counterweight_mass, release_angle]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        if X.shape[1] != 2:
            raise ValueError("Input X must have exactly 2 features: [target_distance, wind_speed]")
        
        # Normalize inputs and predict
        X_norm = self.input_scaler.transform(X)
        y_norm = self.model.predict(X_norm)
        
        # Denormalize outputs
        return self._denormalize_outputs(y_norm)
    
    def predict_single(self, target_distance: float, wind_speed: float) -> Tuple[float, float]:
        """
        Predict parameters for a single target condition.
        
        Args:
            target_distance: Target distance in meters
            wind_speed: Wind speed in m/s
            
        Returns:
            Tuple of (counterweight_mass, release_angle)
        """
        X = np.array([[target_distance, wind_speed]])
        prediction = self.predict(X)
        return prediction[0, 0], prediction[0, 1]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score for the model.
        
        Args:
            X: Input features
            y: True target values
            
        Returns:
            R² score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        y_pred = self.predict(X)
        
        # Calculate R² manually to handle multi-output
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_training_history(self) -> dict:
        """
        Get training history from the MLPRegressor.
        
        Returns:
            Dictionary containing training metrics
        """
        if not self.is_fitted or self.model is None:
            return {}
        
        return {
            'loss_curve': getattr(self.model, 'loss_curve_', []),
            'validation_scores': getattr(self.model, 'validation_scores_', []),
            'best_validation_score': getattr(self.model, 'best_validation_score_', None),
            'n_iter': getattr(self.model, 'n_iter_', 0)
        }
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
            'hyperparameters': {
                'hidden_layer_sizes': self.hidden_layer_sizes,
                'activation': self.activation,
                'learning_rate_init': self.learning_rate_init,
                'max_iter': self.max_iter,
                'random_state': self.random_state
            },
            'bounds': {
                'mass_bounds': self.mass_bounds,
                'angle_bounds': self.angle_bounds
            }
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.input_scaler = model_data['input_scaler']
        self.output_scaler = model_data['output_scaler']
        
        # Restore hyperparameters
        hyperparams = model_data['hyperparameters']
        self.hidden_layer_sizes = hyperparams['hidden_layer_sizes']
        self.activation = hyperparams['activation']
        self.learning_rate_init = hyperparams['learning_rate_init']
        self.max_iter = hyperparams['max_iter']
        self.random_state = hyperparams['random_state']
        
        # Restore bounds
        bounds = model_data['bounds']
        self.mass_bounds = bounds['mass_bounds']
        self.angle_bounds = bounds['angle_bounds']
        
        self.is_fitted = True


class DataGenerator:
    """
    Generates training data for the trebuchet controller using the physics simulator.
    """
    
    def __init__(self, simulator):
        """
        Initialize with a physics simulator.
        
        Args:
            simulator: ProjectileSimulator instance
        """
        self.simulator = simulator
    
    def generate_random_dataset(self, 
                              n_samples: int = 1000,
                              distance_range: Tuple[float, float] = (50, 300),
                              wind_range: Tuple[float, float] = (-5, 5),
                              mass_range: Tuple[float, float] = (50, 500),
                              angle_range: Tuple[float, float] = (30, 60)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a random dataset for training.
        
        Args:
            n_samples: Number of samples to generate
            distance_range: Range of target distances
            wind_range: Range of wind speeds
            mass_range: Range of counterweight masses to sample
            angle_range: Range of release angles to sample
            
        Returns:
            Tuple of (inputs, outputs, achieved_distances)
        """
        np.random.seed(42)  # For reproducibility
        
        # Generate random parameters
        masses = np.random.uniform(mass_range[0], mass_range[1], n_samples)
        angles = np.random.uniform(angle_range[0], angle_range[1], n_samples)
        wind_speeds = np.random.uniform(wind_range[0], wind_range[1], n_samples)
        
        inputs = []
        outputs = []
        achieved_distances = []
        
        for i in range(n_samples):
            # Set wind speed for this simulation
            self.simulator.wind_speed = wind_speeds[i]
            
            # Calculate achieved distance with these parameters
            distance = self.simulator.calculate_distance_only(masses[i], angles[i])
            
            # Create input-output pair
            # Note: We're creating "inverse" training data where we know the parameters
            # that achieve a certain distance, so we can train the model to predict them
            inputs.append([distance, wind_speeds[i]])
            outputs.append([masses[i], angles[i]])
            achieved_distances.append(distance)
        
        return np.array(inputs), np.array(outputs), np.array(achieved_distances)
    
    def generate_targeted_dataset(self,
                                target_distances: List[float],
                                wind_speeds: List[float],
                                n_samples_per_condition: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dataset for specific target conditions using optimization.
        
        Args:
            target_distances: List of target distances
            wind_speeds: List of wind speeds
            n_samples_per_condition: Number of parameter combinations to try per condition
            
        Returns:
            Tuple of (inputs, outputs)
        """
        from scipy.optimize import minimize
        
        inputs = []
        outputs = []
        
        for target_dist in target_distances:
            for wind_speed in wind_speeds:
                self.simulator.wind_speed = wind_speed
                
                def objective(params):
                    mass, angle = params
                    achieved_dist = self.simulator.calculate_distance_only(mass, angle)
                    return (achieved_dist - target_dist) ** 2
                
                # Multiple random starts to find different solutions
                for _ in range(n_samples_per_condition):
                    initial_guess = [
                        np.random.uniform(50, 500),  # mass
                        np.random.uniform(30, 60)    # angle
                    ]
                    
                    bounds = [(50, 500), (30, 60)]
                    
                    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
                    
                    if result.success and result.fun < 1.0:  # Good fit
                        inputs.append([target_dist, wind_speed])
                        outputs.append(result.x)
        
        return np.array(inputs), np.array(outputs)


if __name__ == "__main__":
    # Test the controller
    from simulator import ProjectileSimulator
    
    # Create simulator and data generator
    simulator = ProjectileSimulator()
    data_gen = DataGenerator(simulator)
    
    # Generate training data
    print("Generating training data...")
    X, y, distances = data_gen.generate_random_dataset(n_samples=500)
    
    print(f"Generated {len(X)} training samples")
    print(f"Distance range: {distances.min():.1f} - {distances.max():.1f} m")
    print(f"Mass range: {y[:, 0].min():.1f} - {y[:, 0].max():.1f} kg")
    print(f"Angle range: {y[:, 1].min():.1f} - {y[:, 1].max():.1f} deg")
    
    # Train controller
    print("\nTraining controller...")
    controller = TrebuchetController(max_iter=200)
    controller.fit(X, y)
    
    # Test prediction
    test_distance = 150.0
    test_wind = 2.0
    predicted_mass, predicted_angle = controller.predict_single(test_distance, test_wind)
    
    print(f"\nTest prediction:")
    print(f"Target: {test_distance}m distance, {test_wind}m/s wind")
    print(f"Predicted: {predicted_mass:.1f}kg mass, {predicted_angle:.1f}° angle")
    
    # Verify prediction
    simulator.wind_speed = test_wind
    achieved_distance = simulator.calculate_distance_only(predicted_mass, predicted_angle)
    print(f"Achieved distance: {achieved_distance:.1f}m (error: {abs(achieved_distance - test_distance):.1f}m)")
    
    # Training history
    history = controller.get_training_history()
    print(f"\nTraining completed in {history['n_iter']} iterations")
    if history['loss_curve']:
        print(f"Final training loss: {history['loss_curve'][-1]:.6f}")
    if history['best_validation_score'] is not None:
        print(f"Best validation score: {history['best_validation_score']:.6f}")