"""
Training logic for the differentiable trebuchet system.
Handles data generation, model training, and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path

from simulator import ProjectileSimulator
from model import TrebuchetController, DataGenerator


class TrebuchetTrainer:
    """
    Comprehensive training system for the trebuchet controller.
    Handles data generation, training, evaluation, and visualization.
    """
    
    def __init__(self, 
                 simulator: ProjectileSimulator,
                 controller: TrebuchetController,
                 save_dir: str = "visualizations"):
        """
        Initialize the trainer.
        
        Args:
            simulator: Physics simulator instance
            controller: Neural network controller
            save_dir: Directory to save visualizations and models
        """
        self.simulator = simulator
        self.controller = controller
        self.data_generator = DataGenerator(simulator)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.training_history = {}
        self.evaluation_results = {}
        
    def generate_training_data(self, 
                             n_samples: int = 1000,
                             strategy: str = "random") -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data using specified strategy.
        
        Args:
            n_samples: Number of training samples
            strategy: Data generation strategy ("random" or "targeted")
            
        Returns:
            Tuple of (X, y) training data
        """
        print(f"Generating {n_samples} training samples using '{strategy}' strategy...")
        
        if strategy == "random":
            X, y, _ = self.data_generator.generate_random_dataset(n_samples)
        elif strategy == "targeted":
            # Generate targeted data for specific conditions
            target_distances = np.linspace(50, 300, 10)
            wind_speeds = [-5, -2, 0, 2, 5]
            samples_per_condition = max(1, n_samples // (len(target_distances) * len(wind_speeds)))
            X, y = self.data_generator.generate_targeted_dataset(
                target_distances, wind_speeds, samples_per_condition
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"Generated {len(X)} samples")
        print(f"Target distance range: {X[:, 0].min():.1f} - {X[:, 0].max():.1f} m")
        print(f"Wind speed range: {X[:, 1].min():.1f} - {X[:, 1].max():.1f} m/s")
        
        return X, y
    
    def train_model(self, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   validation_split: float = 0.2,
                   save_model: bool = True) -> Dict:
        """
        Train the controller model.
        
        Args:
            X: Input features
            y: Target outputs
            validation_split: Fraction of data for validation
            save_model: Whether to save the trained model
            
        Returns:
            Training results dictionary
        """
        print(f"Training model on {len(X)} samples...")
        
        # Split data
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Train model
        start_time = time.time()
        self.controller.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        train_score = self.controller.score(X_train, y_train)
        val_score = self.controller.score(X_val, y_val)
        
        # Get training history
        history = self.controller.get_training_history()
        
        # Store results
        results = {
            'training_time': training_time,
            'train_score': train_score,
            'val_score': val_score,
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'history': history,
            'final_loss': history.get('loss_curve', [0])[-1] if history.get('loss_curve') else 0
        }
        
        self.training_history = results
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Training R² score: {train_score:.4f}")
        print(f"Validation R² score: {val_score:.4f}")
        
        if save_model:
            model_path = self.save_dir / "trained_model.pkl"
            self.controller.save_model(str(model_path))
            print(f"Model saved to {model_path}")
        
        return results
    
    def evaluate_model(self, 
                      test_conditions: List[Tuple[float, float]] = None,
                      n_test_samples: int = 100) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            test_conditions: List of (target_distance, wind_speed) tuples
            n_test_samples: Number of random test samples
            
        Returns:
            Evaluation results dictionary
        """
        print("Evaluating model performance...")
        
        if test_conditions is None:
            # Generate random test conditions
            np.random.seed(123)  # Different seed from training
            test_conditions = [
                (np.random.uniform(50, 300), np.random.uniform(-5, 5))
                for _ in range(n_test_samples)
            ]
        
        results = {
            'test_conditions': test_conditions,
            'predictions': [],
            'achieved_distances': [],
            'errors': [],
            'relative_errors': []
        }
        
        for target_dist, wind_speed in test_conditions:
            # Get model prediction
            pred_mass, pred_angle = self.controller.predict_single(target_dist, wind_speed)
            
            # Simulate with predicted parameters
            self.simulator.wind_speed = wind_speed
            achieved_dist = self.simulator.calculate_distance_only(pred_mass, pred_angle)
            
            # Calculate errors
            error = abs(achieved_dist - target_dist)
            rel_error = error / target_dist * 100
            
            results['predictions'].append((pred_mass, pred_angle))
            results['achieved_distances'].append(achieved_dist)
            results['errors'].append(error)
            results['relative_errors'].append(rel_error)
        
        # Summary statistics
        errors = np.array(results['errors'])
        rel_errors = np.array(results['relative_errors'])
        
        results['summary'] = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'mean_rel_error': np.mean(rel_errors),
            'std_rel_error': np.std(rel_errors),
            'accuracy_within_5m': np.sum(errors <= 5.0) / len(errors) * 100,
            'accuracy_within_10m': np.sum(errors <= 10.0) / len(errors) * 100
        }
        
        self.evaluation_results = results
        
        print(f"Mean absolute error: {results['summary']['mean_error']:.2f} ± {results['summary']['std_error']:.2f} m")
        print(f"Mean relative error: {results['summary']['mean_rel_error']:.1f} ± {results['summary']['std_rel_error']:.1f} %")
        print(f"Accuracy within 5m: {results['summary']['accuracy_within_5m']:.1f}%")
        print(f"Accuracy within 10m: {results['summary']['accuracy_within_10m']:.1f}%")
        
        return results
    
    def create_training_visualizations(self, save_plots: bool = True) -> Dict[str, go.Figure]:
        """
        Create comprehensive training visualizations.
        
        Args:
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary of plotly figures
        """
        figures = {}
        
        if not self.training_history:
            print("No training history available. Train model first.")
            return figures
        
        # 1. Training Loss Curve
        history = self.training_history['history']
        if history.get('loss_curve'):
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history['loss_curve'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ))
            
            if history.get('validation_scores'):
                # Convert validation scores to loss (assuming they're R² scores)
                val_losses = [1 - score for score in history['validation_scores']]
                fig_loss.add_trace(go.Scatter(
                    y=val_losses,
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red')
                ))
            
            fig_loss.update_layout(
                title='Training Loss Convergence',
                xaxis_title='Iteration',
                yaxis_title='Loss',
                template='plotly_white'
            )
            figures['training_loss'] = fig_loss
            
            if save_plots:
                fig_loss.write_html(str(self.save_dir / "plots" / "training_loss.html"))
        
        # 2. Model Performance Summary
        if self.evaluation_results:
            eval_results = self.evaluation_results
            
            # Error distribution
            fig_error = go.Figure()
            fig_error.add_trace(go.Histogram(
                x=eval_results['errors'],
                nbinsx=30,
                name='Absolute Error',
                opacity=0.7
            ))
            fig_error.update_layout(
                title='Prediction Error Distribution',
                xaxis_title='Absolute Error (m)',
                yaxis_title='Frequency',
                template='plotly_white'
            )
            figures['error_distribution'] = fig_error
            
            if save_plots:
                fig_error.write_html(str(self.save_dir / "plots" / "error_distribution.html"))
            
            # Prediction vs Target scatter plot
            targets = [cond[0] for cond in eval_results['test_conditions']]
            achieved = eval_results['achieved_distances']
            
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=targets,
                y=achieved,
                mode='markers',
                name='Predictions',
                marker=dict(
                    color=eval_results['errors'],
                    colorscale='Viridis',
                    colorbar=dict(title='Error (m)'),
                    size=8
                )
            ))
            
            # Perfect prediction line
            min_dist, max_dist = min(targets), max(targets)
            fig_scatter.add_trace(go.Scatter(
                x=[min_dist, max_dist],
                y=[min_dist, max_dist],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig_scatter.update_layout(
                title='Predicted vs Target Distance',
                xaxis_title='Target Distance (m)',
                yaxis_title='Achieved Distance (m)',
                template='plotly_white'
            )
            figures['prediction_scatter'] = fig_scatter
            
            if save_plots:
                fig_scatter.write_html(str(self.save_dir / "plots" / "prediction_scatter.html"))
        
        print(f"Created {len(figures)} visualization plots")
        return figures
    
    def create_physics_visualizations(self, 
                                    example_conditions: List[Tuple[float, float]] = None,
                                    save_plots: bool = True) -> Dict[str, go.Figure]:
        """
        Create physics-based visualizations.
        
        Args:
            example_conditions: List of (target_distance, wind_speed) conditions
            save_plots: Whether to save plots
            
        Returns:
            Dictionary of plotly figures
        """
        figures = {}
        
        if example_conditions is None:
            example_conditions = [(100, 0), (150, 2), (200, -3), (250, 5)]
        
        # 1. Trajectory Comparison
        fig_traj = go.Figure()
        
        for i, (target_dist, wind_speed) in enumerate(example_conditions):
            # Get model prediction
            pred_mass, pred_angle = self.controller.predict_single(target_dist, wind_speed)
            
            # Simulate trajectory
            self.simulator.wind_speed = wind_speed
            t_points, trajectory, final_dist = self.simulator.simulate_trajectory(
                pred_mass, pred_angle
            )
            
            fig_traj.add_trace(go.Scatter(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                mode='lines',
                name=f'Target: {target_dist}m, Wind: {wind_speed}m/s',
                line=dict(width=2)
            ))
            
            # Mark landing point
            fig_traj.add_trace(go.Scatter(
                x=[final_dist],
                y=[0],
                mode='markers',
                marker=dict(size=10, symbol='diamond'),
                name=f'Landing: {final_dist:.1f}m',
                showlegend=False
            ))
        
        fig_traj.update_layout(
            title='Trajectory Comparison for Different Conditions',
            xaxis_title='Horizontal Distance (m)',
            yaxis_title='Height (m)',
            template='plotly_white'
        )
        figures['trajectory_comparison'] = fig_traj
        
        if save_plots:
            fig_traj.write_html(str(self.save_dir / "plots" / "trajectory_comparison.html"))
        
        # 2. Parameter Sensitivity Analysis
        distance_range = np.linspace(50, 300, 20)
        wind_range = np.linspace(-5, 5, 15)
        
        predicted_masses = np.zeros((len(wind_range), len(distance_range)))
        predicted_angles = np.zeros((len(wind_range), len(distance_range)))
        
        for i, wind in enumerate(wind_range):
            for j, dist in enumerate(distance_range):
                mass, angle = self.controller.predict_single(dist, wind)
                predicted_masses[i, j] = mass
                predicted_angles[i, j] = angle
        
        # Mass heatmap
        fig_mass = go.Figure()
        fig_mass.add_trace(go.Heatmap(
            z=predicted_masses,
            x=distance_range,
            y=wind_range,
            colorscale='Viridis',
            colorbar=dict(title='Mass (kg)')
        ))
        fig_mass.update_layout(
            title='Predicted Counterweight Mass',
            xaxis_title='Target Distance (m)',
            yaxis_title='Wind Speed (m/s)',
            template='plotly_white'
        )
        figures['mass_sensitivity'] = fig_mass
        
        # Angle heatmap
        fig_angle = go.Figure()
        fig_angle.add_trace(go.Heatmap(
            z=predicted_angles,
            x=distance_range,
            y=wind_range,
            colorscale='Plasma',
            colorbar=dict(title='Angle (deg)')
        ))
        fig_angle.update_layout(
            title='Predicted Release Angle',
            xaxis_title='Target Distance (m)',
            yaxis_title='Wind Speed (m/s)',
            template='plotly_white'
        )
        figures['angle_sensitivity'] = fig_angle
        
        if save_plots:
            fig_mass.write_html(str(self.save_dir / "plots" / "mass_sensitivity.html"))
            fig_angle.write_html(str(self.save_dir / "plots" / "angle_sensitivity.html"))
        
        print(f"Created {len(figures)} physics visualization plots")
        return figures
    
    def full_training_pipeline(self, 
                             n_samples: int = 1000,
                             strategy: str = "random",
                             create_visualizations: bool = True) -> Dict:
        """
        Complete training pipeline from data generation to evaluation.
        
        Args:
            n_samples: Number of training samples
            strategy: Data generation strategy
            create_visualizations: Whether to create visualization plots
            
        Returns:
            Complete results dictionary
        """
        print("=" * 60)
        print("DIFFERENTIABLE TREBUCHET TRAINING PIPELINE")
        print("=" * 60)
        
        # 1. Generate training data
        X, y = self.generate_training_data(n_samples, strategy)
        
        # 2. Train model
        training_results = self.train_model(X, y)
        
        # 3. Evaluate model
        evaluation_results = self.evaluate_model()
        
        # 4. Create visualizations
        if create_visualizations:
            print("\nCreating visualizations...")
            training_viz = self.create_training_visualizations()
            physics_viz = self.create_physics_visualizations()
            
            # Update visualization log
            self._update_visualization_log()
        
        # 5. Compile complete results
        complete_results = {
            'training': training_results,
            'evaluation': evaluation_results,
            'data_info': {
                'n_samples': len(X),
                'strategy': strategy,
                'input_range': {
                    'distance': (X[:, 0].min(), X[:, 0].max()),
                    'wind': (X[:, 1].min(), X[:, 1].max())
                },
                'output_range': {
                    'mass': (y[:, 0].min(), y[:, 0].max()),
                    'angle': (y[:, 1].min(), y[:, 1].max())
                }
            }
        }
        
        print("\n" + "=" * 60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return complete_results
    
    def _update_visualization_log(self):
        """Update the visualization log with current timestamp."""
        log_path = self.save_dir / "visualization_log.md"
        if log_path.exists():
            content = log_path.read_text()
            updated_content = content.replace(
                "*Last updated: [Auto-generated timestamp]*",
                f"*Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}*"
            )
            log_path.write_text(updated_content)


if __name__ == "__main__":
    # Example usage
    print("Testing Trebuchet Training System")
    
    # Initialize components
    simulator = ProjectileSimulator(wind_speed=0.0)
    controller = TrebuchetController(max_iter=100)  # Reduced for testing
    trainer = TrebuchetTrainer(simulator, controller)
    
    # Run training pipeline
    results = trainer.full_training_pipeline(
        n_samples=200,  # Reduced for testing
        strategy="random",
        create_visualizations=True
    )
    
    print("\nTraining pipeline completed successfully!")
    print(f"Final validation R² score: {results['training']['val_score']:.4f}")
    print(f"Mean prediction error: {results['evaluation']['summary']['mean_error']:.2f} m")