# Visualization Documentation

This document tracks all visualizations, plots, and animations generated by the Differentiable Trebuchet project.

## 📊 Training Visualizations

### Loss Convergence Plots
- **Location**: `visualizations/plots/training_loss.html`
- **Description**: Interactive Plotly chart showing model loss convergence over training iterations
- **Features**:
  - Real-time updates during training
  - Hover tooltips with exact values
  - Zoom and pan capabilities
  - Export functionality

### Parameter Convergence
- **Location**: `visualizations/plots/parameter_convergence.html`
- **Description**: Shows how predicted parameters (counterweight mass, release angle) converge to optimal values
- **Features**:
  - Dual y-axis for different parameter scales
  - Color-coded parameter tracking
  - Target lines for reference

### Prediction Accuracy
- **Location**: `visualizations/plots/prediction_accuracy.html`
- **Description**: Comparison between predicted distance and target distance over training
- **Features**:
  - Scatter plot with trend line
  - Error bounds visualization
  - Final accuracy metrics display

## 🎯 Physics Simulation Visualizations

### Trajectory Plots
- **Location**: `visualizations/plots/trajectories/`
- **Description**: 2D trajectory plots showing projectile path under different conditions
- **Features**:
  - Wind effect visualization
  - Multiple trajectory comparison
  - Physics parameter annotations

### Parameter Sensitivity Analysis
- **Location**: `visualizations/plots/sensitivity_analysis.html`
- **Description**: Heatmaps showing how distance varies with parameter changes
- **Features**:
  - Interactive parameter sliders
  - Contour plots for optimization landscape
  - Gradient visualization

## 🎬 Animations

### Training Progress Animation
- **Location**: `visualizations/animations/training_progress.gif`
- **Description**: Animated visualization of training progress over time
- **Features**:
  - Parameter evolution over iterations
  - Loss surface exploration
  - Convergence visualization

### Projectile Motion Animation
- **Location**: `visualizations/animations/projectile_motion.gif`
- **Description**: Real-time animation of projectile flight with different parameters
- **Features**:
  - Physics-accurate motion
  - Wind effect visualization
  - Trajectory comparison

## 📈 Performance Metrics

### Model Performance Dashboard
- **Location**: `visualizations/plots/performance_dashboard.html`
- **Description**: Comprehensive dashboard with all key metrics
- **Features**:
  - Training time analysis
  - Accuracy vs complexity trade-offs
  - Resource utilization metrics

## 🔧 Technical Specifications

All visualizations are generated using:
- **Plotly**: Interactive web-based plots
- **Matplotlib**: Static high-quality plots
- **Custom Animation Engine**: Physics-based animations

### File Naming Convention
- Training plots: `training_[metric]_[timestamp].html`
- Physics plots: `physics_[type]_[parameters].html`
- Animations: `[type]_animation_[timestamp].gif`

### Data Sources
All visualizations pull data from:
- Training logs: Real-time training metrics
- Simulation results: Physics computation outputs
- User interactions: Streamlit app parameter changes

---

*Last updated: [Auto-generated timestamp]*