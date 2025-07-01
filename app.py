"""
Streamlit web application for the Differentiable Trebuchet system.
Interactive interface for training and testing the physics-encoded neural network.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
import io
import base64

from simulator import ProjectileSimulator
from model import TrebuchetController
from trainer import TrebuchetTrainer

# Page configuration
st.set_page_config(
    page_title="Differentiable Trebuchet",
    page_icon="🏹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
}

.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

.info-box {
    background: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'simulator' not in st.session_state:
    st.session_state.simulator = ProjectileSimulator()
if 'controller' not in st.session_state:
    st.session_state.controller = TrebuchetController()
if 'trainer' not in st.session_state:
    st.session_state.trainer = TrebuchetTrainer(
        st.session_state.simulator, 
        st.session_state.controller
    )

# Main title
st.markdown("""
<div class="main-header">
    <h1>🏹 Differentiable Trebuchet: Physics-Encoded Neural Network</h1>
    <p>An inverse dynamics solver that learns optimal trebuchet parameters through differentiable physics simulation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🎛️ Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "🎯 Interactive Simulator", "🧠 Train Model", "📊 Model Analysis", "📚 Documentation"]
)

if page == "🏠 Home":
    # Home page with project overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Project Overview")
        st.markdown("""
        This application demonstrates **Differentiable Programming** by combining:
        
        1. **🧠 Neural Network Controller**: Learns to predict optimal trebuchet parameters
        2. **⚡ Physics Simulator**: Accurate projectile motion with air resistance and wind
        3. **🔄 Differentiable Bridge**: Gradients flow through physics to train the network
        
        ### How it Works
        
        Given a **target distance** and **wind conditions**, the system:
        - Uses a neural network to propose optimal parameters (counterweight mass, release angle)
        - Simulates the projectile trajectory using real physics equations
        - Compares achieved distance with target and backpropagates errors
        - Learns to make better predictions over time
        """)
        
        st.subheader("Key Features")
        features = [
            "🎯 **Inverse Problem Solving**: Find parameters to achieve specific outcomes",
            "⚙️ **Physics-Encoded Learning**: Neural networks constrained by physical laws",
            "🌪️ **Environmental Factors**: Accounts for wind speed and air resistance",
            "📊 **Interactive Visualization**: Real-time plotting and analysis",
            "🔬 **Scientific Accuracy**: Based on peer-reviewed research in differentiable programming"
        ]
        
        for feature in features:
            st.markdown(feature)
    
    with col2:
        st.subheader("Quick Stats")
        
        if st.session_state.model_trained:
            history = st.session_state.controller.get_training_history()
            eval_results = getattr(st.session_state.trainer, 'evaluation_results', {})
            
            if history:
                st.metric("Training Iterations", history.get('n_iter', 'N/A'))
                if history.get('loss_curve'):
                    final_loss = history['loss_curve'][-1]
                    st.metric("Final Training Loss", f"{final_loss:.6f}")
            
            if eval_results and 'summary' in eval_results:
                summary = eval_results['summary']
                st.metric("Mean Error", f"{summary['mean_error']:.2f} m")
                st.metric("Accuracy (±5m)", f"{summary['accuracy_within_5m']:.1f}%")
        else:
            st.info("Train a model to see performance statistics")
        
        st.subheader("System Status")
        st.success("✅ Physics Simulator Ready")
        st.success("✅ Neural Network Initialized")
        
        if st.session_state.model_trained:
            st.success("✅ Model Trained")
        else:
            st.warning("⏳ Model Not Trained")

elif page == "🎯 Interactive Simulator":
    # Interactive physics simulator
    st.header("🎯 Interactive Physics Simulator")
    st.markdown("Explore trebuchet physics with real-time parameter adjustment")
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trebuchet Parameters")
        mass = st.slider(
            "Counterweight Mass (kg)", 
            min_value=50.0, 
            max_value=500.0, 
            value=200.0, 
            step=10.0,
            help="Heavier counterweights provide more initial velocity"
        )
        
        angle = st.slider(
            "Release Angle (degrees)", 
            min_value=30.0, 
            max_value=60.0, 
            value=45.0, 
            step=1.0,
            help="Optimal angle balances height and distance"
        )
    
    with col2:
        st.subheader("Environmental Conditions")
        wind_speed = st.slider(
            "Wind Speed (m/s)", 
            min_value=-10.0, 
            max_value=10.0, 
            value=0.0, 
            step=0.5,
            help="Positive values are tailwind, negative are headwind"
        )
        
        show_trajectory = st.checkbox("Show Trajectory", value=True)
        show_physics_info = st.checkbox("Show Physics Information", value=True)
    
    # Update simulator
    st.session_state.simulator.wind_speed = wind_speed
    
    # Calculate and display results
    if st.button("🚀 Launch Simulation", type="primary"):
        with st.spinner("Calculating trajectory..."):
            # Run simulation
            t_points, trajectory, final_distance = st.session_state.simulator.simulate_trajectory(
                mass, angle
            )
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Achieved Distance", f"{final_distance:.1f} m")
            
            with col2:
                max_height = np.max(trajectory[:, 1])
                st.metric("Maximum Height", f"{max_height:.1f} m")
            
            with col3:
                flight_time = t_points[-1]
                st.metric("Flight Time", f"{flight_time:.2f} s")
            
            # Trajectory plot
            if show_trajectory:
                fig = go.Figure()
                
                # Trajectory line
                fig.add_trace(go.Scatter(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    mode='lines',
                    name='Trajectory',
                    line=dict(color='blue', width=3)
                ))
                
                # Launch point
                fig.add_trace(go.Scatter(
                    x=[0],
                    y=[2],
                    mode='markers',
                    name='Launch Point',
                    marker=dict(color='green', size=12, symbol='triangle-up')
                ))
                
                # Landing point
                fig.add_trace(go.Scatter(
                    x=[final_distance],
                    y=[0],
                    mode='markers',
                    name='Landing Point',
                    marker=dict(color='red', size=12, symbol='diamond')
                ))
                
                # Ground line
                fig.add_hline(y=0, line_dash="dash", line_color="brown", opacity=0.5)
                
                fig.update_layout(
                    title=f"Projectile Trajectory (Mass: {mass}kg, Angle: {angle}°, Wind: {wind_speed}m/s)",
                    xaxis_title="Horizontal Distance (m)",
                    yaxis_title="Height (m)",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Physics information
            if show_physics_info:
                st.subheader("Physics Analysis")
                
                # Calculate initial velocity
                initial_speed = np.sqrt(2 * 9.81 * 2.0 * mass / 1.0)
                v0_x = initial_speed * np.cos(np.deg2rad(angle))
                v0_y = initial_speed * np.sin(np.deg2rad(angle))
                
                physics_data = {
                    "Parameter": [
                        "Initial Speed", "Initial Horizontal Velocity", "Initial Vertical Velocity",
                        "Projectile Mass", "Air Density", "Drag Coefficient"
                    ],
                    "Value": [
                        f"{initial_speed:.2f} m/s", f"{v0_x:.2f} m/s", f"{v0_y:.2f} m/s",
                        "1.0 kg", "1.225 kg/m³", "0.47"
                    ]
                }
                
                st.table(pd.DataFrame(physics_data))

elif page == "🧠 Train Model":
    # Model training interface
    st.header("🧠 Neural Network Training")
    st.markdown("Train the physics-encoded neural network to learn optimal trebuchet parameters")
    
    # Training parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Configuration")
        
        n_samples = st.selectbox(
            "Number of Training Samples",
            [100, 500, 1000, 2000, 5000],
            index=2,
            help="More samples = better accuracy but longer training time"
        )
        
        strategy = st.selectbox(
            "Data Generation Strategy",
            ["random", "targeted"],
            help="Random: diverse conditions, Targeted: specific optimization scenarios"
        )
        
        max_iter = st.selectbox(
            "Maximum Training Iterations",
            [100, 200, 500, 1000],
            index=1,
            help="More iterations = better convergence but longer training time"
        )
    
    with col2:
        st.subheader("Model Architecture")
        
        # Model hyperparameters
        hidden_layers = st.selectbox(
            "Hidden Layer Configuration",
            ["(32, 32)", "(64, 64)", "(128, 64, 32)", "(64, 64, 64)"],
            index=1
        )
        
        learning_rate = st.selectbox(
            "Learning Rate",
            [0.0001, 0.001, 0.01, 0.1],
            index=1,
            format_func=lambda x: f"{x:.4f}"
        )
        
        activation = st.selectbox(
            "Activation Function",
            ["relu", "tanh", "logistic"],
            help="ReLU is typically best for this type of problem"
        )
    
    # Convert hidden layers string to tuple
    hidden_layer_sizes = eval(hidden_layers)
    
    # Update controller with new hyperparameters
    st.session_state.controller = TrebuchetController(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        activation=activation
    )
    
    st.session_state.trainer = TrebuchetTrainer(
        st.session_state.simulator,
        st.session_state.controller
    )
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Training info
        st.info(f"Training with {n_samples} samples using {strategy} strategy...")
        
        try:
            # Run training pipeline
            with st.spinner("Training neural network..."):
                progress_bar.progress(20)
                status_text.text("Generating training data...")
                
                results = st.session_state.trainer.full_training_pipeline(
                    n_samples=n_samples,
                    strategy=strategy,
                    create_visualizations=True
                )
                
                progress_bar.progress(100)
                status_text.text("Training completed!")
                
                st.session_state.model_trained = True
                
                # Display results
                st.success("🎉 Training completed successfully!")
                
                # Training metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Training Time", 
                        f"{results['training']['training_time']:.2f}s"
                    )
                
                with col2:
                    st.metric(
                        "Training R² Score", 
                        f"{results['training']['train_score']:.4f}"
                    )
                
                with col3:
                    st.metric(
                        "Validation R² Score", 
                        f"{results['training']['val_score']:.4f}"
                    )
                
                with col4:
                    st.metric(
                        "Mean Error", 
                        f"{results['evaluation']['summary']['mean_error']:.2f}m"
                    )
                
                # Training curve
                history = results['training']['history']
                if history.get('loss_curve'):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history['loss_curve'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='blue')
                    ))
                    
                    fig.update_layout(
                        title="Training Loss Convergence",
                        xaxis_title="Iteration",
                        yaxis_title="Loss",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            progress_bar.progress(0)
            status_text.text("Training failed!")
    
    # Quick test section
    if st.session_state.model_trained:
        st.subheader("🎯 Quick Model Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_distance = st.slider("Target Distance (m)", 50.0, 300.0, 150.0)
            test_wind = st.slider("Wind Speed (m/s)", -5.0, 5.0, 0.0)
        
        with col2:
            if st.button("🎯 Predict Parameters"):
                pred_mass, pred_angle = st.session_state.controller.predict_single(
                    test_distance, test_wind
                )
                
                # Verify prediction
                st.session_state.simulator.wind_speed = test_wind
                achieved_dist = st.session_state.simulator.calculate_distance_only(
                    pred_mass, pred_angle
                )
                
                st.metric("Predicted Mass", f"{pred_mass:.1f} kg")
                st.metric("Predicted Angle", f"{pred_angle:.1f}°")
                st.metric("Achieved Distance", f"{achieved_dist:.1f} m")
                st.metric("Prediction Error", f"{abs(achieved_dist - test_distance):.1f} m")

elif page == "📊 Model Analysis":
    # Model analysis and visualization
    st.header("📊 Model Analysis & Visualization")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Train Model' page")
        st.stop()
    
    # Analysis options
    analysis_type = st.selectbox(
        "Analysis Type",
        [
            "Parameter Sensitivity", 
            "Performance Metrics", 
            "Trajectory Comparison",
            "Error Analysis"
        ]
    )
    
    if analysis_type == "Parameter Sensitivity":
        st.subheader("Parameter Sensitivity Analysis")
        st.markdown("Explore how the model's predictions vary with different target conditions")
        
        # Create parameter heatmaps
        distance_range = np.linspace(50, 300, 25)
        wind_range = np.linspace(-5, 5, 15)
        
        with st.spinner("Generating sensitivity analysis..."):
            predicted_masses = np.zeros((len(wind_range), len(distance_range)))
            predicted_angles = np.zeros((len(wind_range), len(distance_range)))
            
            for i, wind in enumerate(wind_range):
                for j, dist in enumerate(distance_range):
                    mass, angle = st.session_state.controller.predict_single(dist, wind)
                    predicted_masses[i, j] = mass
                    predicted_angles[i, j] = angle
            
            # Mass sensitivity heatmap
            fig_mass = go.Figure()
            fig_mass.add_trace(go.Heatmap(
                z=predicted_masses,
                x=distance_range,
                y=wind_range,
                colorscale='Viridis',
                colorbar=dict(title='Mass (kg)')
            ))
            fig_mass.update_layout(
                title='Predicted Counterweight Mass Sensitivity',
                xaxis_title='Target Distance (m)',
                yaxis_title='Wind Speed (m/s)',
                template='plotly_white'
            )
            st.plotly_chart(fig_mass, use_container_width=True)
            
            # Angle sensitivity heatmap
            fig_angle = go.Figure()
            fig_angle.add_trace(go.Heatmap(
                z=predicted_angles,
                x=distance_range,
                y=wind_range,
                colorscale='Plasma',
                colorbar=dict(title='Angle (deg)')
            ))
            fig_angle.update_layout(
                title='Predicted Release Angle Sensitivity',
                xaxis_title='Target Distance (m)',
                yaxis_title='Wind Speed (m/s)',
                template='plotly_white'
            )
            st.plotly_chart(fig_angle, use_container_width=True)
    
    elif analysis_type == "Performance Metrics":
        st.subheader("Model Performance Metrics")
        
        # Get evaluation results
        eval_results = getattr(st.session_state.trainer, 'evaluation_results', {})
        
        if eval_results and 'summary' in eval_results:
            summary = eval_results['summary']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Error", f"{summary['mean_error']:.2f} m")
            with col2:
                st.metric("Std Error", f"{summary['std_error']:.2f} m")
            with col3:
                st.metric("Max Error", f"{summary['max_error']:.2f} m")
            with col4:
                st.metric("Mean Relative Error", f"{summary['mean_rel_error']:.1f}%")
            
            # Accuracy metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy (±5m)", f"{summary['accuracy_within_5m']:.1f}%")
            with col2:
                st.metric("Accuracy (±10m)", f"{summary['accuracy_within_10m']:.1f}%")
            
            # Error distribution
            fig_error = go.Figure()
            fig_error.add_trace(go.Histogram(
                x=eval_results['errors'],
                nbinsx=30,
                name='Absolute Error Distribution',
                opacity=0.7
            ))
            fig_error.update_layout(
                title='Prediction Error Distribution',
                xaxis_title='Absolute Error (m)',
                yaxis_title='Frequency',
                template='plotly_white'
            )
            st.plotly_chart(fig_error, use_container_width=True)
        else:
            st.warning("No evaluation results available. Please retrain the model.")
    
    elif analysis_type == "Trajectory Comparison":
        st.subheader("Trajectory Comparison")
        st.markdown("Compare trajectories for different target conditions")
        
        # User-defined conditions
        st.write("Define test conditions:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target1 = st.number_input("Target Distance 1 (m)", 50, 300, 100)
            wind1 = st.number_input("Wind Speed 1 (m/s)", -10.0, 10.0, 0.0)
        
        with col2:
            target2 = st.number_input("Target Distance 2 (m)", 50, 300, 150)
            wind2 = st.number_input("Wind Speed 2 (m/s)", -10.0, 10.0, 2.0)
        
        with col3:
            target3 = st.number_input("Target Distance 3 (m)", 50, 300, 200)
            wind3 = st.number_input("Wind Speed 3 (m/s)", -10.0, 10.0, -3.0)
        
        if st.button("Generate Trajectory Comparison"):
            conditions = [(target1, wind1), (target2, wind2), (target3, wind3)]
            colors = ['blue', 'red', 'green']
            
            fig = go.Figure()
            
            for i, (target_dist, wind_speed) in enumerate(conditions):
                # Get prediction
                pred_mass, pred_angle = st.session_state.controller.predict_single(
                    target_dist, wind_speed
                )
                
                # Simulate trajectory
                st.session_state.simulator.wind_speed = wind_speed
                t_points, trajectory, final_dist = st.session_state.simulator.simulate_trajectory(
                    pred_mass, pred_angle
                )
                
                # Plot trajectory
                fig.add_trace(go.Scatter(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    mode='lines',
                    name=f'Target: {target_dist}m, Wind: {wind_speed}m/s',
                    line=dict(color=colors[i], width=3)
                ))
                
                # Mark landing point
                fig.add_trace(go.Scatter(
                    x=[final_dist],
                    y=[0],
                    mode='markers',
                    marker=dict(color=colors[i], size=12, symbol='diamond'),
                    name=f'Landing: {final_dist:.1f}m',
                    showlegend=False
                ))
            
            fig.update_layout(
                title='Trajectory Comparison for Different Conditions',
                xaxis_title='Horizontal Distance (m)',
                yaxis_title='Height (m)',
                template='plotly_white',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif page == "📚 Documentation":
    # Documentation and information
    st.header("📚 Documentation")
    
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔬 Scientific Background", "⚙️ Technical Details", "📊 Data & Models", "🎯 Usage Guide"]
    )
    
    with tab1:
        st.subheader("Scientific Background")
        st.markdown("""
        ### Differentiable Programming
        
        This project demonstrates **Differentiable Programming**, a paradigm that bridges machine learning 
        and scientific computing by making entire computational pipelines differentiable.
        
        ### The Inverse Problem
        
        Traditional physics simulation is a **forward problem**: given parameters, predict outcomes.
        The inverse problem asks: given desired outcomes, what parameters should we use?
        
        ### Physics-Encoded Neural Networks (PeNNs)
        
        Our approach combines:
        - **Data-driven learning** from neural networks
        - **Physical constraints** from differential equations
        - **Differentiable simulation** for end-to-end training
        
        ### Citation
        
        Based on research in differentiable programming:
        > Innes, M., et al. (2019). A differentiable programming system to bridge 
        > machine learning and scientific computing. arXiv:1907.07587
        """)
    
    with tab2:
        st.subheader("Technical Implementation")
        st.markdown("""
        ### System Architecture
        
        1. **Physics Simulator** (`simulator.py`)
           - Implements projectile motion ODEs
           - Accounts for air resistance and wind
           - Uses Runge-Kutta numerical integration
        
        2. **Neural Network Controller** (`model.py`)
           - Scikit-learn MLPRegressor
           - Input: [target_distance, wind_speed]
           - Output: [counterweight_mass, release_angle]
        
        3. **Training System** (`trainer.py`)
           - Data generation strategies
           - Model training and evaluation
           - Comprehensive visualization
        
        ### Key Physics Equations
        
        **Projectile Motion with Air Resistance:**
        ```
        dx/dt = vx
        dy/dt = vy
        dvx/dt = -0.5 * ρ * v² * Cd * A * (vx - wind) / (m * |v|)
        dvy/dt = -g - 0.5 * ρ * v² * Cd * A * vy / (m * |v|)
        ```
        
        **Trebuchet Energy Conversion:**
        ```
        Initial Velocity = √(2 * g * h * M_counterweight / M_projectile)
        ```
        """)
    
    with tab3:
        st.subheader("Data & Model Information")
        
        if st.session_state.model_trained:
            # Model architecture
            st.write("**Current Model Architecture:**")
            controller = st.session_state.controller
            model_info = {
                "Hidden Layers": str(controller.hidden_layer_sizes),
                "Activation Function": controller.activation,
                "Learning Rate": controller.learning_rate_init,
                "Max Iterations": controller.max_iter,
                "Random State": controller.random_state
            }
            st.table(pd.DataFrame(list(model_info.items()), columns=["Parameter", "Value"]))
            
            # Training history
            history = controller.get_training_history()
            if history:
                st.write("**Training Statistics:**")
                training_stats = {
                    "Training Iterations": history.get('n_iter', 'N/A'),
                    "Final Loss": f"{history.get('loss_curve', [0])[-1]:.6f}" if history.get('loss_curve') else 'N/A',
                    "Best Validation Score": f"{history.get('best_validation_score', 0):.6f}" if history.get('best_validation_score') else 'N/A'
                }
                st.table(pd.DataFrame(list(training_stats.items()), columns=["Metric", "Value"]))
        else:
            st.info("Train a model to see detailed information")
        
        st.write("**Physical Constants:**")
        constants = {
            "Gravitational Acceleration": "9.81 m/s²",
            "Air Density (Sea Level)": "1.225 kg/m³",
            "Projectile Drag Coefficient": "0.47 (sphere)",
            "Projectile Radius": "0.05 m",
            "Projectile Mass": "1.0 kg",
            "Trebuchet Height": "2.0 m"
        }
        st.table(pd.DataFrame(list(constants.items()), columns=["Constant", "Value"]))
    
    with tab4:
        st.subheader("Usage Guide")
        st.markdown("""
        ### Getting Started
        
        1. **🎯 Interactive Simulator**: Experiment with physics parameters
        2. **🧠 Train Model**: Generate data and train the neural network
        3. **📊 Model Analysis**: Analyze model performance and behavior
        
        ### Best Practices
        
        **For Training:**
        - Start with 1000+ samples for good performance
        - Use "random" strategy for diverse training data
        - Monitor both training and validation scores
        
        **For Analysis:**
        - Check parameter sensitivity heatmaps
        - Verify predictions with trajectory comparisons
        - Monitor error distributions for outliers
        
        ### Troubleshooting
        
        **Poor Model Performance:**
        - Increase training samples
        - Adjust learning rate
        - Try different activation functions
        
        **Slow Training:**
        - Reduce number of samples for testing
        - Lower maximum iterations
        - Use simpler network architecture
        
        ### Extensions
        
        This framework can be extended to:
        - Different projectile types (varying mass, drag)
        - Complex wind patterns (varying with height)
        - Multiple objectives (accuracy + minimum energy)
        - Real-time control applications
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🏹 Differentiable Trebuchet v1.0 | Built with Streamlit, Scikit-learn & Physics
        <br>
        <a href='https://github.com/Sakeeb91/Inverse-Dynamics-Solver' target='_blank'>View Source Code</a>
    </div>
    """, 
    unsafe_allow_html=True
)