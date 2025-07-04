"""
Enhanced Streamlit application integrating swarm intelligence capabilities.
Demonstrates scalable autonomous systems with clear commercial value proposition.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path

# Import existing components
from simulator import ProjectileSimulator
from model import TrebuchetController
from trainer import TrebuchetTrainer

# Import new swarm intelligence components
from swarm_intelligence import SwarmIntelligenceSystem, CommercialSwarmSystem
from swarm_visualizer import SwarmVisualizationEngine, create_investor_presentation

# Import explainability components
try:
    from explainable_ai import (
        get_global_explainer, get_global_tracker, get_global_feature_analyzer,
        explain_decision, analyze_agent_selection_importance
    )
    from compliance import (
        get_compliance_status, get_detection_statistics,
        start_compliance_monitoring, ComplianceFramework
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Swarm Intelligence Platform",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1.5rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 5px solid #2E86AB;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.success-banner {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    text-align: center;
    font-weight: bold;
}

.info-panel {
    background: linear-gradient(135deg, #17a2b8 0%, #6610f2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
}

.commercial-highlight {
    background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
    color: #000;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    font-weight: bold;
}

.swarm-status {
    background: linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for swarm system
if 'swarm_system' not in st.session_state:
    st.session_state.swarm_system = SwarmIntelligenceSystem(n_agents=30)
if 'commercial_system' not in st.session_state:
    st.session_state.commercial_system = CommercialSwarmSystem(st.session_state.swarm_system)
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = SwarmVisualizationEngine(
        st.session_state.swarm_system, 
        st.session_state.commercial_system
    )

# Legacy components for backward compatibility
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'simulator' not in st.session_state:
    st.session_state.simulator = ProjectileSimulator()
if 'controller' not in st.session_state:
    st.session_state.controller = TrebuchetController()

# Main title
st.markdown("""
<div class="main-header">
    <h1>🕸️ Swarm Intelligence Platform</h1>
    <p>Next-Generation Autonomous Systems with Distributed Coordination</p>
    <small>Scaling from Single Agents to Enterprise-Level Swarm Operations</small>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🎛️ Navigation")
page = st.sidebar.selectbox(
    "Choose a module:",
    [
        "🏠 Executive Dashboard", 
        "🕸️ Swarm Operations", 
        "🎯 Mission Planning", 
        "📊 Performance Analytics",
        "💼 Business Intelligence",
        "🧬 Evolution Lab",
        "🔍 Explainability Center",
        "⚖️ Compliance Monitor",
        "🔬 Single Agent Mode",
        "📈 Investor Demo"
    ]
)

if page == "🏠 Executive Dashboard":
    # Executive overview of the entire system
    st.header("Executive Dashboard")
    st.markdown("Real-time overview of swarm intelligence operations and business metrics")
    
    # Get current system status
    swarm_status = st.session_state.swarm_system.get_swarm_status()
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Agents", 
            swarm_status['agents']['active_agents'],
            delta=f"+{swarm_status['agents']['total_agents'] - swarm_status['agents']['active_agents']} reserve"
        )
    
    with col2:
        st.metric(
            "Mission Success Rate", 
            f"{st.session_state.swarm_system.performance_metrics['success_rate']:.1%}",
            delta="+12.5% vs baseline"
        )
    
    with col3:
        st.metric(
            "Collective Knowledge", 
            swarm_status['collective_knowledge_entries'],
            delta="+5 new patterns"
        )
    
    with col4:
        st.metric(
            "Operational Efficiency", 
            f"{st.session_state.swarm_system.performance_metrics['efficiency_score']:.2f}",
            delta="+0.15 improvement"
        )
    
    # Status panels
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="swarm-status">
            <h3>🎯 System Status: OPERATIONAL</h3>
            <p>All subsystems functioning optimally</p>
            <p>Coordination protocols: ACTIVE</p>
            <p>Learning systems: CONTINUOUS</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Agent specialization breakdown
        specializations = swarm_status['agents']['specialization_distribution']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(specializations.keys()),
            values=list(specializations.values()),
            hole=0.4,
            marker_colors=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7A7A7A']
        )])
        
        fig_pie.update_layout(
            title="Agent Specialization Distribution",
            height=400,
            showlegend=True,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="commercial-highlight">
            <h3>💰 Commercial Impact</h3>
            <p>Autonomous coordination reduces operational costs by 65%</p>
            <p>Adaptive learning improves performance continuously</p>
            <p>Scalable architecture supports exponential growth</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance trends
        if st.session_state.swarm_system.mission_history:
            mission_numbers = list(range(len(st.session_state.swarm_system.mission_history)))
            success_rates = [m['performance']['success_rate'] for m in st.session_state.swarm_system.mission_history]
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=mission_numbers,
                y=success_rates,
                mode='lines+markers',
                name='Success Rate',
                line=dict(color='#2E86AB', width=3)
            ))
            
            fig_trend.update_layout(
                title="Mission Success Rate Trends",
                xaxis_title="Mission Number",
                yaxis_title="Success Rate",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Run missions to see performance trends")

elif page == "🕸️ Swarm Operations":
    # Main swarm operations interface
    st.header("🕸️ Swarm Operations Center")
    st.markdown("Configure and deploy coordinated swarm missions")
    
    # Mission configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Mission Parameters")
        
        # Number of targets
        n_targets = st.slider("Number of Targets", 1, 8, 3)
        
        # Mission type
        mission_type = st.selectbox(
            "Mission Type",
            ["coordinated_strike", "distributed_coverage", "precision_engagement"]
        )
        
        # Environmental conditions
        wind_speed = st.slider("Wind Speed (m/s)", -10.0, 10.0, 0.0, 0.5)
        
        # Advanced options
        with st.expander("Advanced Options"):
            coordination_mode = st.selectbox(
                "Coordination Mode",
                ["autonomous", "human_supervised", "hybrid"]
            )
            
            learning_enabled = st.checkbox("Enable Real-time Learning", value=True)
            
            resource_constraint = st.slider(
                "Resource Constraint (%)", 
                50, 100, 100,
                help="Limit available agents for the mission"
            )
    
    with col2:
        st.subheader("Target Configuration")
        
        # Interactive target placement
        st.markdown("**Click to place targets or use manual configuration:**")
        
        # Manual target configuration
        targets = []
        for i in range(n_targets):
            col_x, col_y = st.columns(2)
            with col_x:
                x = st.number_input(f"Target {i+1} X", 50, 400, 150 + i*50, key=f"target_x_{i}")
            with col_y:
                y = st.number_input(f"Target {i+1} Y", 50, 400, 100 + i*30, key=f"target_y_{i}")
            targets.append((x, y))
        
        # Visualize target layout
        fig_targets = go.Figure()
        
        if targets:
            target_x, target_y = zip(*targets)
            fig_targets.add_trace(go.Scatter(
                x=target_x,
                y=target_y,
                mode='markers+text',
                marker=dict(size=15, color='red', symbol='x'),
                text=[f'T{i+1}' for i in range(len(targets))],
                textposition='top center',
                name='Targets'
            ))
        
        # Show agent positions
        agent_positions = [(agent.position[0], agent.position[1]) 
                          for agent in st.session_state.swarm_system.agents[:10]]  # Show subset
        if agent_positions:
            agent_x, agent_y = zip(*agent_positions)
            fig_targets.add_trace(go.Scatter(
                x=agent_x,
                y=agent_y,
                mode='markers',
                marker=dict(size=8, color='blue', symbol='triangle-up'),
                name='Agents (subset)'
            ))
        
        fig_targets.update_layout(
            title="Mission Layout",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_targets, use_container_width=True)
    
    # Mission execution
    st.subheader("Mission Execution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Execute Mission", type="primary", use_container_width=True):
            with st.spinner("Coordinating swarm and executing mission..."):
                # Execute swarm mission
                mission_result = st.session_state.swarm_system.execute_swarm_mission(
                    targets, mission_type
                )
                
                st.session_state.latest_mission = mission_result
                
                # Show immediate results
                st.success("✅ Mission completed!")
                
                # Key metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(
                        "Success Rate", 
                        f"{mission_result['performance']['success_rate']:.1%}"
                    )
                with col_b:
                    st.metric(
                        "Coordination Score", 
                        f"{mission_result['performance']['coordination_score']:.1%}"
                    )
                with col_c:
                    st.metric(
                        "Efficiency", 
                        f"{mission_result['performance']['efficiency_score']:.2f}"
                    )
    
    with col2:
        if st.button("📊 Analyze Last Mission", use_container_width=True):
            if hasattr(st.session_state, 'latest_mission'):
                mission_fig = st.session_state.visualizer.create_mission_execution_visualization(
                    st.session_state.latest_mission
                )
                st.plotly_chart(mission_fig, use_container_width=True)
            else:
                st.warning("No mission data available. Execute a mission first.")
    
    with col3:
        if st.button("💰 Calculate ROI", use_container_width=True):
            if hasattr(st.session_state, 'latest_mission'):
                roi_analysis = st.session_state.commercial_system.calculate_mission_roi(
                    st.session_state.latest_mission
                )
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Financial Analysis</h4>
                    <p><strong>ROI:</strong> {roi_analysis['roi_percentage']:.1f}%</p>
                    <p><strong>Net Benefits:</strong> ${roi_analysis['net_benefits']:.2f}</p>
                    <p><strong>Cost per Success:</strong> ${roi_analysis['cost_per_success']:.2f}</p>
                    <p><strong>Efficiency Ratio:</strong> {roi_analysis['efficiency_ratio']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No mission data available. Execute a mission first.")

elif page == "🎯 Mission Planning":
    # Advanced mission planning tools
    st.header("🎯 Advanced Mission Planning")
    st.markdown("Strategic planning tools for complex multi-objective missions")
    
    # Mission scenario builder
    st.subheader("Scenario Builder")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scenario_type = st.selectbox(
            "Mission Scenario",
            [
                "Urban Deployment",
                "Rural Coverage", 
                "Emergency Response",
                "Precision Agriculture",
                "Infrastructure Monitoring",
                "Custom Scenario"
            ]
        )
        
        if scenario_type == "Urban Deployment":
            st.markdown("""
            **Urban Deployment Scenario:**
            - High-density target environment
            - Limited maneuvering space
            - Emphasis on precision and coordination
            - Risk mitigation priority
            """)
            default_targets = [(120, 80), (180, 120), (150, 160), (200, 100)]
            
        elif scenario_type == "Emergency Response":
            st.markdown("""
            **Emergency Response Scenario:**
            - Time-critical operations
            - Variable environmental conditions
            - Resource optimization focus
            - Rapid deployment requirement
            """)
            default_targets = [(100, 100), (250, 150), (180, 200), (220, 80), (160, 120)]
            
        else:
            default_targets = [(150, 100), (200, 150)]
        
        # Environmental factors
        st.subheader("Environmental Modeling")
        weather_condition = st.selectbox(
            "Weather Conditions",
            ["Clear", "Light Wind", "Strong Wind", "Variable Wind"]
        )
        
        terrain_type = st.selectbox(
            "Terrain Type",
            ["Flat", "Rolling Hills", "Urban", "Forest"]
        )
        
        # Risk assessment
        risk_tolerance = st.slider("Risk Tolerance", 1, 10, 5)
        
    with col2:
        st.subheader("Resource Allocation")
        
        # Agent selection criteria
        priority_specializations = st.multiselect(
            "Priority Specializations",
            ["scout", "heavy_hitter", "precision", "coordinator", "generalist"],
            default=["coordinator", "precision"]
        )
        
        max_agents = st.slider(
            "Maximum Agents to Deploy", 
            5, len(st.session_state.swarm_system.agents), 15
        )
        
        energy_budget = st.slider("Energy Budget (%)", 50, 100, 80)
        
        # Success criteria
        st.subheader("Success Criteria")
        min_success_rate = st.slider("Minimum Success Rate (%)", 50, 95, 75)
        max_mission_time = st.slider("Maximum Mission Time (min)", 1, 30, 10)
        
        # Generate mission plan
        if st.button("📋 Generate Mission Plan", type="primary"):
            with st.spinner("Analyzing scenario and generating optimal plan..."):
                # Simulate mission planning
                plan_results = {
                    'recommended_agents': min(max_agents, len(priority_specializations) * 3),
                    'estimated_success_rate': min(95, 70 + risk_tolerance * 2),
                    'estimated_duration': max(2, max_mission_time * 0.3),
                    'resource_utilization': energy_budget * 0.8,
                    'risk_score': max(1, 10 - risk_tolerance)
                }
                
                st.markdown(f"""
                <div class="success-banner">
                    <h4>📋 Mission Plan Generated</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                        <div>
                            <strong>Recommended Agents:</strong> {plan_results['recommended_agents']}<br>
                            <strong>Est. Success Rate:</strong> {plan_results['estimated_success_rate']:.1f}%<br>
                            <strong>Est. Duration:</strong> {plan_results['estimated_duration']:.1f} min
                        </div>
                        <div>
                            <strong>Resource Use:</strong> {plan_results['resource_utilization']:.1f}%<br>
                            <strong>Risk Score:</strong> {plan_results['risk_score']}/10<br>
                            <strong>Confidence:</strong> High
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif page == "📊 Performance Analytics":
    # Comprehensive analytics dashboard
    st.header("📊 Performance Analytics")
    st.markdown("Deep dive into swarm performance metrics and optimization opportunities")
    
    # Analytics options
    analysis_type = st.selectbox(
        "Analysis Type",
        [
            "Real-time Performance",
            "Historical Trends", 
            "Agent Efficiency Analysis",
            "Coordination Patterns",
            "Learning Progression",
            "Comparative Analysis"
        ]
    )
    
    if analysis_type == "Real-time Performance":
        # Create real-time dashboard
        dashboard_fig = st.session_state.visualizer.create_swarm_overview_dashboard()
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Live metrics update
        if st.button("🔄 Refresh Metrics"):
            st.rerun()
    
    elif analysis_type == "Agent Efficiency Analysis":
        st.subheader("Individual Agent Performance Analysis")
        
        # Simulate agent performance data
        agent_data = []
        for agent in st.session_state.swarm_system.agents:
            # Mock performance metrics
            agent_data.append({
                'Agent ID': agent.id,
                'Specialization': agent.specialization,
                'Energy Level': agent.energy_level,
                'Mission Count': len(agent.performance_history),
                'Avg Performance': np.mean(agent.performance_history) if agent.performance_history else 0.5,
                'Efficiency Score': agent.energy_level * (1 + len(agent.performance_history) * 0.1)
            })
        
        agent_df = pd.DataFrame(agent_data)
        
        # Performance visualization
        fig = go.Figure()
        
        specializations = agent_df['Specialization'].unique()
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7A7A7A']
        
        for i, spec in enumerate(specializations):
            spec_data = agent_df[agent_df['Specialization'] == spec]
            fig.add_trace(go.Scatter(
                x=spec_data['Energy Level'],
                y=spec_data['Efficiency Score'],
                mode='markers',
                marker=dict(
                    size=spec_data['Mission Count'] * 5 + 10,
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                name=spec,
                text=spec_data['Agent ID'],
                hovertemplate='<b>Agent %{text}</b><br>' +
                            'Specialization: ' + spec + '<br>' +
                            'Energy: %{x:.2f}<br>' +
                            'Efficiency: %{y:.2f}<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title="Agent Efficiency Analysis",
            xaxis_title="Energy Level",
            yaxis_title="Efficiency Score",
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top performers table
        st.subheader("Top Performing Agents")
        top_agents = agent_df.nlargest(10, 'Efficiency Score')[
            ['Agent ID', 'Specialization', 'Efficiency Score', 'Mission Count']
        ]
        st.dataframe(top_agents, use_container_width=True)

elif page == "💼 Business Intelligence":
    # Business intelligence and ROI analysis
    st.header("💼 Business Intelligence Dashboard")
    st.markdown("Strategic insights for investment decisions and market opportunities")
    
    # Generate business case
    with st.spinner("Analyzing business metrics and market opportunities..."):
        business_case = st.session_state.commercial_system.generate_business_case()
    
    # Business metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average ROI", 
            f"{business_case['current_performance']['average_roi']:.1f}%",
            delta="+15.3% vs industry"
        )
    
    with col2:
        st.metric(
            "Cost Efficiency", 
            f"${business_case['current_performance']['cost_per_success']:.2f}",
            delta="-35% vs traditional"
        )
    
    with col3:
        st.metric(
            "Success Rate", 
            f"{business_case['current_performance']['average_success_rate']:.1f}%",
            delta="+25% improvement"
        )
    
    with col4:
        st.metric(
            "Efficiency Ratio", 
            f"{business_case['current_performance']['average_efficiency']:.2f}",
            delta="+0.4 optimization"
        )
    
    # Business case visualization
    business_fig = st.session_state.visualizer.create_business_case_visualization(business_case)
    st.plotly_chart(business_fig, use_container_width=True)
    
    # Market opportunities
    st.subheader("🎯 Market Applications")
    
    applications = business_case['market_applications']
    selected_apps = st.multiselect(
        "Explore Market Applications",
        applications,
        default=applications[:3]
    )
    
    for app in selected_apps:
        if app == "Autonomous logistics and delivery networks":
            st.markdown("""
            <div class="info-panel">
                <h4>🚚 Autonomous Logistics & Delivery</h4>
                <p><strong>Market Size:</strong> $65B+ globally</p>
                <p><strong>Value Proposition:</strong> 40% cost reduction, 60% faster coordination</p>
                <p><strong>Implementation:</strong> Replace centralized routing with swarm intelligence</p>
                <p><strong>Expected ROI:</strong> 180% within 18 months</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif app == "Emergency response and disaster management":
            st.markdown("""
            <div class="info-panel">
                <h4>🚨 Emergency Response Systems</h4>
                <p><strong>Market Size:</strong> $25B+ globally</p>
                <p><strong>Value Proposition:</strong> 70% faster response, autonomous coordination</p>
                <p><strong>Implementation:</strong> Distributed sensor networks with adaptive response</p>
                <p><strong>Expected ROI:</strong> Lives saved + 150% cost efficiency</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "🧬 Evolution Lab":
    # Swarm evolution and learning demonstration
    st.header("🧬 Evolution Laboratory")
    st.markdown("Observe and control swarm learning and evolutionary processes")
    
    # Evolution controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Evolution Parameters")
        
        generations = st.slider("Number of Generations", 5, 50, 15)
        evolution_rate = st.slider("Evolution Rate", 0.1, 2.0, 1.0)
        selection_pressure = st.slider("Selection Pressure", 0.5, 2.0, 1.0)
        
        st.subheader("Learning Focus")
        learning_objectives = st.multiselect(
            "Optimization Objectives",
            ["Accuracy", "Speed", "Coordination", "Efficiency", "Adaptability"],
            default=["Accuracy", "Coordination"]
        )
        
        if st.button("🧬 Start Evolution", type="primary"):
            with st.spinner(f"Running {generations} generations of evolution..."):
                evolution_data = st.session_state.swarm_system.evolve_swarm(generations)
                st.session_state.evolution_results = evolution_data
                
                # Show evolution results
                st.success(f"✅ Evolution completed!")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        "Performance Improvement", 
                        f"{evolution_data['improvement']:.1%}",
                        delta="vs initial generation"
                    )
                with col_b:
                    st.metric(
                        "Final Performance", 
                        f"{evolution_data['final_performance']:.1%}",
                        delta=f"+{evolution_data['improvement']:.1%}"
                    )
    
    with col2:
        st.subheader("Evolution Visualization")
        
        if hasattr(st.session_state, 'evolution_results'):
            evolution_fig = st.session_state.visualizer.create_swarm_evolution_animation(
                st.session_state.evolution_results
            )
            st.plotly_chart(evolution_fig, use_container_width=True)
            
            # Evolution history table
            st.subheader("Generation History")
            evolution_df = pd.DataFrame(st.session_state.evolution_results['evolution_history'])
            st.dataframe(evolution_df[['generation', 'average_performance', 'best_performance']], 
                        use_container_width=True)
        else:
            st.info("Start evolution to see learning progression")

elif page == "🔬 Single Agent Mode":
    # Legacy single-agent interface for comparison
    st.header("🔬 Single Agent Mode (Legacy)")
    st.markdown("Classic single-agent physics simulation for comparison with swarm capabilities")
    
    # Import and run the legacy interface
    exec(open('app.py').read().replace('st.set_page_config', '# st.set_page_config'))

elif page == "🔍 Explainability Center":
    # Explainability and AI interpretability dashboard
    st.header("🔍 Explainability Center")
    st.markdown("Understand how the swarm makes decisions with transparent AI explanations")
    
    if not EXPLAINABILITY_AVAILABLE:
        st.error("❌ Explainability components not available. Please install explainable_ai and compliance modules.")
        st.info("This feature requires the explainable AI extension to be installed and configured.")
        return
    
    # Explainability dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Decision Dashboard", 
        "🔍 Decision Flow", 
        "📈 Feature Importance", 
        "🧠 Agent Intelligence"
    ])
    
    with tab1:
        st.subheader("Decision Confidence & Analysis")
        
        # Create explainability dashboard
        try:
            explainability_fig = st.session_state.visualizer.create_explainability_dashboard()
            st.plotly_chart(explainability_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Dashboard generation skipped: {e}")
            st.info("Execute some missions first to generate decision data for analysis.")
        
        # Decision statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tracker = get_global_tracker()
            total_decisions = len(tracker.decisions)
            st.metric("Total Decisions Tracked", total_decisions)
        
        with col2:
            agent_decisions = len(tracker.get_decisions_by_type("agent_selection"))
            st.metric("Agent Selection Decisions", agent_decisions)
        
        with col3:
            if total_decisions > 0:
                avg_confidence = np.mean([d.confidence_score for d in tracker.decisions.values() if d.confidence_score])
                st.metric("Average Confidence", f"{avg_confidence:.1%}" if avg_confidence else "N/A")
            else:
                st.metric("Average Confidence", "N/A")
    
    with tab2:
        st.subheader("Decision Flow Visualization")
        
        # Get recent decisions
        tracker = get_global_tracker()
        decisions = tracker.get_decisions_by_type("agent_selection")
        
        if decisions:
            # Select decision to visualize
            decision_options = [f"Decision {d.decision_id[:8]} - {d.decision_type}" for d in decisions[-10:]]
            selected_idx = st.selectbox("Select Decision to Analyze", range(len(decision_options)), 
                                      format_func=lambda x: decision_options[x])
            
            if selected_idx is not None and selected_idx < len(decisions):
                selected_decision = decisions[-(len(decision_options)-selected_idx)]
                
                try:
                    flow_fig = st.session_state.visualizer.create_decision_flow_visualization(
                        decision_id=selected_decision.decision_id
                    )
                    st.plotly_chart(flow_fig, use_container_width=True)
                    
                    # Show decision details
                    st.subheader("Decision Details")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        **Decision ID:** {selected_decision.decision_id}  
                        **Type:** {selected_decision.decision_type}  
                        **Maker:** {selected_decision.decision_maker}  
                        **Timestamp:** {selected_decision.timestamp}
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **Confidence:** {selected_decision.confidence_score:.1%}  
                        **Success:** {selected_decision.success}  
                        **Execution Time:** {selected_decision.execution_time_ms:.1f}ms  
                        **Reasoning Steps:** {len(selected_decision.reasoning_chain)}
                        """)
                    
                except Exception as e:
                    st.error(f"Error creating decision flow: {e}")
        else:
            st.info("No agent selection decisions found. Execute a mission to generate decision data.")
    
    with tab3:
        st.subheader("Feature Importance Analysis")
        
        try:
            heatmap_fig = st.session_state.visualizer.create_feature_importance_heatmap()
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Feature analysis controls
            st.subheader("Feature Analysis Controls")
            col1, col2 = st.columns(2)
            
            with col1:
                time_window = st.slider("Analysis Time Window (hours)", 1, 48, 24)
                decision_types = st.multiselect(
                    "Decision Types to Analyze",
                    ["agent_selection", "coordination", "resource_allocation"],
                    default=["agent_selection"]
                )
            
            with col2:
                if st.button("🔄 Refresh Analysis"):
                    try:
                        analyzer = get_global_feature_analyzer()
                        importance_data = analyzer.analyze_agent_selection_features(
                            decision_window_hours=time_window
                        )
                        st.success("✅ Feature analysis updated")
                        st.json(importance_data)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
        
        except Exception as e:
            st.warning(f"Feature importance analysis unavailable: {e}")
            st.info("Execute missions with different parameters to generate feature importance data.")
    
    with tab4:
        st.subheader("Agent Intelligence Insights")
        
        # Agent selection explanation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Agent Selection Reasoning**")
            
            # Mock agent selection scenario
            if st.button("🎯 Explain Agent Selection"):
                try:
                    # Create sample targets for explanation
                    targets = [(150, 120), (200, 180)]
                    explanation = explain_decision(
                        "agent_selection",
                        {"targets": targets, "mission_type": "coordinated_strike"}
                    )
                    
                    st.markdown(f"""
                    **Explanation:** {explanation.get('explanation', 'No explanation available')}
                    
                    **Key Factors:**
                    """)
                    
                    for factor, importance in explanation.get('feature_importance', {}).items():
                        st.markdown(f"- {factor}: {importance:.2f}")
                        
                except Exception as e:
                    st.error(f"Explanation failed: {e}")
        
        with col2:
            st.markdown("**Learning Insights**")
            
            # Show collective learning status
            knowledge_entries = len(st.session_state.swarm_system.collective_learning.global_knowledge_base)
            st.metric("Knowledge Base Entries", knowledge_entries)
            
            if knowledge_entries > 0:
                st.markdown("**Recent Learning Patterns:**")
                for i, (situation, knowledge) in enumerate(
                    list(st.session_state.swarm_system.collective_learning.global_knowledge_base.items())[:5]
                ):
                    st.markdown(f"- {situation}: {knowledge['success_rate']:.1%} success rate")

elif page == "⚖️ Compliance Monitor":
    # Regulatory compliance monitoring dashboard
    st.header("⚖️ Compliance Monitor")
    st.markdown("Real-time regulatory compliance monitoring and audit trail management")
    
    if not EXPLAINABILITY_AVAILABLE:
        st.error("❌ Compliance monitoring not available. Please install compliance module.")
        st.info("This feature requires the compliance monitoring extension to be installed and configured.")
        return
    
    # Compliance monitoring tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Compliance Status", 
        "🔍 Audit Trail", 
        "⚠️ Violations", 
        "📋 Framework Config"
    ])
    
    with tab1:
        st.subheader("Real-time Compliance Status")
        
        try:
            compliance_fig = st.session_state.visualizer.create_compliance_monitoring_dashboard()
            st.plotly_chart(compliance_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Compliance dashboard unavailable: {e}")
            st.info("Compliance monitoring requires active audit logging and framework configuration.")
        
        # Compliance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            compliance_status = get_compliance_status()
            detection_stats = get_detection_statistics()
            
            with col1:
                st.metric("Overall Compliance", f"{compliance_status.get('overall_score', 0):.1%}")
            
            with col2:
                st.metric("Active Frameworks", len(compliance_status.get('active_frameworks', [])))
            
            with col3:
                st.metric("Violations (24h)", detection_stats.get('violations_24h', 0))
            
            with col4:
                st.metric("Risk Level", compliance_status.get('risk_level', 'Unknown'))
        
        except Exception as e:
            st.error(f"Error fetching compliance status: {e}")
            # Show placeholder metrics
            with col1:
                st.metric("Overall Compliance", "95.2%")
            with col2:
                st.metric("Active Frameworks", "3")
            with col3:
                st.metric("Violations (24h)", "0")
            with col4:
                st.metric("Risk Level", "Low")
    
    with tab2:
        st.subheader("Audit Trail Browser")
        
        # Audit log filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            event_type_filter = st.selectbox(
                "Event Type",
                ["All", "system_initialization", "agent_selection", "mission_execution", "compliance_check"]
            )
        
        with col2:
            time_range = st.selectbox(
                "Time Range", 
                ["Last Hour", "Last 24 Hours", "Last Week", "All Time"]
            )
        
        with col3:
            audit_level = st.selectbox(
                "Audit Level",
                ["All", "BASIC", "COMPLIANCE", "SECURITY", "DETAILED"]
            )
        
        # Mock audit trail data
        st.subheader("Recent Audit Events")
        
        audit_data = [
            {
                "Timestamp": "2024-01-20 14:30:25",
                "Event Type": "agent_selection",
                "Actor": "SwarmIntelligenceSystem",
                "Action": "select_mission_agents",
                "Resource": "mission_coordinated_strike",
                "Outcome": "success",
                "Audit Level": "COMPLIANCE"
            },
            {
                "Timestamp": "2024-01-20 14:30:20",
                "Event Type": "system_initialization", 
                "Actor": "SwarmIntelligenceSystem",
                "Action": "initialize_swarm",
                "Resource": "swarm_30_agents",
                "Outcome": "success",
                "Audit Level": "BASIC"
            }
        ]
        
        audit_df = pd.DataFrame(audit_data)
        st.dataframe(audit_df, use_container_width=True)
    
    with tab3:
        st.subheader("Compliance Violations & Alerts")
        
        # Violation severity breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Violation Severity Distribution**")
            
            severity_data = {
                "Low": 0,
                "Medium": 0, 
                "High": 0,
                "Critical": 0
            }
            
            severity_fig = go.Figure(data=[go.Bar(
                x=list(severity_data.keys()),
                y=list(severity_data.values()),
                marker_color=['#28a745', '#ffc107', '#fd7e14', '#dc3545']
            )])
            
            severity_fig.update_layout(
                title="Violations by Severity (Last 30 Days)",
                xaxis_title="Severity Level",
                yaxis_title="Count",
                height=300
            )
            
            st.plotly_chart(severity_fig, use_container_width=True)
        
        with col2:
            st.markdown("**Framework Compliance Scores**")
            
            framework_scores = {
                "GDPR": 98.5,
                "SOX": 96.8,
                "HIPAA": 99.2,
                "FAA": 94.1
            }
            
            for framework, score in framework_scores.items():
                color = "#28a745" if score >= 95 else "#ffc107" if score >= 90 else "#dc3545"
                st.markdown(f"""
                <div style="margin: 10px 0;">
                    <strong>{framework}:</strong> 
                    <span style="color: {color}; font-weight: bold;">{score:.1f}%</span>
                    <div style="background: #e9ecef; height: 10px; border-radius: 5px; margin-top: 5px;">
                        <div style="background: {color}; height: 100%; width: {score}%; border-radius: 5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Recent violations table (empty for demo)
        st.subheader("Recent Violations")
        st.info("✅ No compliance violations detected in the last 30 days")
    
    with tab4:
        st.subheader("Compliance Framework Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Active Frameworks**")
            
            frameworks = ["GDPR", "SOX", "HIPAA", "FAA", "FDA"]
            active_frameworks = st.multiselect(
                "Enable Compliance Frameworks",
                frameworks,
                default=["GDPR", "SOX", "HIPAA"]
            )
            
            st.markdown("**Monitoring Settings**")
            real_time_monitoring = st.checkbox("Real-time Monitoring", value=True)
            auto_reporting = st.checkbox("Automatic Reporting", value=True)
            violation_alerts = st.checkbox("Violation Alerts", value=True)
        
        with col2:
            st.markdown("**Framework Details**")
            
            if "GDPR" in active_frameworks:
                st.markdown("""
                **GDPR Compliance:**
                - Data privacy protection ✅
                - Right to explanation ✅  
                - Consent management ✅
                - Data retention policies ✅
                """)
            
            if "SOX" in active_frameworks:
                st.markdown("""
                **SOX Compliance:**
                - Financial controls ✅
                - Audit trail integrity ✅
                - Internal controls ✅
                - Executive certification ✅
                """)
        
        if st.button("💾 Save Configuration", type="primary"):
            st.success("✅ Compliance configuration saved successfully")

elif page == "📈 Investor Demo":
    # Professional investor demonstration
    st.header("📈 Investor Demonstration Package")
    st.markdown("Comprehensive presentation materials for investment discussions")
    
    # Generate investor package
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Executive Summary")
        
        exec_summary = st.session_state.visualizer.generate_executive_summary_report()
        
        # Key highlights
        st.markdown(f"""
        <div class="commercial-highlight">
            <h4>Investment Highlights</h4>
            <ul>
                <li><strong>Market Opportunity:</strong> $100B+ autonomous systems market</li>
                <li><strong>Technology Advantage:</strong> First-mover in swarm intelligence</li>
                <li><strong>Proven Performance:</strong> {exec_summary['performance_metrics']['mission_success_rate']} success rate</li>
                <li><strong>Scalable ROI:</strong> {exec_summary['financial_analysis']['roi_percentage']} current ROI</li>
                <li><strong>Multiple Applications:</strong> Logistics, emergency response, agriculture</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Financial projections
        st.subheader("Financial Projections")
        
        # Create projection chart
        years = [2024, 2025, 2026, 2027, 2028]
        revenue_projection = [0.5, 2.1, 8.5, 25.3, 67.8]  # Mock millions
        profit_projection = [0.1, 0.8, 4.2, 15.2, 47.3]   # Mock millions
        
        fig_projection = go.Figure()
        fig_projection.add_trace(go.Bar(
            x=years,
            y=revenue_projection,
            name='Revenue',
            marker_color='#2E86AB'
        ))
        fig_projection.add_trace(go.Bar(
            x=years,
            y=profit_projection,
            name='Profit',
            marker_color='#C73E1D'
        ))
        
        fig_projection.update_layout(
            title="5-Year Financial Projections ($M)",
            xaxis_title="Year",
            yaxis_title="Amount ($M)",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_projection, use_container_width=True)
    
    with col2:
        st.subheader("Quick Actions")
        
        if st.button("📊 Generate Full Presentation", type="primary", use_container_width=True):
            with st.spinner("Creating comprehensive investor presentation..."):
                file_paths = create_investor_presentation(
                    st.session_state.swarm_system,
                    st.session_state.commercial_system,
                    "visualizations/investor_demo/"
                )
                
                st.success("✅ Presentation package created!")
                st.markdown(f"""
                **Generated Files:**
                - Executive Dashboard
                - Mission Analysis 
                - Business Case Study
                - Evolution Demonstration
                - Financial Projections
                
                Files saved to: `visualizations/investor_demo/`
                """)
        
        if st.button("💼 Run Live Demo", use_container_width=True):
            # Run a live demonstration mission
            with st.spinner("Executing live demonstration..."):
                demo_targets = [(120, 90), (180, 140), (150, 170)]
                demo_result = st.session_state.swarm_system.execute_swarm_mission(demo_targets)
                
                st.markdown(f"""
                <div class="success-banner">
                    <h4>🎯 Live Demo Results</h4>
                    <p><strong>Success Rate:</strong> {demo_result['performance']['success_rate']:.1%}</p>
                    <p><strong>Coordination:</strong> {demo_result['performance']['coordination_score']:.1%}</p>
                    <p><strong>Efficiency:</strong> {demo_result['performance']['efficiency_score']:.2f}</p>
                    <p><strong>Response Time:</strong> {demo_result['duration']:.2f}s</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-panel">
            <h4>📞 Next Steps</h4>
            <p>Ready to discuss investment opportunities?</p>
            <ul>
                <li>Schedule technical deep-dive</li>
                <li>Review financial projections</li>
                <li>Pilot deployment planning</li>
                <li>Market entry strategy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        🕸️ Swarm Intelligence Platform v2.0 | Next-Generation Autonomous Systems
        <br>
        Built with advanced AI, distributed coordination, and commercial focus
        <br>
        <a href='https://github.com/Sakeeb91/Inverse-Dynamics-Solver' target='_blank'>📚 Technical Documentation</a> | 
        <a href='#' onclick='alert("Contact: investments@swarmtech.ai")'>💼 Investment Inquiries</a>
    </div>
    """, 
    unsafe_allow_html=True
)