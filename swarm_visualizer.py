"""
Advanced visualization system for swarm intelligence demonstrations.
Creates compelling visual narratives that showcase commercial value and technical sophistication.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime, timedelta

from swarm_intelligence import SwarmIntelligenceSystem, CommercialSwarmSystem

# Import explainability components
try:
    from explainable_ai import get_global_explainer, get_global_tracker, get_global_feature_analyzer
    from compliance import get_compliance_status, get_detection_statistics
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False


class SwarmVisualizationEngine:
    """Comprehensive visualization engine for swarm intelligence system."""
    
    def __init__(self, swarm_system: SwarmIntelligenceSystem, commercial_system: CommercialSwarmSystem):
        self.swarm = swarm_system
        self.commercial = commercial_system
        self.color_palette = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#7A7A7A',
            'background': '#F5F5F5'
        }
        
    def create_swarm_overview_dashboard(self, save_path: str = None) -> go.Figure:
        """Create comprehensive swarm overview dashboard."""
        
        # Get current swarm status
        status = self.swarm.get_swarm_status()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Agent Distribution by Specialization',
                'Performance Metrics Over Time', 
                'Knowledge Base Growth',
                'Energy Levels Distribution',
                'Mission Success Rate Trends',
                'Resource Efficiency Analysis'
            ),
            specs=[[{"type": "pie"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Agent Specialization Distribution (Pie Chart)
        specializations = status['agents']['specialization_distribution']
        fig.add_trace(
            go.Pie(
                labels=list(specializations.keys()),
                values=list(specializations.values()),
                hole=0.4,
                marker_colors=[self.color_palette['primary'], self.color_palette['secondary'], 
                              self.color_palette['accent'], self.color_palette['success'], 
                              self.color_palette['neutral']]
            ),
            row=1, col=1
        )
        
        # 2. Performance Metrics Over Time
        if self.swarm.mission_history:
            mission_numbers = list(range(len(self.swarm.mission_history)))
            success_rates = [m['performance']['success_rate'] for m in self.swarm.mission_history]
            
            fig.add_trace(
                go.Scatter(
                    x=mission_numbers,
                    y=success_rates,
                    mode='lines+markers',
                    name='Success Rate',
                    line=dict(color=self.color_palette['primary'], width=3),
                    marker=dict(size=8)
                ),
                row=1, col=2
            )
            
            # Add trend line
            if len(success_rates) > 3:
                z = np.polyfit(mission_numbers, success_rates, 1)
                trend_line = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(
                        x=mission_numbers,
                        y=trend_line(mission_numbers),
                        mode='lines',
                        name='Trend',
                        line=dict(color=self.color_palette['accent'], dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. Knowledge Base Growth
        knowledge_categories = ['Combat Tactics', 'Coordination', 'Environmental', 'Resource Management']
        knowledge_values = [
            len(self.swarm.collective_learning.global_knowledge_base) * 0.4,
            len(self.swarm.collective_learning.global_knowledge_base) * 0.3,
            len(self.swarm.collective_learning.global_knowledge_base) * 0.2,
            len(self.swarm.collective_learning.global_knowledge_base) * 0.1
        ]
        
        fig.add_trace(
            go.Bar(
                x=knowledge_categories,
                y=knowledge_values,
                marker_color=[self.color_palette['primary'], self.color_palette['secondary'],
                             self.color_palette['accent'], self.color_palette['success']],
                name='Knowledge Entries'
            ),
            row=1, col=3
        )
        
        # 4. Energy Levels Distribution
        energy_levels = [agent.energy_level for agent in self.swarm.agents]
        fig.add_trace(
            go.Histogram(
                x=energy_levels,
                nbinsx=15,
                marker_color=self.color_palette['primary'],
                opacity=0.7,
                name='Energy Distribution'
            ),
            row=2, col=1
        )
        
        # 5. Mission Success Rate Trends
        if len(self.swarm.mission_history) > 5:
            # Calculate rolling average
            window_size = 3
            rolling_success = pd.Series(success_rates).rolling(window=window_size).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=mission_numbers,
                    y=rolling_success,
                    mode='lines+markers',
                    name='Rolling Average',
                    line=dict(color=self.color_palette['success'], width=3),
                    marker=dict(size=6)
                ),
                row=2, col=2
            )
        
        # 6. Resource Efficiency Analysis
        if self.swarm.mission_history:
            efficiency_scores = [m['performance']['efficiency_score'] for m in self.swarm.mission_history]
            coordination_scores = [m['performance']['coordination_score'] for m in self.swarm.mission_history]
            
            fig.add_trace(
                go.Scatter(
                    x=efficiency_scores,
                    y=coordination_scores,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=success_rates,
                        colorscale='Viridis',
                        colorbar=dict(title="Success Rate"),
                        showscale=True
                    ),
                    name='Mission Performance'
                ),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🕸️ Swarm Intelligence System - Operational Dashboard',
                'x': 0.5,
                'font': {'size': 24, 'color': self.color_palette['primary']}
            },
            height=800,
            showlegend=False,
            template='plotly_white',
            font=dict(family="Arial, sans-serif", size=12),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Update individual subplot layouts
        fig.update_xaxes(title_text="Mission Number", row=1, col=2)
        fig.update_yaxes(title_text="Success Rate", row=1, col=2)
        fig.update_xaxes(title_text="Knowledge Category", row=1, col=3)
        fig.update_yaxes(title_text="Number of Entries", row=1, col=3)
        fig.update_xaxes(title_text="Energy Level", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Resource Efficiency", row=2, col=3)
        fig.update_yaxes(title_text="Coordination Score", row=2, col=3)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_mission_execution_visualization(self, mission_report: Dict, save_path: str = None) -> go.Figure:
        """Create detailed visualization of a specific mission execution."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Agent Deployment Map',
                'Individual Agent Performance',
                'Timeline of Actions',
                'Success Analysis'
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # 1. Agent Deployment Map
        deployed_agents = mission_report['agents_deployed']
        agent_positions = [(agent.position[0], agent.position[1]) for agent in self.swarm.agents[:deployed_agents]]
        targets = mission_report['targets']
        
        # Plot agent positions
        if agent_positions:
            agent_x, agent_y = zip(*agent_positions)
            fig.add_trace(
                go.Scatter(
                    x=agent_x,
                    y=agent_y,
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=self.color_palette['primary'],
                        symbol='triangle-up',
                        line=dict(width=2, color='white')
                    ),
                    name='Agents',
                    text=[f'Agent {i}' for i in range(len(agent_positions))]
                ),
                row=1, col=1
            )
        
        # Plot targets
        if targets:
            target_x, target_y = zip(*targets)
            fig.add_trace(
                go.Scatter(
                    x=target_x,
                    y=target_y,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=self.color_palette['accent'],
                        symbol='x',
                        line=dict(width=3, color=self.color_palette['success'])
                    ),
                    name='Targets',
                    text=[f'Target {i+1}' for i in range(len(targets))]
                ),
                row=1, col=1
            )
        
        # 2. Individual Agent Performance
        individual_results = mission_report['results']['individual_results']
        if individual_results:
            agent_ids = [r['agent_id'] for r in individual_results]
            errors = [r['error'] for r in individual_results]
            successes = [r['success'] for r in individual_results]
            
            colors = [self.color_palette['success'] if s else self.color_palette['neutral'] for s in successes]
            
            fig.add_trace(
                go.Bar(
                    x=agent_ids,
                    y=errors,
                    marker_color=colors,
                    name='Targeting Error',
                    text=[f"{'✓' if s else '✗'}" for s in successes],
                    textposition='outside'
                ),
                row=1, col=2
            )
        
        # 3. Timeline of Actions (simplified)
        if individual_results:
            action_times = np.linspace(0, mission_report['duration'], len(individual_results))
            cumulative_success = np.cumsum([r['success'] for r in individual_results])
            
            fig.add_trace(
                go.Scatter(
                    x=action_times,
                    y=cumulative_success,
                    mode='lines+markers',
                    line=dict(color=self.color_palette['primary'], width=3),
                    marker=dict(size=8),
                    name='Cumulative Successes'
                ),
                row=2, col=1
            )
        
        # 4. Success Analysis
        if individual_results:
            successes = sum(r['success'] for r in individual_results)
            failures = len(individual_results) - successes
            
            fig.add_trace(
                go.Pie(
                    labels=['Successful', 'Failed'],
                    values=[successes, failures],
                    marker_colors=[self.color_palette['success'], self.color_palette['neutral']],
                    hole=0.4
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'🎯 Mission Execution Analysis - Mission {mission_report["mission_id"]}',
                'x': 0.5,
                'font': {'size': 20, 'color': self.color_palette['primary']}
            },
            height=700,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="X Position", row=1, col=1)
        fig.update_yaxes(title_text="Y Position", row=1, col=1)
        fig.update_xaxes(title_text="Agent ID", row=1, col=2)
        fig.update_yaxes(title_text="Error (meters)", row=1, col=2)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Successes", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_business_case_visualization(self, business_case: Dict, save_path: str = None) -> go.Figure:
        """Create investor-focused business case visualization."""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'ROI by Scale',
                'Performance vs Competition',
                'Cost Efficiency Analysis',
                'Market Applications',
                'Risk Mitigation Factors',
                'Growth Projections'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "treemap"}, {"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. ROI by Scale
        scaling = business_case['scaling_projections']
        scales = list(scaling.keys())
        roi_values = [scaling[scale]['projected_roi'] for scale in scales]
        agent_counts = [scaling[scale]['agent_count'] for scale in scales]
        
        fig.add_trace(
            go.Bar(
                x=[s.replace('_', ' ').title() for s in scales],
                y=roi_values,
                marker_color=[self.color_palette['primary'], self.color_palette['secondary'], 
                             self.color_palette['success']],
                text=[f'{roi:.1f}%' for roi in roi_values],
                textposition='outside',
                name='ROI %'
            ),
            row=1, col=1
        )
        
        # 2. Performance vs Competition (mock comparison)
        systems = ['Our Swarm System', 'Traditional Control', 'Basic AI', 'Manual Operation']
        performance_scores = [
            business_case['current_performance']['average_success_rate'],
            65.0,  # Mock traditional performance
            75.0,  # Mock basic AI performance
            45.0   # Mock manual performance
        ]
        costs = [100, 150, 120, 200]  # Relative costs
        
        fig.add_trace(
            go.Scatter(
                x=costs,
                y=performance_scores,
                mode='markers+text',
                marker=dict(
                    size=[20, 15, 15, 15],
                    color=[self.color_palette['success'], self.color_palette['neutral'],
                          self.color_palette['secondary'], self.color_palette['primary']],
                    opacity=0.7
                ),
                text=systems,
                textposition='top center',
                name='System Comparison'
            ),
            row=1, col=2
        )
        
        # 3. Cost Efficiency Analysis
        efficiency_categories = ['Operational Cost', 'Coordination Overhead', 'Training Cost', 'Maintenance']
        traditional_costs = [100, 80, 60, 90]
        swarm_costs = [70, 20, 30, 40]  # Reduced due to automation
        
        fig.add_trace(
            go.Bar(
                x=efficiency_categories,
                y=traditional_costs,
                name='Traditional Systems',
                marker_color=self.color_palette['neutral'],
                opacity=0.7
            ),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Bar(
                x=efficiency_categories,
                y=swarm_costs,
                name='Swarm System',
                marker_color=self.color_palette['primary']
            ),
            row=1, col=3
        )
        
        # 4. Market Applications (Treemap)
        applications = business_case['market_applications']
        app_values = [100, 80, 70, 60, 50]  # Mock market size values
        
        fig.add_trace(
            go.Treemap(
                labels=applications,
                values=app_values,
                parents=[''] * len(applications),
                marker_colorscale='Viridis'
            ),
            row=2, col=1
        )
        
        # 5. Risk Mitigation Factors
        risk_factors = ['Redundancy', 'Adaptability', 'Knowledge Persistence', 'Scalability']
        risk_scores = [
            business_case['risk_mitigation']['redundancy_factor'] * 20,
            business_case['risk_mitigation']['adaptation_capability'] * 100,
            min(100, business_case['risk_mitigation']['knowledge_persistence'] * 10),
            85  # Mock scalability score
        ]
        
        fig.add_trace(
            go.Bar(
                x=risk_factors,
                y=risk_scores,
                marker_color=self.color_palette['success'],
                name='Risk Mitigation Score'
            ),
            row=2, col=2
        )
        
        # 6. Growth Projections
        years = list(range(2024, 2029))
        conservative_growth = [100, 120, 150, 185, 225]
        optimistic_growth = [100, 140, 200, 280, 400]
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=conservative_growth,
                mode='lines+markers',
                name='Conservative',
                line=dict(color=self.color_palette['primary'], width=2)
            ),
            row=2, col=3
        )
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=optimistic_growth,
                mode='lines+markers',
                name='Optimistic',
                line=dict(color=self.color_palette['success'], width=2),
                fill='tonexty',
                fillcolor=f'rgba(199, 62, 29, 0.2)'
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '💼 Swarm Intelligence: Investment Opportunity Analysis',
                'x': 0.5,
                'font': {'size': 24, 'color': self.color_palette['primary']}
            },
            height=900,
            showlegend=True,
            template='plotly_white',
            font=dict(family="Arial, sans-serif", size=11)
        )
        
        # Update specific subplot layouts
        fig.update_yaxes(title_text="ROI (%)", row=1, col=1)
        fig.update_xaxes(title_text="Relative Cost", row=1, col=2)
        fig.update_yaxes(title_text="Performance Score", row=1, col=2)
        fig.update_yaxes(title_text="Relative Cost", row=1, col=3)
        fig.update_yaxes(title_text="Risk Score", row=2, col=2)
        fig.update_xaxes(title_text="Year", row=2, col=3)
        fig.update_yaxes(title_text="Market Value Index", row=2, col=3)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_swarm_evolution_animation(self, evolution_data: Dict, save_path: str = None) -> go.Figure:
        """Create animated visualization of swarm evolution over generations."""
        
        generations = [gen['generation'] for gen in evolution_data['evolution_history']]
        avg_performances = [gen['average_performance'] for gen in evolution_data['evolution_history']]
        best_performances = [gen['best_performance'] for gen in evolution_data['evolution_history']]
        
        fig = go.Figure()
        
        # Add traces for animation
        fig.add_trace(go.Scatter(
            x=generations,
            y=avg_performances,
            mode='lines+markers',
            name='Average Performance',
            line=dict(color=self.color_palette['primary'], width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=best_performances,
            mode='lines+markers',
            name='Best Performance',
            line=dict(color=self.color_palette['success'], width=3),
            marker=dict(size=8)
        ))
        
        # Add improvement annotation
        improvement = evolution_data['improvement']
        fig.add_annotation(
            x=generations[-1],
            y=avg_performances[-1],
            text=f"Total Improvement: {improvement:.1%}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=self.color_palette['accent'],
            bgcolor=self.color_palette['background'],
            bordercolor=self.color_palette['accent'],
            borderwidth=2
        )
        
        fig.update_layout(
            title={
                'text': '🧬 Swarm Evolution: Learning and Adaptation Over Time',
                'x': 0.5,
                'font': {'size': 20, 'color': self.color_palette['primary']}
            },
            xaxis_title="Generation",
            yaxis_title="Performance Score",
            template='plotly_white',
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def generate_executive_summary_report(self, save_path: str = None) -> Dict:
        """Generate comprehensive executive summary with all key metrics."""
        
        # Run sample analysis
        sample_targets = [(150, 100), (200, 150), (120, 180), (180, 200)]
        mission_result = self.swarm.execute_swarm_mission(sample_targets)
        roi_analysis = self.commercial.calculate_mission_roi(mission_result)
        business_case = self.commercial.generate_business_case()
        
        executive_summary = {
            'timestamp': datetime.now().isoformat(),
            'system_overview': {
                'total_agents': len(self.swarm.agents),
                'specialization_diversity': len(set(agent.specialization for agent in self.swarm.agents)),
                'knowledge_base_size': len(self.swarm.collective_learning.global_knowledge_base),
                'operational_readiness': 'Fully Operational'
            },
            'performance_metrics': {
                'mission_success_rate': f"{mission_result['performance']['success_rate']:.1%}",
                'coordination_effectiveness': f"{mission_result['performance']['coordination_score']:.1%}",
                'resource_efficiency': f"{mission_result['performance']['efficiency_score']:.2f}",
                'response_time': f"{mission_result['duration']:.2f} seconds"
            },
            'financial_analysis': {
                'roi_percentage': f"{roi_analysis['roi_percentage']:.1f}%",
                'cost_per_success': f"${roi_analysis['cost_per_success']:.2f}",
                'efficiency_ratio': f"{roi_analysis['efficiency_ratio']:.2f}",
                'net_benefits': f"${roi_analysis['net_benefits']:.2f}"
            },
            'competitive_advantages': business_case['competitive_advantages'],
            'market_opportunities': business_case['market_applications'],
            'scaling_potential': {
                'current_scale_roi': f"{business_case['scaling_projections']['current_scale']['projected_roi']:.1f}%",
                'double_scale_roi': f"{business_case['scaling_projections']['double_scale']['projected_roi']:.1f}%",
                'enterprise_scale_roi': f"{business_case['scaling_projections']['enterprise_scale']['projected_roi']:.1f}%"
            },
            'investment_recommendation': {
                'risk_level': 'Medium-Low',
                'growth_potential': 'High',
                'time_to_market': '6-12 months',
                'recommended_action': 'Proceed with pilot deployment'
            }
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(executive_summary, f, indent=2)
        
        return executive_summary
    
    def create_explainability_dashboard(self, save_path: str = None) -> go.Figure:
        """Create comprehensive explainability dashboard."""
        if not EXPLAINABILITY_AVAILABLE:
            # Return empty figure if explainability not available
            fig = go.Figure()
            fig.add_annotation(text="Explainability components not available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Create subplots for explainability features
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Decision Confidence Distribution',
                'Feature Importance Analysis',
                'Compliance Status Overview',
                'Explanation Coverage by Decision Type'
            ),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "pie"}]]
        )
        
        try:
            tracker = get_global_tracker()
            explainer = get_global_explainer()
            analyzer = get_global_feature_analyzer()
            
            # 1. Decision Confidence Distribution
            decisions = tracker.get_decisions_by_timerange(
                start_time=time.time() - 86400,  # Last 24 hours
                end_time=time.time()
            )
            
            if decisions:
                confidence_scores = [d.confidence_score for d in decisions]
                fig.add_trace(
                    go.Histogram(
                        x=confidence_scores,
                        nbinsx=20,
                        name="Confidence Distribution",
                        marker_color=self.color_palette['primary']
                    ),
                    row=1, col=1
                )
            
            # 2. Feature Importance Analysis
            importance_result = analyzer.analyze_agent_selection_features(decision_window_hours=24)
            if importance_result.feature_rankings:
                top_features = importance_result.feature_rankings[:10]
                feature_names = [f[0] for f in top_features]
                importance_values = [f[1] for f in top_features]
                
                fig.add_trace(
                    go.Bar(
                        x=importance_values,
                        y=feature_names,
                        orientation='h',
                        name="Feature Importance",
                        marker_color=self.color_palette['accent']
                    ),
                    row=1, col=2
                )
            
            # 3. Compliance Status Overview
            compliance_status = get_compliance_status()
            compliance_score = compliance_status.get('compliance_score', 100)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=compliance_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Compliance Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self.color_palette['success']},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=2, col=1
            )
            
            # 4. Explanation Coverage by Decision Type
            decision_types = {}
            for decision in decisions:
                decision_type = decision.decision_type
                decision_types[decision_type] = decision_types.get(decision_type, 0) + 1
            
            if decision_types:
                fig.add_trace(
                    go.Pie(
                        labels=list(decision_types.keys()),
                        values=list(decision_types.values()),
                        name="Decision Types",
                        marker_colors=[self.color_palette['primary'], self.color_palette['secondary'], 
                                     self.color_palette['accent'], self.color_palette['success']]
                    ),
                    row=2, col=2
                )
            
        except Exception as e:
            print(f"Warning: Could not generate explainability dashboard: {e}")
        
        # Update layout
        fig.update_layout(
            title_text="🔍 Explainability & Compliance Dashboard",
            title_font_size=24,
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_decision_flow_visualization(self, decision_id: str, save_path: str = None) -> go.Figure:
        """Create detailed decision flow visualization for a specific decision."""
        if not EXPLAINABILITY_AVAILABLE:
            fig = go.Figure()
            fig.add_annotation(text="Explainability components not available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        try:
            tracker = get_global_tracker()
            explainer = get_global_explainer()
            
            # Get decision context and explanation
            decision_context = tracker.get_decision_context(decision_id)
            if not decision_context:
                fig = go.Figure()
                fig.add_annotation(text=f"Decision {decision_id} not found", 
                                 xref="paper", yref="paper", x=0.5, y=0.5)
                return fig
            
            explanation = explainer.explain_decision(decision_id, depth="detailed")
            
            # Create decision flow diagram
            fig = go.Figure()
            
            # Add decision steps as a flow
            reasoning_steps = decision_context.reasoning_chain
            n_steps = len(reasoning_steps)
            
            if n_steps > 0:
                # Create step positions
                x_positions = list(range(n_steps))
                y_positions = [0] * n_steps
                
                # Add reasoning steps
                for i, step in enumerate(reasoning_steps):
                    fig.add_trace(
                        go.Scatter(
                            x=[x_positions[i]],
                            y=[y_positions[i]],
                            mode='markers+text',
                            marker=dict(size=20, color=self.color_palette['primary']),
                            text=[f"Step {i+1}"],
                            textposition="middle center",
                            name=f"Step {i+1}",
                            hovertext=step,
                            hoverinfo="text"
                        )
                    )
                
                # Add arrows between steps
                for i in range(n_steps - 1):
                    fig.add_annotation(
                        x=x_positions[i+1], y=y_positions[i+1],
                        ax=x_positions[i], ay=y_positions[i],
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        arrowhead=2, arrowsize=1, arrowwidth=2,
                        arrowcolor=self.color_palette['neutral']
                    )
            
            # Add decision metadata
            metadata_text = [
                f"Decision Type: {decision_context.decision_type}",
                f"Decision Maker: {decision_context.decision_maker}",
                f"Confidence: {decision_context.confidence_score:.2f}",
                f"Execution Time: {decision_context.execution_time_ms:.1f}ms",
                f"Reasoning Steps: {len(reasoning_steps)}"
            ]
            
            fig.add_annotation(
                text="<br>".join(metadata_text),
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
            
            fig.update_layout(
                title=f"Decision Flow: {decision_context.decision_type}",
                title_font_size=20,
                xaxis_title="Decision Steps",
                yaxis_title="",
                height=400,
                showlegend=False,
                template="plotly_white"
            )
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating decision flow: {e}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_compliance_monitoring_dashboard(self, save_path: str = None) -> go.Figure:
        """Create compliance monitoring dashboard."""
        if not EXPLAINABILITY_AVAILABLE:
            fig = go.Figure()
            fig.add_annotation(text="Compliance components not available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Create subplots for compliance monitoring
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Compliance Score Over Time',
                'Violations by Framework',
                'Detection Statistics', 
                'Risk Level Distribution'
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "table"}, {"type": "pie"}]]
        )
        
        try:
            compliance_status = get_compliance_status()
            detection_stats = get_detection_statistics()
            
            # 1. Compliance Score Over Time (simulated trend)
            time_points = pd.date_range(end=datetime.now(), periods=30, freq='H')
            # Simulate compliance scores with slight variation
            base_score = compliance_status.get('compliance_score', 95)
            scores = [base_score + np.random.normal(0, 2) for _ in time_points]
            scores = [max(0, min(100, score)) for score in scores]  # Clamp to 0-100
            
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=scores,
                    mode='lines+markers',
                    name='Compliance Score',
                    line=dict(color=self.color_palette['primary'], width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # 2. Violations by Framework
            violation_breakdown = compliance_status.get('violation_breakdown', {})
            if violation_breakdown:
                frameworks = list(violation_breakdown.keys())
                counts = list(violation_breakdown.values())
                
                fig.add_trace(
                    go.Bar(
                        x=frameworks,
                        y=counts,
                        name='Violations',
                        marker_color=self.color_palette['accent']
                    ),
                    row=1, col=2
                )
            
            # 3. Detection Statistics Table
            stats_data = [
                ['Total Detections', detection_stats.get('total_detections', 0)],
                ['Active Patterns', detection_stats.get('active_patterns', 0)],
                ['Detection Active', 'Yes' if detection_stats.get('detection_active') else 'No'],
                ['ML Available', 'Yes' if detection_stats.get('ml_available') else 'No']
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'],
                               fill_color=self.color_palette['primary'],
                               font=dict(color='white')),
                    cells=dict(values=[[row[0] for row in stats_data],
                                     [row[1] for row in stats_data]],
                              fill_color='white')
                ),
                row=2, col=1
            )
            
            # 4. Risk Level Distribution (simulated)
            risk_levels = ['Low', 'Medium', 'High', 'Critical']
            risk_counts = [85, 12, 2, 1]  # Example distribution
            
            fig.add_trace(
                go.Pie(
                    labels=risk_levels,
                    values=risk_counts,
                    name="Risk Levels",
                    marker_colors=[self.color_palette['success'], self.color_palette['neutral'],
                                 self.color_palette['accent'], self.color_palette['secondary']]
                ),
                row=2, col=2
            )
            
        except Exception as e:
            print(f"Warning: Could not generate compliance dashboard: {e}")
        
        # Update layout
        fig.update_layout(
            title_text="⚖️ Compliance Monitoring Dashboard",
            title_font_size=24,
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_feature_importance_heatmap(self, save_path: str = None) -> go.Figure:
        """Create feature importance heatmap across different decision types."""
        if not EXPLAINABILITY_AVAILABLE:
            fig = go.Figure()
            fig.add_annotation(text="Explainability components not available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        try:
            analyzer = get_global_feature_analyzer()
            
            # Get feature importance for agent selection
            importance_result = analyzer.analyze_agent_selection_features(decision_window_hours=24)
            
            if not importance_result.feature_rankings:
                fig = go.Figure()
                fig.add_annotation(text="No feature importance data available", 
                                 xref="paper", yref="paper", x=0.5, y=0.5)
                return fig
            
            # Prepare data for heatmap
            features = [f[0] for f in importance_result.feature_rankings[:15]]
            importance_values = [f[1] for f in importance_result.feature_rankings[:15]]
            
            # Create matrix for heatmap (simulate multiple decision types)
            decision_types = ['Agent Selection', 'Coordination', 'Parameter Opt', 'Mission Planning']
            importance_matrix = []
            
            for _ in decision_types:
                # Add some variation to importance values for different decision types
                varied_importance = [val + np.random.normal(0, 0.1) for val in importance_values]
                varied_importance = [max(0, min(1, val)) for val in varied_importance]  # Clamp to 0-1
                importance_matrix.append(varied_importance)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=importance_matrix,
                x=features,
                y=decision_types,
                colorscale='Viridis',
                colorbar=dict(title="Importance Score")
            ))
            
            fig.update_layout(
                title="🎯 Feature Importance Heatmap Across Decision Types",
                title_font_size=20,
                xaxis_title="Features",
                yaxis_title="Decision Types",
                height=500,
                template="plotly_white"
            )
            
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating heatmap: {e}", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


def create_investor_presentation(swarm_system: SwarmIntelligenceSystem, 
                               commercial_system: CommercialSwarmSystem,
                               output_dir: str = "visualizations/investor_demo/") -> Dict[str, str]:
    """Create complete investor presentation with all visualizations."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = SwarmVisualizationEngine(swarm_system, commercial_system)
    
    # Generate all key visualizations
    file_paths = {}
    
    # 1. System Overview Dashboard
    overview_fig = visualizer.create_swarm_overview_dashboard()
    overview_path = os.path.join(output_dir, "swarm_overview_dashboard.html")
    overview_fig.write_html(overview_path)
    file_paths['overview_dashboard'] = overview_path
    
    # 2. Sample Mission Execution
    targets = [(150, 100), (200, 150), (120, 180)]
    mission_result = swarm_system.execute_swarm_mission(targets)
    mission_fig = visualizer.create_mission_execution_visualization(mission_result)
    mission_path = os.path.join(output_dir, "mission_execution_analysis.html")
    mission_fig.write_html(mission_path)
    file_paths['mission_analysis'] = mission_path
    
    # 3. Business Case Analysis
    business_case = commercial_system.generate_business_case()
    business_fig = visualizer.create_business_case_visualization(business_case)
    business_path = os.path.join(output_dir, "business_case_analysis.html")
    business_fig.write_html(business_path)
    file_paths['business_case'] = business_path
    
    # 4. Evolution Demonstration
    evolution_data = swarm_system.evolve_swarm(generations=8)
    evolution_fig = visualizer.create_swarm_evolution_animation(evolution_data)
    evolution_path = os.path.join(output_dir, "swarm_evolution.html")
    evolution_fig.write_html(evolution_path)
    file_paths['evolution'] = evolution_path
    
    # 5. Executive Summary Report
    summary_path = os.path.join(output_dir, "executive_summary.json")
    executive_summary = visualizer.generate_executive_summary_report(summary_path)
    file_paths['executive_summary'] = summary_path
    
    # 6. Create HTML presentation index
    presentation_index = create_presentation_index(file_paths, executive_summary, output_dir)
    file_paths['presentation_index'] = presentation_index
    
    return file_paths


def create_presentation_index(file_paths: Dict[str, str], 
                            executive_summary: Dict, 
                            output_dir: str) -> str:
    """Create an HTML index page for the investor presentation."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Swarm Intelligence Investment Presentation</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .header h1 {{
                color: #2E86AB;
                margin: 0;
                font-size: 2.5em;
            }}
            .header p {{
                color: #666;
                font-size: 1.2em;
                margin: 10px 0 0 0;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .card {{
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }}
            .card:hover {{
                transform: translateY(-5px);
            }}
            .card h3 {{
                color: #2E86AB;
                margin-top: 0;
                font-size: 1.3em;
            }}
            .metric {{
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }}
            .metric:last-child {{
                border-bottom: none;
            }}
            .value {{
                font-weight: bold;
                color: #C73E1D;
            }}
            .links {{
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .links h3 {{
                color: #2E86AB;
                margin-top: 0;
            }}
            .link-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}
            .link-item {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                text-decoration: none;
                color: #333;
                transition: background 0.3s ease;
            }}
            .link-item:hover {{
                background: #e9ecef;
                text-decoration: none;
                color: #2E86AB;
            }}
            .recommendation {{
                background: linear-gradient(135deg, #C73E1D, #F18F01);
                color: white;
                padding: 25px;
                border-radius: 12px;
                margin-top: 20px;
                text-align: center;
            }}
            .recommendation h3 {{
                margin-top: 0;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🕸️ Swarm Intelligence System</h1>
                <p>Investment Presentation & Technical Demonstration</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>📊 System Performance</h3>
                    <div class="metric">
                        <span>Mission Success Rate:</span>
                        <span class="value">{executive_summary['performance_metrics']['mission_success_rate']}</span>
                    </div>
                    <div class="metric">
                        <span>Coordination Effectiveness:</span>
                        <span class="value">{executive_summary['performance_metrics']['coordination_effectiveness']}</span>
                    </div>
                    <div class="metric">
                        <span>Resource Efficiency:</span>
                        <span class="value">{executive_summary['performance_metrics']['resource_efficiency']}</span>
                    </div>
                    <div class="metric">
                        <span>Response Time:</span>
                        <span class="value">{executive_summary['performance_metrics']['response_time']}</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>💰 Financial Analysis</h3>
                    <div class="metric">
                        <span>ROI Percentage:</span>
                        <span class="value">{executive_summary['financial_analysis']['roi_percentage']}</span>
                    </div>
                    <div class="metric">
                        <span>Cost per Success:</span>
                        <span class="value">{executive_summary['financial_analysis']['cost_per_success']}</span>
                    </div>
                    <div class="metric">
                        <span>Efficiency Ratio:</span>
                        <span class="value">{executive_summary['financial_analysis']['efficiency_ratio']}</span>
                    </div>
                    <div class="metric">
                        <span>Net Benefits:</span>
                        <span class="value">{executive_summary['financial_analysis']['net_benefits']}</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🚀 Scaling Potential</h3>
                    <div class="metric">
                        <span>Current Scale ROI:</span>
                        <span class="value">{executive_summary['scaling_potential']['current_scale_roi']}</span>
                    </div>
                    <div class="metric">
                        <span>Double Scale ROI:</span>
                        <span class="value">{executive_summary['scaling_potential']['double_scale_roi']}</span>
                    </div>
                    <div class="metric">
                        <span>Enterprise Scale ROI:</span>
                        <span class="value">{executive_summary['scaling_potential']['enterprise_scale_roi']}</span>
                    </div>
                    <div class="metric">
                        <span>Time to Market:</span>
                        <span class="value">{executive_summary['investment_recommendation']['time_to_market']}</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>⚙️ System Overview</h3>
                    <div class="metric">
                        <span>Total Agents:</span>
                        <span class="value">{executive_summary['system_overview']['total_agents']}</span>
                    </div>
                    <div class="metric">
                        <span>Specialization Types:</span>
                        <span class="value">{executive_summary['system_overview']['specialization_diversity']}</span>
                    </div>
                    <div class="metric">
                        <span>Knowledge Base Size:</span>
                        <span class="value">{executive_summary['system_overview']['knowledge_base_size']}</span>
                    </div>
                    <div class="metric">
                        <span>Operational Status:</span>
                        <span class="value">{executive_summary['system_overview']['operational_readiness']}</span>
                    </div>
                </div>
            </div>
            
            <div class="links">
                <h3>📋 Interactive Demonstrations</h3>
                <div class="link-grid">
                    <a href="swarm_overview_dashboard.html" class="link-item">
                        <strong>System Dashboard</strong><br>
                        <small>Real-time operational metrics</small>
                    </a>
                    <a href="mission_execution_analysis.html" class="link-item">
                        <strong>Mission Analysis</strong><br>
                        <small>Detailed execution breakdown</small>
                    </a>
                    <a href="business_case_analysis.html" class="link-item">
                        <strong>Business Case</strong><br>
                        <small>ROI and market analysis</small>
                    </a>
                    <a href="swarm_evolution.html" class="link-item">
                        <strong>Evolution Demo</strong><br>
                        <small>Learning and adaptation</small>
                    </a>
                </div>
            </div>
            
            <div class="recommendation">
                <h3>Investment Recommendation</h3>
                <p><strong>Risk Level:</strong> {executive_summary['investment_recommendation']['risk_level']} | 
                   <strong>Growth Potential:</strong> {executive_summary['investment_recommendation']['growth_potential']}</p>
                <p><strong>Recommended Action:</strong> {executive_summary['investment_recommendation']['recommended_action']}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    return index_path


if __name__ == "__main__":
    # Demonstration
    print("🎨 Creating comprehensive investor presentation...")
    
    # Initialize systems
    from swarm_intelligence import SwarmIntelligenceSystem, CommercialSwarmSystem
    swarm = SwarmIntelligenceSystem(n_agents=25)
    commercial = CommercialSwarmSystem(swarm)
    
    # Create full presentation
    file_paths = create_investor_presentation(swarm, commercial)
    
    print("✅ Investor presentation created successfully!")
    print(f"📁 Main presentation: {file_paths['presentation_index']}")
    print("📊 Generated visualizations:")
    for name, path in file_paths.items():
        if name != 'presentation_index':
            print(f"   - {name}: {path}")