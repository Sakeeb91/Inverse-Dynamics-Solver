#!/usr/bin/env python3
"""
Feature Importance Analysis for Swarm Intelligence

This module provides specialized feature importance analysis for swarm
intelligence decisions, focusing on agent capabilities, environmental
factors, and coordination parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import time

from .decision_tracker import get_global_tracker
from .audit_logger import audit_log, AuditLevel


@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis results."""
    method: str
    feature_scores: Dict[str, float]
    feature_rankings: List[Tuple[str, float]]
    baseline_score: float
    analysis_metadata: Dict[str, Any]
    timestamp: float


class SwarmFeatureAnalyzer:
    """
    Specialized feature importance analyzer for swarm intelligence decisions.
    
    This class analyzes which factors (agent capabilities, environmental conditions,
    mission parameters) have the most impact on decision outcomes and mission success.
    """
    
    def __init__(self):
        """Initialize the feature analyzer."""
        self.decision_tracker = get_global_tracker()
        self.feature_history = []
        self.cached_importance = {}
        
        # Feature categories for swarm intelligence
        self.feature_categories = {
            'agent_capabilities': [
                'max_mass', 'angle_precision', 'reload_speed',
                'communication_range', 'energy_efficiency', 'energy_level'
            ],
            'environmental_factors': [
                'field_size_x', 'field_size_y', 'wind_speed', 'wind_direction',
                'temperature', 'humidity', 'visibility'
            ],
            'mission_parameters': [
                'target_count', 'target_distance', 'mission_complexity',
                'time_constraint', 'coordination_requirement'
            ],
            'coordination_factors': [
                'agent_count', 'specialization_diversity', 'communication_density',
                'formation_quality', 'timing_synchronization'
            ]
        }
        
        # Importance calculation methods
        self.importance_methods = {
            'correlation': self._calculate_correlation_importance,
            'permutation': self._calculate_permutation_importance,
            'gradient': self._calculate_gradient_importance,
            'variance': self._calculate_variance_importance
        }
    
    def analyze_agent_selection_features(self, 
                                       decision_window_hours: int = 24) -> FeatureImportanceResult:
        """
        Analyze feature importance for agent selection decisions.
        
        Args:
            decision_window_hours: Time window for analysis
            
        Returns:
            Feature importance results for agent selection
        """
        # Get recent agent selection decisions
        cutoff_time = time.time() - (decision_window_hours * 3600)
        decisions = self.decision_tracker.get_decisions_by_type("agent_selection")
        recent_decisions = [d for d in decisions if d.timestamp >= cutoff_time]
        
        if len(recent_decisions) < 2:
            return FeatureImportanceResult(
                method="insufficient_data",
                feature_scores={},
                feature_rankings=[],
                baseline_score=0.0,
                analysis_metadata={"error": "Insufficient decision data"},
                timestamp=time.time()
            )
        
        # Extract features and outcomes
        feature_data = []
        outcomes = []
        
        for decision in recent_decisions:
            features = self._extract_decision_features(decision)
            outcome = self._get_decision_outcome_score(decision.decision_id)
            
            if features and outcome is not None:
                feature_data.append(features)
                outcomes.append(outcome)
        
        if len(feature_data) < 2:
            return FeatureImportanceResult(
                method="insufficient_features",
                feature_scores={},
                feature_rankings=[],
                baseline_score=0.0,
                analysis_metadata={"error": "Could not extract sufficient features"},
                timestamp=time.time()
            )
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(feature_data)
        feature_names = list(df.columns)
        X = df.values
        y = np.array(outcomes)
        
        # Calculate feature importance using multiple methods
        importance_results = {}
        for method_name, method_func in self.importance_methods.items():
            try:
                importance_scores = method_func(X, y, feature_names)
                importance_results[method_name] = importance_scores
            except Exception as e:
                print(f"Warning: {method_name} importance calculation failed: {e}")
        
        # Aggregate importance scores
        aggregated_scores = self._aggregate_importance_scores(importance_results, feature_names)
        
        # Rank features
        feature_rankings = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
        
        result = FeatureImportanceResult(
            method="multi_method_aggregate",
            feature_scores=aggregated_scores,
            feature_rankings=feature_rankings,
            baseline_score=np.mean(y),
            analysis_metadata={
                "decisions_analyzed": len(recent_decisions),
                "features_extracted": len(feature_names),
                "methods_used": list(importance_results.keys()),
                "time_window_hours": decision_window_hours
            },
            timestamp=time.time()
        )
        
        # Log analysis
        audit_log(
            event_type="feature_importance_analysis",
            actor="SwarmFeatureAnalyzer",
            action="analyze_agent_selection_features",
            resource="agent_selection_decisions",
            outcome="success",
            details={
                "decisions_analyzed": len(recent_decisions),
                "top_features": [f[0] for f in feature_rankings[:5]],
                "analysis_methods": list(importance_results.keys())
            },
            audit_level=AuditLevel.COMPLIANCE
        )
        
        return result
    
    def _extract_decision_features(self, decision) -> Optional[Dict[str, float]]:
        """Extract numerical features from a decision context."""
        features = {}
        
        try:
            # Agent state features
            agent_states = decision.agent_states
            if agent_states:
                # Specialization distribution
                specializations = agent_states.get('specializations', {})
                total_agents = sum(specializations.values()) if specializations else 1
                
                for spec, count in specializations.items():
                    features[f'spec_{spec}_ratio'] = count / total_agents
                
                # Average energy
                features['average_energy'] = agent_states.get('average_energy', 0.5)
            
            # Environmental features
            env_factors = decision.environmental_factors
            if env_factors:
                field_size = env_factors.get('field_size', (1000, 1000))
                if isinstance(field_size, (list, tuple)) and len(field_size) >= 2:
                    features['field_size_x'] = float(field_size[0])
                    features['field_size_y'] = float(field_size[1])
                    features['field_area'] = float(field_size[0] * field_size[1])
                
                features['communication_range'] = float(env_factors.get('communication_range', 100))
            
            # Mission parameters
            input_params = decision.input_parameters
            if input_params:
                targets = input_params.get('targets', [])
                features['target_count'] = float(len(targets))
                features['available_agents'] = float(input_params.get('available_agents', 10))
                features['agent_to_target_ratio'] = features['available_agents'] / max(1, features['target_count'])
                
                # Calculate target complexity metrics
                if targets:
                    distances = [np.sqrt(t[0]**2 + t[1]**2) for t in targets if len(t) >= 2]
                    if distances:
                        features['avg_target_distance'] = float(np.mean(distances))
                        features['max_target_distance'] = float(np.max(distances))
                        features['target_spread'] = float(np.std(distances))
            
            # Decision complexity features
            features['reasoning_steps'] = float(len(decision.reasoning_chain))
            features['confidence_score'] = float(decision.confidence_score)
            features['execution_time_ms'] = float(decision.execution_time_ms)
            
            return features
            
        except Exception as e:
            print(f"Warning: Could not extract features from decision {decision.decision_id}: {e}")
            return None
    
    def _get_decision_outcome_score(self, decision_id: str) -> Optional[float]:
        """Get a numerical outcome score for a decision."""
        outcome = self.decision_tracker.get_decision_outcome(decision_id)
        
        if not outcome:
            return None
        
        # Base score on success
        score = 1.0 if outcome.success else 0.0
        
        # Adjust based on impact metrics
        if outcome.impact_metrics:
            # Add normalized impact metrics
            for metric_name, value in outcome.impact_metrics.items():
                if isinstance(value, (int, float)):
                    # Normalize different metrics
                    if 'ratio' in metric_name or 'rate' in metric_name:
                        score += float(value) * 0.1
                    elif 'count' in metric_name:
                        score += min(float(value) / 10, 0.2)  # Cap contribution
                    elif metric_name == 'coordinator_included':
                        score += 0.1 if value else 0.0
        
        return min(score, 2.0)  # Cap maximum score
    
    def _calculate_correlation_importance(self, X: np.ndarray, y: np.ndarray, 
                                        feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance based on correlation with outcomes."""
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            try:
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                # Use absolute correlation as importance
                importance_scores[feature_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            except:
                importance_scores[feature_name] = 0.0
        
        return importance_scores
    
    def _calculate_permutation_importance(self, X: np.ndarray, y: np.ndarray,
                                        feature_names: List[str]) -> Dict[str, float]:
        """Calculate permutation-based feature importance."""
        if len(X) < 5:  # Need minimum samples for reliable permutation importance
            return {name: 0.0 for name in feature_names}
        
        try:
            # Use a simple model for permutation importance
            model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)
            model.fit(X, y)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(model, X, y, n_repeats=3, random_state=42)
            
            importance_scores = {}
            for i, feature_name in enumerate(feature_names):
                importance_scores[feature_name] = float(perm_importance.importances_mean[i])
            
            return importance_scores
            
        except Exception as e:
            print(f"Permutation importance calculation failed: {e}")
            return {name: 0.0 for name in feature_names}
    
    def _calculate_gradient_importance(self, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str]) -> Dict[str, float]:
        """Calculate gradient-based feature importance."""
        importance_scores = {}
        
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate gradients (simple linear approximation)
            for i, feature_name in enumerate(feature_names):
                feature_values = X_scaled[:, i]
                
                # Calculate gradient using finite differences
                if len(feature_values) > 1:
                    gradient = np.gradient(y, feature_values)
                    importance_scores[feature_name] = float(np.mean(np.abs(gradient)))
                else:
                    importance_scores[feature_name] = 0.0
                    
        except Exception as e:
            print(f"Gradient importance calculation failed: {e}")
            importance_scores = {name: 0.0 for name in feature_names}
        
        return importance_scores
    
    def _calculate_variance_importance(self, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str]) -> Dict[str, float]:
        """Calculate variance-based feature importance."""
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_variance = np.var(X[:, i])
            outcome_variance = np.var(y)
            
            # Importance based on feature variance relative to outcome variance
            if outcome_variance > 0:
                importance_scores[feature_name] = float(feature_variance / outcome_variance)
            else:
                importance_scores[feature_name] = 0.0
        
        return importance_scores
    
    def _aggregate_importance_scores(self, importance_results: Dict[str, Dict[str, float]],
                                   feature_names: List[str]) -> Dict[str, float]:
        """Aggregate importance scores from multiple methods."""
        aggregated = {}
        
        for feature_name in feature_names:
            scores = []
            for method_name, method_results in importance_results.items():
                if feature_name in method_results:
                    scores.append(method_results[feature_name])
            
            if scores:
                # Use median to reduce outlier impact
                aggregated[feature_name] = float(np.median(scores))
            else:
                aggregated[feature_name] = 0.0
        
        # Normalize scores to [0, 1]
        max_score = max(aggregated.values()) if aggregated else 1.0
        if max_score > 0:
            aggregated = {k: v / max_score for k, v in aggregated.items()}
        
        return aggregated
    
    def analyze_coordination_effectiveness_features(self) -> FeatureImportanceResult:
        """Analyze features affecting coordination effectiveness."""
        # Get coordination decisions
        coordination_decisions = self.decision_tracker.get_decisions_by_type("coordination")
        
        if len(coordination_decisions) < 2:
            return FeatureImportanceResult(
                method="insufficient_coordination_data",
                feature_scores={},
                feature_rankings=[],
                baseline_score=0.0,
                analysis_metadata={"error": "Insufficient coordination decisions"},
                timestamp=time.time()
            )
        
        # Similar analysis to agent selection but focused on coordination metrics
        return self._analyze_generic_features(coordination_decisions, "coordination_effectiveness")
    
    def _analyze_generic_features(self, decisions: List, analysis_type: str) -> FeatureImportanceResult:
        """Generic feature analysis for any decision type."""
        feature_data = []
        outcomes = []
        
        for decision in decisions:
            features = self._extract_decision_features(decision)
            outcome = self._get_decision_outcome_score(decision.decision_id)
            
            if features and outcome is not None:
                feature_data.append(features)
                outcomes.append(outcome)
        
        if len(feature_data) < 2:
            return FeatureImportanceResult(
                method="insufficient_data",
                feature_scores={},
                feature_rankings=[],
                baseline_score=0.0,
                analysis_metadata={"error": f"Insufficient data for {analysis_type}"},
                timestamp=time.time()
            )
        
        # Perform analysis
        df = pd.DataFrame(feature_data)
        feature_names = list(df.columns)
        X = df.values
        y = np.array(outcomes)
        
        # Use correlation method for quick analysis
        importance_scores = self._calculate_correlation_importance(X, y, feature_names)
        feature_rankings = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return FeatureImportanceResult(
            method="correlation_analysis",
            feature_scores=importance_scores,
            feature_rankings=feature_rankings,
            baseline_score=np.mean(y),
            analysis_metadata={
                "analysis_type": analysis_type,
                "decisions_analyzed": len(decisions),
                "features_extracted": len(feature_names)
            },
            timestamp=time.time()
        )
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get a summary of all feature importance analyses."""
        agent_importance = self.analyze_agent_selection_features()
        
        summary = {
            "timestamp": time.time(),
            "agent_selection": {
                "top_features": agent_importance.feature_rankings[:5] if agent_importance.feature_rankings else [],
                "baseline_score": agent_importance.baseline_score,
                "metadata": agent_importance.analysis_metadata
            },
            "feature_categories": self.feature_categories,
            "available_methods": list(self.importance_methods.keys())
        }
        
        return summary


# Global feature analyzer instance
_global_feature_analyzer = SwarmFeatureAnalyzer()

def get_global_feature_analyzer() -> SwarmFeatureAnalyzer:
    """Get the global feature analyzer instance."""
    return _global_feature_analyzer

def analyze_agent_selection_importance(hours: int = 24) -> FeatureImportanceResult:
    """Convenience function for analyzing agent selection feature importance."""
    return _global_feature_analyzer.analyze_agent_selection_features(hours)