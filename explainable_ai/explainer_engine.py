#!/usr/bin/env python3
"""
SHAP/LIME Explainability Engine for Swarm Intelligence

This module provides comprehensive explainability for machine learning
decisions in the swarm intelligence system using SHAP and LIME.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
import warnings
warnings.filterwarnings('ignore')

# Machine learning explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")

from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor
import json
import time
from dataclasses import dataclass

from .decision_tracker import get_global_tracker
from .audit_logger import audit_log, AuditLevel


@dataclass
class ExplanationResult:
    """Container for explanation results."""
    method: str
    feature_importance: Dict[str, float]
    explanation_data: Dict[str, Any]
    confidence_score: float
    prediction: Any
    baseline_prediction: Any
    timestamp: float
    computation_time_ms: float


class SwarmMLExplainerWrapper:
    """
    Wrapper for machine learning models to provide explainability features.
    
    This class wraps existing ML models (like TrebuchetController) to provide
    SHAP and LIME explanations for their predictions.
    """
    
    def __init__(self, model: BaseEstimator, feature_names: List[str] = None):
        """
        Initialize the explainer wrapper.
        
        Args:
            model: Scikit-learn compatible model to explain
            feature_names: Names of input features
        """
        self.model = model
        self.feature_names = feature_names or []
        self.is_fitted = hasattr(model, 'predict')
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.training_data = None
        self.explanation_cache = {}
        
    def set_training_data(self, X: np.ndarray, feature_names: List[str] = None):
        """
        Set training data for explainer initialization.
        
        Args:
            X: Training feature matrix
            feature_names: Optional feature names
        """
        self.training_data = X
        if feature_names:
            self.feature_names = feature_names
        
        # Initialize explainers with training data
        self._initialize_explainers(X)
    
    def _initialize_explainers(self, X: np.ndarray):
        """Initialize SHAP and LIME explainers with training data."""
        try:
            if SHAP_AVAILABLE and self.is_fitted:
                # Use TreeExplainer for tree models, KernelExplainer for others
                if hasattr(self.model, 'tree_'):
                    self.shap_explainer = shap.TreeExplainer(self.model)
                else:
                    # Sample background data for KernelExplainer
                    background_size = min(100, len(X))
                    background_indices = np.random.choice(len(X), background_size, replace=False)
                    background_data = X[background_indices]
                    self.shap_explainer = shap.KernelExplainer(self.model.predict, background_data)
                    
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
        
        try:
            if LIME_AVAILABLE and self.is_fitted:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X,
                    feature_names=self.feature_names,
                    mode='regression',
                    discretize_continuous=True
                )
        except Exception as e:
            print(f"Warning: Could not initialize LIME explainer: {e}")
    
    def explain_prediction(self, 
                          X: np.ndarray, 
                          method: str = "shap",
                          num_features: int = None) -> ExplanationResult:
        """
        Generate explanation for a prediction.
        
        Args:
            X: Input features for prediction
            method: Explanation method ("shap", "lime", or "both")
            num_features: Number of top features to include
            
        Returns:
            ExplanationResult containing explanation data
        """
        start_time = time.time()
        
        # Make prediction
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        prediction = self.model.predict(X)
        
        # Generate explanation based on method
        if method.lower() == "shap" and self.shap_explainer:
            explanation = self._explain_with_shap(X, num_features)
        elif method.lower() == "lime" and self.lime_explainer:
            explanation = self._explain_with_lime(X, num_features)
        elif method.lower() == "both":
            explanation = self._explain_with_both(X, num_features)
        else:
            explanation = self._explain_fallback(X, num_features)
        
        computation_time = (time.time() - start_time) * 1000
        
        # Create result
        result = ExplanationResult(
            method=method,
            feature_importance=explanation.get('feature_importance', {}),
            explanation_data=explanation,
            confidence_score=explanation.get('confidence', 0.5),
            prediction=prediction[0] if len(prediction) == 1 else prediction,
            baseline_prediction=explanation.get('baseline', None),
            timestamp=time.time(),
            computation_time_ms=computation_time
        )
        
        # Log explanation for audit
        audit_log(
            event_type="ml_explanation",
            actor="SwarmMLExplainerWrapper",
            action="generate_explanation",
            resource=f"model_{type(self.model).__name__}",
            outcome="success",
            details={
                "method": method,
                "num_features_explained": len(result.feature_importance),
                "confidence": result.confidence_score,
                "computation_time_ms": computation_time
            },
            audit_level=AuditLevel.COMPLIANCE
        )
        
        return result
    
    def _explain_with_shap(self, X: np.ndarray, num_features: int = None) -> Dict[str, Any]:
        """Generate SHAP-based explanation."""
        try:
            shap_values = self.shap_explainer.shap_values(X)
            
            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # Take first sample
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, importance in enumerate(shap_values):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                feature_importance[feature_name] = float(importance)
            
            # Sort by absolute importance
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
            
            if num_features:
                sorted_features = sorted_features[:num_features]
            
            return {
                'feature_importance': dict(sorted_features),
                'shap_values': shap_values.tolist(),
                'baseline': float(self.shap_explainer.expected_value) if hasattr(self.shap_explainer, 'expected_value') else 0.0,
                'confidence': min(1.0, 1.0 - np.std(shap_values) / (np.mean(np.abs(shap_values)) + 1e-8)),
                'method_specific': {
                    'explainer_type': type(self.shap_explainer).__name__,
                    'shap_values_raw': shap_values.tolist()
                }
            }
            
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return self._explain_fallback(X, num_features)
    
    def _explain_with_lime(self, X: np.ndarray, num_features: int = None) -> Dict[str, Any]:
        """Generate LIME-based explanation."""
        try:
            # LIME explains one instance at a time
            instance = X[0] if X.ndim > 1 else X
            
            explanation = self.lime_explainer.explain_instance(
                instance,
                self.model.predict,
                num_features=num_features or len(self.feature_names),
                num_samples=500
            )
            
            # Extract feature importance
            feature_importance = {}
            for feature_idx, importance in explanation.as_list():
                if isinstance(feature_idx, str):
                    feature_name = feature_idx
                else:
                    feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                feature_importance[feature_name] = float(importance)
            
            # Calculate confidence based on explanation score
            score = explanation.score if hasattr(explanation, 'score') else 0.5
            
            return {
                'feature_importance': feature_importance,
                'baseline': 0.0,  # LIME doesn't provide baseline
                'confidence': float(score),
                'method_specific': {
                    'local_prediction': float(explanation.local_pred[0]) if hasattr(explanation, 'local_pred') else None,
                    'intercept': float(explanation.intercept[0]) if hasattr(explanation, 'intercept') else None,
                    'explanation_score': float(score)
                }
            }
            
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return self._explain_fallback(X, num_features)
    
    def _explain_with_both(self, X: np.ndarray, num_features: int = None) -> Dict[str, Any]:
        """Generate explanations using both SHAP and LIME."""
        shap_explanation = self._explain_with_shap(X, num_features)
        lime_explanation = self._explain_with_lime(X, num_features)
        
        # Combine feature importance (average)
        all_features = set(shap_explanation.get('feature_importance', {}).keys()) | \
                      set(lime_explanation.get('feature_importance', {}).keys())
        
        combined_importance = {}
        for feature in all_features:
            shap_value = shap_explanation.get('feature_importance', {}).get(feature, 0.0)
            lime_value = lime_explanation.get('feature_importance', {}).get(feature, 0.0)
            combined_importance[feature] = (shap_value + lime_value) / 2
        
        # Combined confidence
        shap_conf = shap_explanation.get('confidence', 0.5)
        lime_conf = lime_explanation.get('confidence', 0.5)
        combined_confidence = (shap_conf + lime_conf) / 2
        
        return {
            'feature_importance': combined_importance,
            'baseline': shap_explanation.get('baseline', 0.0),
            'confidence': combined_confidence,
            'method_specific': {
                'shap_results': shap_explanation,
                'lime_results': lime_explanation,
                'consensus_score': abs(shap_conf - lime_conf)  # Lower is better consensus
            }
        }
    
    def _explain_fallback(self, X: np.ndarray, num_features: int = None) -> Dict[str, Any]:
        """Fallback explanation when SHAP/LIME are not available."""
        # Use simple feature magnitude as importance
        instance = X[0] if X.ndim > 1 else X
        
        feature_importance = {}
        for i, value in enumerate(instance):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            feature_importance[feature_name] = float(abs(value))
        
        # Normalize
        max_importance = max(feature_importance.values()) if feature_importance else 1.0
        feature_importance = {k: v / max_importance for k, v in feature_importance.items()}
        
        # Sort and limit
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        if num_features:
            sorted_features = sorted_features[:num_features]
        
        return {
            'feature_importance': dict(sorted_features),
            'baseline': 0.0,
            'confidence': 0.3,  # Low confidence for fallback method
            'method_specific': {
                'method': 'magnitude_based_fallback',
                'note': 'Using feature magnitude as importance due to unavailable explainers'
            }
        }


class SwarmDecisionExplainer:
    """
    Comprehensive explainer for swarm intelligence decisions.
    
    This class provides explainability for high-level swarm decisions
    including agent selection, coordination strategies, and mission planning.
    """
    
    def __init__(self):
        """Initialize the swarm decision explainer."""
        self.ml_explainers = {}  # Model wrappers
        self.decision_tracker = get_global_tracker()
        self.explanation_history = []
        
        # Decision explanation templates
        self.explanation_templates = {
            'agent_selection': self._explain_agent_selection,
            'coordination': self._explain_coordination_strategy,
            'parameter_optimization': self._explain_parameter_optimization,
            'mission_planning': self._explain_mission_planning
        }
    
    def register_ml_model(self, model_name: str, model: BaseEstimator, 
                         feature_names: List[str] = None):
        """
        Register a machine learning model for explanation.
        
        Args:
            model_name: Unique identifier for the model
            model: The ML model to explain
            feature_names: Names of input features
        """
        self.ml_explainers[model_name] = SwarmMLExplainerWrapper(model, feature_names)
        
        audit_log(
            event_type="model_registration",
            actor="SwarmDecisionExplainer",
            action="register_ml_model",
            resource=model_name,
            outcome="success",
            details={
                "model_type": type(model).__name__,
                "feature_count": len(feature_names) if feature_names else 0,
                "is_fitted": hasattr(model, 'predict')
            },
            audit_level=AuditLevel.BASIC
        )
    
    def explain_decision(self, decision_id: str, 
                        explanation_depth: str = "standard") -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a tracked decision.
        
        Args:
            decision_id: ID of the decision to explain
            explanation_depth: "basic", "standard", or "detailed"
            
        Returns:
            Comprehensive explanation including context and reasoning
        """
        # Get decision context
        decision_context = self.decision_tracker.get_decision_context(decision_id)
        decision_outcome = self.decision_tracker.get_decision_outcome(decision_id)
        
        if not decision_context:
            return {"error": "Decision not found", "decision_id": decision_id}
        
        # Generate explanation based on decision type
        explanation_func = self.explanation_templates.get(
            decision_context.decision_type,
            self._explain_generic_decision
        )
        
        explanation = explanation_func(decision_context, decision_outcome, explanation_depth)
        
        # Add metadata
        explanation.update({
            "decision_id": decision_id,
            "explanation_timestamp": time.time(),
            "explanation_depth": explanation_depth,
            "explainer_version": "1.0.0"
        })
        
        # Store explanation
        self.explanation_history.append(explanation)
        
        # Log explanation generation
        audit_log(
            event_type="decision_explanation",
            actor="SwarmDecisionExplainer",
            action="generate_explanation",
            resource=decision_id,
            outcome="success",
            details={
                "decision_type": decision_context.decision_type,
                "explanation_depth": explanation_depth,
                "reasoning_steps": len(decision_context.reasoning_chain)
            },
            audit_level=AuditLevel.COMPLIANCE
        )
        
        return explanation
    
    def _explain_agent_selection(self, context, outcome, depth: str) -> Dict[str, Any]:
        """Explain agent selection decisions."""
        explanation = {
            "decision_type": "Agent Selection",
            "summary": "Selection of optimal agents for mission execution",
            "key_factors": [],
            "reasoning_analysis": {},
            "alternatives": {},
            "impact_assessment": {}
        }
        
        # Analyze input parameters
        input_params = context.input_parameters
        targets = input_params.get('targets', [])
        mission_type = input_params.get('mission_type', 'unknown')
        
        explanation["key_factors"] = [
            f"Mission type: {mission_type}",
            f"Number of targets: {len(targets)}",
            f"Available agents: {input_params.get('available_agents', 'unknown')}",
            f"Mission complexity: {len(targets)} targets requiring coordination"
        ]
        
        # Analyze reasoning chain
        reasoning_analysis = {
            "total_steps": len(context.reasoning_chain),
            "decision_flow": context.reasoning_chain,
            "confidence_evolution": [context.confidence_score]  # Could track evolution
        }
        
        if depth in ["standard", "detailed"]:
            # Analyze agent states
            agent_states = context.agent_states
            if agent_states:
                specializations = agent_states.get('specializations', {})
                explanation["agent_composition"] = {
                    "specialization_distribution": specializations,
                    "average_energy": agent_states.get('average_energy', 0),
                    "total_available": sum(specializations.values())
                }
        
        if depth == "detailed":
            # Environmental factors
            env_factors = context.environmental_factors
            explanation["environmental_considerations"] = {
                "field_size": env_factors.get('field_size', 'unknown'),
                "communication_range": env_factors.get('communication_range', 'unknown')
            }
            
            # Outcome analysis
            if outcome:
                explanation["outcome_analysis"] = {
                    "result": outcome.outcome,
                    "success": outcome.success,
                    "impact_metrics": outcome.impact_metrics
                }
        
        explanation["reasoning_analysis"] = reasoning_analysis
        return explanation
    
    def _explain_coordination_strategy(self, context, outcome, depth: str) -> Dict[str, Any]:
        """Explain coordination strategy decisions."""
        return {
            "decision_type": "Coordination Strategy",
            "summary": "Development of multi-agent coordination approach",
            "key_factors": ["Agent capabilities", "Target distribution", "Communication constraints"],
            "reasoning_analysis": {
                "coordination_method": "distributed_consensus",
                "formation_strategy": "adaptive_positioning",
                "timing_coordination": "synchronized_execution"
            }
        }
    
    def _explain_parameter_optimization(self, context, outcome, depth: str) -> Dict[str, Any]:
        """Explain parameter optimization decisions."""
        return {
            "decision_type": "Parameter Optimization",
            "summary": "Optimization of system parameters for improved performance",
            "key_factors": ["Performance history", "Environmental conditions", "Success metrics"],
            "reasoning_analysis": {
                "optimization_method": "gradient_based",
                "convergence_criteria": "performance_threshold",
                "parameter_space": "continuous_bounded"
            }
        }
    
    def _explain_mission_planning(self, context, outcome, depth: str) -> Dict[str, Any]:
        """Explain mission planning decisions."""
        return {
            "decision_type": "Mission Planning",
            "summary": "Strategic planning for mission execution",
            "key_factors": ["Target priorities", "Resource allocation", "Risk assessment"],
            "reasoning_analysis": {
                "planning_horizon": "tactical",
                "risk_tolerance": "moderate",
                "success_criteria": "target_acquisition"
            }
        }
    
    def _explain_generic_decision(self, context, outcome, depth: str) -> Dict[str, Any]:
        """Generic explanation for unknown decision types."""
        return {
            "decision_type": f"Generic ({context.decision_type})",
            "summary": f"Decision made by {context.decision_maker}",
            "key_factors": list(context.input_parameters.keys()),
            "reasoning_analysis": {
                "reasoning_steps": len(context.reasoning_chain),
                "reasoning_chain": context.reasoning_chain,
                "confidence": context.confidence_score
            },
            "execution_time": f"{context.execution_time_ms:.2f}ms"
        }
    
    def explain_ml_prediction(self, model_name: str, X: np.ndarray, 
                            method: str = "shap") -> ExplanationResult:
        """
        Explain a machine learning prediction.
        
        Args:
            model_name: Name of registered model
            X: Input features
            method: Explanation method
            
        Returns:
            ExplanationResult with prediction explanation
        """
        if model_name not in self.ml_explainers:
            raise ValueError(f"Model '{model_name}' not registered")
        
        explainer = self.ml_explainers[model_name]
        return explainer.explain_prediction(X, method)
    
    def generate_explanation_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Generate summary of all explanations in a time window.
        
        Args:
            time_window_hours: Hours to look back
            
        Returns:
            Summary of explanation activity and patterns
        """
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_explanations = [
            exp for exp in self.explanation_history
            if exp.get('explanation_timestamp', 0) >= cutoff_time
        ]
        
        summary = {
            "time_window_hours": time_window_hours,
            "total_explanations": len(recent_explanations),
            "explanation_types": {},
            "most_explained_decisions": [],
            "average_explanation_depth": "standard"
        }
        
        # Analyze explanation types
        for exp in recent_explanations:
            decision_type = exp.get('decision_type', 'unknown')
            summary["explanation_types"][decision_type] = \
                summary["explanation_types"].get(decision_type, 0) + 1
        
        return summary


# Global explainer instance
_global_explainer = SwarmDecisionExplainer()

def get_global_explainer() -> SwarmDecisionExplainer:
    """Get the global decision explainer instance."""
    return _global_explainer

def explain_decision(decision_id: str, depth: str = "standard") -> Dict[str, Any]:
    """Convenience function for explaining decisions."""
    return _global_explainer.explain_decision(decision_id, depth)