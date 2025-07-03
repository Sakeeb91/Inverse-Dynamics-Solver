"""
Explainable AI module for swarm intelligence system.

This module provides comprehensive explainability and regulatory compliance
capabilities for the swarm intelligence platform.
"""

from .decision_tracker import DecisionTracker, get_global_tracker, decision_tracker_decorator
from .audit_logger import AuditLogger, get_global_audit_logger, audit_log, AuditLevel, ComplianceFramework
from .explainer_engine import SwarmDecisionExplainer, get_global_explainer, explain_decision
from .feature_importance import SwarmFeatureAnalyzer, get_global_feature_analyzer, analyze_agent_selection_importance

__all__ = [
    'DecisionTracker',
    'get_global_tracker', 
    'decision_tracker_decorator',
    'AuditLogger',
    'get_global_audit_logger',
    'audit_log',
    'AuditLevel',
    'ComplianceFramework',
    'SwarmDecisionExplainer',
    'get_global_explainer',
    'explain_decision',
    'SwarmFeatureAnalyzer',
    'get_global_feature_analyzer',
    'analyze_agent_selection_importance'
]

__version__ = '1.0.0'