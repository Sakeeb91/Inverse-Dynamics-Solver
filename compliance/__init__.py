"""
Compliance module for regulatory framework support.

This module provides comprehensive regulatory compliance monitoring,
policy management, and automated violation detection for multiple
compliance frameworks including GDPR, SOX, HIPAA, FAA, and FDA.
"""

from .regulatory_monitor import RegulatoryMonitor, get_global_monitor, start_compliance_monitoring, get_compliance_status
from .policy_engine import PolicyEngine, get_global_policy_engine, evaluate_compliance_policies
from .violation_detector import ViolationDetector, get_global_detector, start_violation_detection, get_detection_statistics

__all__ = [
    'RegulatoryMonitor',
    'get_global_monitor',
    'start_compliance_monitoring',
    'get_compliance_status',
    'PolicyEngine',
    'get_global_policy_engine',
    'evaluate_compliance_policies',
    'ViolationDetector',
    'get_global_detector',
    'start_violation_detection',
    'get_detection_statistics'
]

__version__ = '1.0.0'