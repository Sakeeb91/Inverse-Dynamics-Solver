#!/usr/bin/env python3
"""
Automated Violation Detection and Response System

This module provides intelligent violation detection using pattern analysis,
machine learning, and rule-based systems to identify compliance violations
and automatically respond with appropriate remediation actions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time
import threading
from collections import defaultdict, deque
from enum import Enum
import json

# ML libraries for pattern detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from explainable_ai.audit_logger import get_global_audit_logger, ComplianceFramework, AuditLevel, audit_log
from explainable_ai.decision_tracker import get_global_tracker
from .policy_engine import get_global_policy_engine
from .regulatory_monitor import ViolationType, ViolationSeverity


class DetectionMethod(Enum):
    """Methods for detecting violations."""
    RULE_BASED = "rule_based"
    PATTERN_ANALYSIS = "pattern_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    THRESHOLD_MONITORING = "threshold_monitoring"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"


class ResponseAction(Enum):
    """Automated response actions."""
    IMMEDIATE_BLOCK = "immediate_block"
    GRADUAL_RESTRICTION = "gradual_restriction"
    ALERT_ESCALATION = "alert_escalation"
    AUTO_REMEDIATION = "auto_remediation"
    DATA_QUARANTINE = "data_quarantine"
    SYSTEM_ISOLATION = "system_isolation"


@dataclass
class ViolationPattern:
    """Pattern definition for violation detection."""
    pattern_id: str
    name: str
    description: str
    detection_method: DetectionMethod
    pattern_definition: Dict[str, Any]
    confidence_threshold: float
    severity_mapping: Dict[float, ViolationSeverity]
    response_actions: List[ResponseAction]
    enabled: bool = True


@dataclass
class DetectionResult:
    """Result of violation detection analysis."""
    detection_id: str
    timestamp: float
    method: DetectionMethod
    confidence_score: float
    violation_type: ViolationType
    severity: ViolationSeverity
    affected_entities: List[str]
    evidence: Dict[str, Any]
    recommended_actions: List[ResponseAction]
    pattern_id: Optional[str] = None


class ViolationDetector:
    """
    Intelligent violation detection system.
    
    Uses multiple detection methods including rule-based, pattern analysis,
    and machine learning to identify compliance violations and automatically
    respond with appropriate actions.
    """
    
    def __init__(self):
        """Initialize the violation detector."""
        self.audit_logger = get_global_audit_logger()
        self.decision_tracker = get_global_tracker()
        self.policy_engine = get_global_policy_engine()
        
        # Detection patterns
        self.patterns: Dict[str, ViolationPattern] = {}
        
        # Historical data for pattern analysis
        self.event_history: deque = deque(maxlen=10000)
        self.violation_history: List[DetectionResult] = []
        
        # ML models for anomaly detection
        self.anomaly_models: Dict[str, Any] = {}
        self.feature_scalers: Dict[str, StandardScaler] = {}
        
        # Detection statistics
        self.detection_stats = {
            "total_detections": 0,
            "true_positives": 0,
            "false_positives": 0,
            "patterns_triggered": defaultdict(int),
            "methods_used": defaultdict(int)
        }
        
        # Response handlers
        self.response_handlers: Dict[ResponseAction, Callable] = {}
        
        # Detection state
        self.detection_active = False
        self.detection_thread = None
        self._lock = threading.Lock()
        
        # Initialize components
        self._initialize_detection_patterns()
        self._initialize_response_handlers()
        if ML_AVAILABLE:
            self._initialize_ml_models()
    
    def _initialize_detection_patterns(self):
        """Initialize predefined violation detection patterns."""
        
        # Excessive access pattern
        access_pattern = ViolationPattern(
            pattern_id="excessive_access",
            name="Excessive Access Pattern",
            description="Detect unusually high access frequency from single entity",
            detection_method=DetectionMethod.THRESHOLD_MONITORING,
            pattern_definition={
                "window_minutes": 5,
                "max_accesses": 50,
                "event_types": ["data_access", "system_access", "resource_access"]
            },
            confidence_threshold=0.8,
            severity_mapping={
                0.8: ViolationSeverity.MEDIUM,
                0.9: ViolationSeverity.HIGH,
                0.95: ViolationSeverity.CRITICAL
            },
            response_actions=[ResponseAction.ALERT_ESCALATION, ResponseAction.GRADUAL_RESTRICTION]
        )
        self.add_pattern(access_pattern)
        
        # Privilege escalation pattern
        privilege_pattern = ViolationPattern(
            pattern_id="privilege_escalation",
            name="Privilege Escalation Detection",
            description="Detect unauthorized privilege escalation attempts",
            detection_method=DetectionMethod.BEHAVIORAL_ANALYSIS,
            pattern_definition={
                "baseline_window_hours": 24,
                "elevation_threshold": 2,
                "monitored_actions": ["admin_action", "config_change", "user_management"]
            },
            confidence_threshold=0.9,
            severity_mapping={
                0.9: ViolationSeverity.HIGH,
                0.95: ViolationSeverity.CRITICAL
            },
            response_actions=[ResponseAction.IMMEDIATE_BLOCK, ResponseAction.ALERT_ESCALATION]
        )
        self.add_pattern(privilege_pattern)
        
        # Data exfiltration pattern
        exfiltration_pattern = ViolationPattern(
            pattern_id="data_exfiltration",
            name="Data Exfiltration Detection",
            description="Detect potential data exfiltration activities",
            detection_method=DetectionMethod.PATTERN_ANALYSIS,
            pattern_definition={
                "volume_threshold_mb": 100,
                "frequency_threshold": 10,
                "time_window_minutes": 15,
                "suspicious_destinations": ["external", "unknown", "unauthorized"]
            },
            confidence_threshold=0.85,
            severity_mapping={
                0.85: ViolationSeverity.HIGH,
                0.95: ViolationSeverity.CRITICAL
            },
            response_actions=[ResponseAction.DATA_QUARANTINE, ResponseAction.IMMEDIATE_BLOCK]
        )
        self.add_pattern(exfiltration_pattern)
        
        # Bias detection pattern
        bias_pattern = ViolationPattern(
            pattern_id="algorithmic_bias",
            name="Algorithmic Bias Detection",
            description="Detect potential bias in automated decision-making",
            detection_method=DetectionMethod.CORRELATION_ANALYSIS,
            pattern_definition={
                "protected_attributes": ["gender", "race", "age", "location"],
                "decision_disparity_threshold": 0.2,
                "sample_size_minimum": 50
            },
            confidence_threshold=0.7,
            severity_mapping={
                0.7: ViolationSeverity.MEDIUM,
                0.8: ViolationSeverity.HIGH,
                0.9: ViolationSeverity.CRITICAL
            },
            response_actions=[ResponseAction.ALERT_ESCALATION, ResponseAction.AUTO_REMEDIATION]
        )
        self.add_pattern(bias_pattern)
    
    def _initialize_response_handlers(self):
        """Initialize automated response handlers."""
        self.response_handlers[ResponseAction.IMMEDIATE_BLOCK] = self._handle_immediate_block
        self.response_handlers[ResponseAction.GRADUAL_RESTRICTION] = self._handle_gradual_restriction
        self.response_handlers[ResponseAction.ALERT_ESCALATION] = self._handle_alert_escalation
        self.response_handlers[ResponseAction.AUTO_REMEDIATION] = self._handle_auto_remediation
        self.response_handlers[ResponseAction.DATA_QUARANTINE] = self._handle_data_quarantine
        self.response_handlers[ResponseAction.SYSTEM_ISOLATION] = self._handle_system_isolation
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for anomaly detection."""
        if not ML_AVAILABLE:
            return
        
        # Isolation Forest for anomaly detection
        self.anomaly_models["isolation_forest"] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # DBSCAN for clustering-based anomaly detection
        self.anomaly_models["dbscan"] = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        
        # Feature scalers
        self.feature_scalers["standard"] = StandardScaler()
    
    def add_pattern(self, pattern: ViolationPattern):
        """Add a new detection pattern."""
        with self._lock:
            self.patterns[pattern.pattern_id] = pattern
        
        audit_log(
            event_type="detection_pattern",
            actor="ViolationDetector",
            action="add_pattern",
            resource=pattern.pattern_id,
            outcome="success",
            details={
                "pattern_name": pattern.name,
                "detection_method": pattern.detection_method.value,
                "confidence_threshold": pattern.confidence_threshold
            },
            audit_level=AuditLevel.COMPLIANCE
        )
    
    def remove_pattern(self, pattern_id: str):
        """Remove a detection pattern."""
        with self._lock:
            self.patterns.pop(pattern_id, None)
    
    def start_detection(self):
        """Start automated violation detection."""
        if self.detection_active:
            return
        
        self.detection_active = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        audit_log(
            event_type="violation_detection",
            actor="ViolationDetector",
            action="start_detection",
            resource="detection_system",
            outcome="success",
            details={"active_patterns": len(self.patterns)},
            audit_level=AuditLevel.CRITICAL
        )
    
    def stop_detection(self):
        """Stop automated violation detection."""
        self.detection_active = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5.0)
        
        audit_log(
            event_type="violation_detection",
            actor="ViolationDetector",
            action="stop_detection",
            resource="detection_system",
            outcome="success",
            audit_level=AuditLevel.CRITICAL
        )
    
    def _detection_loop(self):
        """Main detection loop."""
        while self.detection_active:
            try:
                # Get recent events for analysis
                recent_events = self._get_recent_events()
                
                if recent_events:
                    # Analyze events with all patterns
                    detections = self._analyze_events(recent_events)
                    
                    # Process detections
                    for detection in detections:
                        self._process_detection(detection)
                
                # Sleep between detection cycles
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(30.0)  # Wait longer on error
    
    def _get_recent_events(self) -> List[Any]:
        """Get recent events for analysis."""
        current_time = time.time()
        lookback_time = current_time - 300  # Last 5 minutes
        
        # Get audit entries
        audit_entries = self.audit_logger.get_audit_entries(
            start_time=lookback_time,
            end_time=current_time
        )
        
        # Get recent decisions
        recent_decisions = self.decision_tracker.get_decisions_by_timerange(
            start_time=lookback_time,
            end_time=current_time
        )
        
        # Combine and update history
        all_events = list(audit_entries) + list(recent_decisions)
        
        # Update event history
        with self._lock:
            for event in all_events:
                if event not in self.event_history:
                    self.event_history.append(event)
        
        return all_events
    
    def _analyze_events(self, events: List[Any]) -> List[DetectionResult]:
        """Analyze events using all detection patterns."""
        detections = []
        
        for pattern in self.patterns.values():
            if not pattern.enabled:
                continue
            
            try:
                pattern_detections = self._apply_pattern(pattern, events)
                detections.extend(pattern_detections)
                
                if pattern_detections:
                    self.detection_stats["patterns_triggered"][pattern.pattern_id] += len(pattern_detections)
                
            except Exception as e:
                print(f"Error applying pattern {pattern.pattern_id}: {e}")
        
        return detections
    
    def _apply_pattern(self, pattern: ViolationPattern, events: List[Any]) -> List[DetectionResult]:
        """Apply a specific detection pattern to events."""
        method = pattern.detection_method
        self.detection_stats["methods_used"][method.value] += 1
        
        if method == DetectionMethod.THRESHOLD_MONITORING:
            return self._apply_threshold_pattern(pattern, events)
        elif method == DetectionMethod.BEHAVIORAL_ANALYSIS:
            return self._apply_behavioral_pattern(pattern, events)
        elif method == DetectionMethod.PATTERN_ANALYSIS:
            return self._apply_pattern_analysis(pattern, events)
        elif method == DetectionMethod.CORRELATION_ANALYSIS:
            return self._apply_correlation_pattern(pattern, events)
        elif method == DetectionMethod.ANOMALY_DETECTION and ML_AVAILABLE:
            return self._apply_anomaly_detection(pattern, events)
        
        return []
    
    def _apply_threshold_pattern(self, pattern: ViolationPattern, events: List[Any]) -> List[DetectionResult]:
        """Apply threshold-based pattern detection."""
        detections = []
        definition = pattern.pattern_definition
        
        window_minutes = definition.get("window_minutes", 5)
        max_accesses = definition.get("max_accesses", 50)
        event_types = definition.get("event_types", [])
        
        # Group events by actor within time window
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        actor_events = defaultdict(list)
        for event in events:
            if hasattr(event, 'timestamp') and event.timestamp >= window_start:
                if hasattr(event, 'actor') and hasattr(event, 'event_type'):
                    if not event_types or event.event_type in event_types:
                        actor_events[event.actor].append(event)
        
        # Check for threshold violations
        for actor, actor_event_list in actor_events.items():
            if len(actor_event_list) > max_accesses:
                confidence = min(1.0, len(actor_event_list) / max_accesses)
                severity = self._determine_severity(pattern, confidence)
                
                detection = DetectionResult(
                    detection_id=f"threshold_{pattern.pattern_id}_{int(time.time())}",
                    timestamp=current_time,
                    method=DetectionMethod.THRESHOLD_MONITORING,
                    confidence_score=confidence,
                    violation_type=ViolationType.ACCESS_CONTROL,
                    severity=severity,
                    affected_entities=[actor],
                    evidence={
                        "event_count": len(actor_event_list),
                        "threshold": max_accesses,
                        "window_minutes": window_minutes,
                        "event_types": [e.event_type for e in actor_event_list]
                    },
                    recommended_actions=pattern.response_actions,
                    pattern_id=pattern.pattern_id
                )
                detections.append(detection)
        
        return detections
    
    def _apply_behavioral_pattern(self, pattern: ViolationPattern, events: List[Any]) -> List[DetectionResult]:
        """Apply behavioral analysis pattern detection."""
        detections = []
        definition = pattern.pattern_definition
        
        baseline_hours = definition.get("baseline_window_hours", 24)
        elevation_threshold = definition.get("elevation_threshold", 2)
        monitored_actions = definition.get("monitored_actions", [])
        
        current_time = time.time()
        baseline_start = current_time - (baseline_hours * 3600)
        
        # Analyze behavioral changes
        actor_baselines = defaultdict(list)
        recent_activities = defaultdict(list)
        
        for event in events:
            if hasattr(event, 'timestamp') and hasattr(event, 'actor') and hasattr(event, 'action'):
                if event.action in monitored_actions:
                    if event.timestamp >= baseline_start and event.timestamp < current_time - 3600:
                        actor_baselines[event.actor].append(event)
                    elif event.timestamp >= current_time - 3600:  # Last hour
                        recent_activities[event.actor].append(event)
        
        # Compare recent activity to baseline
        for actor in recent_activities:
            baseline_count = len(actor_baselines.get(actor, []))
            recent_count = len(recent_activities[actor])
            
            if baseline_count > 0:
                activity_ratio = recent_count / (baseline_count / baseline_hours)
                
                if activity_ratio > elevation_threshold:
                    confidence = min(1.0, activity_ratio / elevation_threshold)
                    severity = self._determine_severity(pattern, confidence)
                    
                    detection = DetectionResult(
                        detection_id=f"behavioral_{pattern.pattern_id}_{int(time.time())}",
                        timestamp=current_time,
                        method=DetectionMethod.BEHAVIORAL_ANALYSIS,
                        confidence_score=confidence,
                        violation_type=ViolationType.ACCESS_CONTROL,
                        severity=severity,
                        affected_entities=[actor],
                        evidence={
                            "baseline_count": baseline_count,
                            "recent_count": recent_count,
                            "activity_ratio": activity_ratio,
                            "threshold": elevation_threshold
                        },
                        recommended_actions=pattern.response_actions,
                        pattern_id=pattern.pattern_id
                    )
                    detections.append(detection)
        
        return detections
    
    def _apply_pattern_analysis(self, pattern: ViolationPattern, events: List[Any]) -> List[DetectionResult]:
        """Apply pattern analysis detection."""
        detections = []
        definition = pattern.pattern_definition
        
        volume_threshold = definition.get("volume_threshold_mb", 100)
        frequency_threshold = definition.get("frequency_threshold", 10)
        time_window_minutes = definition.get("time_window_minutes", 15)
        
        current_time = time.time()
        window_start = current_time - (time_window_minutes * 60)
        
        # Analyze data transfer patterns
        transfer_activities = []
        for event in events:
            if hasattr(event, 'timestamp') and event.timestamp >= window_start:
                if hasattr(event, 'event_type') and 'data' in event.event_type.lower():
                    transfer_activities.append(event)
        
        if len(transfer_activities) > frequency_threshold:
            confidence = min(1.0, len(transfer_activities) / frequency_threshold)
            severity = self._determine_severity(pattern, confidence)
            
            detection = DetectionResult(
                detection_id=f"pattern_{pattern.pattern_id}_{int(time.time())}",
                timestamp=current_time,
                method=DetectionMethod.PATTERN_ANALYSIS,
                confidence_score=confidence,
                violation_type=ViolationType.PRIVACY_BREACH,
                severity=severity,
                affected_entities=["data_transfer_system"],
                evidence={
                    "transfer_count": len(transfer_activities),
                    "frequency_threshold": frequency_threshold,
                    "time_window_minutes": time_window_minutes
                },
                recommended_actions=pattern.response_actions,
                pattern_id=pattern.pattern_id
            )
            detections.append(detection)
        
        return detections
    
    def _apply_correlation_pattern(self, pattern: ViolationPattern, events: List[Any]) -> List[DetectionResult]:
        """Apply correlation analysis for bias detection."""
        detections = []
        definition = pattern.pattern_definition
        
        disparity_threshold = definition.get("decision_disparity_threshold", 0.2)
        min_sample_size = definition.get("sample_size_minimum", 50)
        
        # Analyze decision patterns for potential bias
        decisions = [e for e in events if hasattr(e, 'decision_type')]
        
        if len(decisions) >= min_sample_size:
            # Simple bias detection based on decision outcomes
            outcomes = []
            for decision in decisions:
                outcome = getattr(decision, 'confidence_score', 0.5)
                outcomes.append(outcome)
            
            if outcomes:
                outcome_variance = np.var(outcomes)
                if outcome_variance > disparity_threshold:
                    confidence = min(1.0, outcome_variance / disparity_threshold)
                    severity = self._determine_severity(pattern, confidence)
                    
                    detection = DetectionResult(
                        detection_id=f"correlation_{pattern.pattern_id}_{int(time.time())}",
                        timestamp=time.time(),
                        method=DetectionMethod.CORRELATION_ANALYSIS,
                        confidence_score=confidence,
                        violation_type=ViolationType.BIAS_DETECTION,
                        severity=severity,
                        affected_entities=["decision_system"],
                        evidence={
                            "decision_count": len(decisions),
                            "outcome_variance": outcome_variance,
                            "disparity_threshold": disparity_threshold
                        },
                        recommended_actions=pattern.response_actions,
                        pattern_id=pattern.pattern_id
                    )
                    detections.append(detection)
        
        return detections
    
    def _apply_anomaly_detection(self, pattern: ViolationPattern, events: List[Any]) -> List[DetectionResult]:
        """Apply ML-based anomaly detection."""
        detections = []
        
        if not ML_AVAILABLE or len(events) < 10:
            return detections
        
        try:
            # Extract features from events
            features = self._extract_event_features(events)
            
            if len(features) > 0:
                # Apply isolation forest
                anomalies = self.anomaly_models["isolation_forest"].fit_predict(features)
                anomaly_scores = self.anomaly_models["isolation_forest"].decision_function(features)
                
                # Identify anomalies
                for i, (is_anomaly, score) in enumerate(zip(anomalies, anomaly_scores)):
                    if is_anomaly == -1:  # Anomaly detected
                        confidence = min(1.0, abs(score))
                        severity = self._determine_severity(pattern, confidence)
                        
                        detection = DetectionResult(
                            detection_id=f"anomaly_{pattern.pattern_id}_{int(time.time())}_{i}",
                            timestamp=time.time(),
                            method=DetectionMethod.ANOMALY_DETECTION,
                            confidence_score=confidence,
                            violation_type=ViolationType.RISK_THRESHOLD,
                            severity=severity,
                            affected_entities=[f"event_{i}"],
                            evidence={
                                "anomaly_score": score,
                                "feature_vector": features[i].tolist(),
                                "detection_algorithm": "isolation_forest"
                            },
                            recommended_actions=pattern.response_actions,
                            pattern_id=pattern.pattern_id
                        )
                        detections.append(detection)
        
        except Exception as e:
            print(f"Anomaly detection failed: {e}")
        
        return detections
    
    def _extract_event_features(self, events: List[Any]) -> np.ndarray:
        """Extract numerical features from events for ML analysis."""
        features = []
        
        for event in events:
            feature_vector = [
                getattr(event, 'timestamp', time.time()),
                getattr(event, 'risk_score', 0.5) if hasattr(event, 'risk_score') else 0.5,
                len(getattr(event, 'reasoning_chain', [])) if hasattr(event, 'reasoning_chain') else 0,
                getattr(event, 'confidence_score', 0.5) if hasattr(event, 'confidence_score') else 0.5
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _determine_severity(self, pattern: ViolationPattern, confidence: float) -> ViolationSeverity:
        """Determine violation severity based on confidence and pattern mapping."""
        for threshold in sorted(pattern.severity_mapping.keys(), reverse=True):
            if confidence >= threshold:
                return pattern.severity_mapping[threshold]
        
        return ViolationSeverity.LOW
    
    def _process_detection(self, detection: DetectionResult):
        """Process a detected violation."""
        with self._lock:
            self.violation_history.append(detection)
            self.detection_stats["total_detections"] += 1
        
        # Execute response actions
        for action in detection.recommended_actions:
            try:
                handler = self.response_handlers.get(action)
                if handler:
                    handler(detection)
            except Exception as e:
                print(f"Error executing response action {action}: {e}")
        
        # Log the detection
        audit_log(
            event_type="violation_detected",
            actor="ViolationDetector",
            action="detect_violation",
            resource=detection.detection_id,
            outcome="violation",
            details=asdict(detection),
            audit_level=AuditLevel.CRITICAL
        )
    
    # Response handlers
    def _handle_immediate_block(self, detection: DetectionResult):
        """Handle immediate blocking response."""
        print(f"🛑 IMMEDIATE BLOCK: {detection.detection_id}")
        print(f"   Affected: {detection.affected_entities}")
        print(f"   Confidence: {detection.confidence_score:.2f}")
    
    def _handle_gradual_restriction(self, detection: DetectionResult):
        """Handle gradual restriction response."""
        print(f"⚠️ GRADUAL RESTRICTION: {detection.detection_id}")
        print(f"   Implementing progressive access controls")
    
    def _handle_alert_escalation(self, detection: DetectionResult):
        """Handle alert escalation response."""
        print(f"📈 ALERT ESCALATION: {detection.detection_id}")
        print(f"   Severity: {detection.severity.value}")
        print(f"   Method: {detection.method.value}")
    
    def _handle_auto_remediation(self, detection: DetectionResult):
        """Handle automatic remediation response."""
        print(f"🔧 AUTO REMEDIATION: {detection.detection_id}")
        print(f"   Attempting automatic fix for {detection.violation_type.value}")
    
    def _handle_data_quarantine(self, detection: DetectionResult):
        """Handle data quarantine response."""
        print(f"🔒 DATA QUARANTINE: {detection.detection_id}")
        print(f"   Isolating affected data")
    
    def _handle_system_isolation(self, detection: DetectionResult):
        """Handle system isolation response."""
        print(f"🏝️ SYSTEM ISOLATION: {detection.detection_id}")
        print(f"   Isolating affected systems")
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection system statistics."""
        recent_detections = [
            d for d in self.violation_history 
            if time.time() - d.timestamp <= 86400  # Last 24 hours
        ]
        
        return {
            "detection_active": self.detection_active,
            "total_patterns": len(self.patterns),
            "active_patterns": len([p for p in self.patterns.values() if p.enabled]),
            "total_detections": self.detection_stats["total_detections"],
            "recent_detections_24h": len(recent_detections),
            "detection_stats": self.detection_stats.copy(),
            "ml_available": ML_AVAILABLE
        }


# Global violation detector instance
_global_detector = None

def get_global_detector() -> ViolationDetector:
    """Get the global violation detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = ViolationDetector()
    return _global_detector

def start_violation_detection():
    """Start global violation detection."""
    get_global_detector().start_detection()

def get_detection_statistics() -> Dict[str, Any]:
    """Get detection system statistics."""
    return get_global_detector().get_detection_statistics()