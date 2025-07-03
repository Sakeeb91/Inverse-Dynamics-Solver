#!/usr/bin/env python3
"""
Regulatory Compliance Monitoring System

This module provides real-time monitoring and alerting for regulatory
compliance requirements across multiple frameworks including GDPR, SOX,
HIPAA, FAA, and FDA regulations.
"""

import time
import json
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque
import warnings

from explainable_ai.audit_logger import get_global_audit_logger, ComplianceFramework, AuditLevel
from explainable_ai.decision_tracker import get_global_tracker


class ViolationType(Enum):
    """Types of compliance violations."""
    DATA_RETENTION = "data_retention"
    ACCESS_CONTROL = "access_control"
    PRIVACY_BREACH = "privacy_breach"
    AUDIT_TRAIL = "audit_trail"
    OPERATIONAL_SAFETY = "operational_safety"
    TRANSPARENCY = "transparency"
    BIAS_DETECTION = "bias_detection"
    RISK_THRESHOLD = "risk_threshold"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceViolation:
    """Container for compliance violation details."""
    violation_id: str
    timestamp: float
    violation_type: ViolationType
    severity: ViolationSeverity
    framework: ComplianceFramework
    description: str
    affected_resources: List[str]
    detection_method: str
    recommended_actions: List[str]
    auto_remediation_possible: bool
    metadata: Dict[str, Any]


@dataclass
class ComplianceRule:
    """Definition of a compliance rule."""
    rule_id: str
    framework: ComplianceFramework
    rule_type: ViolationType
    condition: Callable[[Any], bool]
    threshold: float
    description: str
    severity: ViolationSeverity
    auto_remediation: Optional[Callable] = None
    enabled: bool = True


class RegulatoryMonitor:
    """
    Real-time regulatory compliance monitoring system.
    
    Monitors system activities, decisions, and data handling for compliance
    with multiple regulatory frameworks and generates alerts for violations.
    """
    
    def __init__(self, enabled_frameworks: List[ComplianceFramework] = None):
        """
        Initialize the regulatory monitor.
        
        Args:
            enabled_frameworks: List of compliance frameworks to monitor
        """
        self.enabled_frameworks = enabled_frameworks or [ComplianceFramework.CUSTOM]
        self.audit_logger = get_global_audit_logger()
        self.decision_tracker = get_global_tracker()
        
        # Violation tracking
        self.violations: List[ComplianceViolation] = []
        self.violation_history: Dict[str, List[ComplianceViolation]] = defaultdict(list)
        self.violation_counts: Dict[ViolationType, int] = defaultdict(int)
        
        # Rules engine
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        self._lock = threading.Lock()
        
        # Alert handlers
        self.alert_handlers: List[Callable[[ComplianceViolation], None]] = []
        
        # Monitoring intervals (seconds)
        self.monitoring_intervals = {
            'real_time': 1.0,
            'periodic': 60.0,
            'daily': 86400.0
        }
        
        # Data retention policies
        self.retention_policies = self._load_retention_policies()
        
        # Performance tracking
        self.monitoring_stats = {
            'checks_performed': 0,
            'violations_detected': 0,
            'auto_remediations': 0,
            'last_check_time': 0.0
        }
        
        # Initialize framework-specific rules
        self._initialize_compliance_rules()
    
    def _load_retention_policies(self) -> Dict[ComplianceFramework, Dict[str, int]]:
        """Load data retention policies for different frameworks."""
        return {
            ComplianceFramework.GDPR: {
                'personal_data': 365,  # days
                'audit_logs': 1095,    # 3 years
                'consent_records': 2555  # 7 years
            },
            ComplianceFramework.SOX: {
                'financial_records': 2555,  # 7 years
                'audit_trails': 2555,
                'control_documentation': 2555
            },
            ComplianceFramework.HIPAA: {
                'health_records': 2190,  # 6 years
                'access_logs': 2190,
                'security_documentation': 2190
            },
            ComplianceFramework.FAA: {
                'flight_data': 90,      # 90 days
                'maintenance_records': 365,
                'incident_reports': 1825  # 5 years
            },
            ComplianceFramework.FDA: {
                'clinical_data': 2555,   # 7 years
                'quality_records': 1095, # 3 years
                'audit_documentation': 2555
            }
        }
    
    def _initialize_compliance_rules(self):
        """Initialize compliance rules for enabled frameworks."""
        for framework in self.enabled_frameworks:
            if framework == ComplianceFramework.GDPR:
                self._add_gdpr_rules()
            elif framework == ComplianceFramework.SOX:
                self._add_sox_rules()
            elif framework == ComplianceFramework.HIPAA:
                self._add_hipaa_rules()
            elif framework == ComplianceFramework.FAA:
                self._add_faa_rules()
            elif framework == ComplianceFramework.FDA:
                self._add_fda_rules()
            else:
                self._add_custom_rules()
    
    def _add_gdpr_rules(self):
        """Add GDPR-specific compliance rules."""
        # Data retention rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="gdpr_data_retention",
            framework=ComplianceFramework.GDPR,
            rule_type=ViolationType.DATA_RETENTION,
            condition=lambda data: self._check_data_retention(data, ComplianceFramework.GDPR),
            threshold=365.0,  # days
            description="GDPR Article 5(e) - Data retention limitation",
            severity=ViolationSeverity.HIGH
        ))
        
        # Privacy by design rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="gdpr_privacy_by_design",
            framework=ComplianceFramework.GDPR,
            rule_type=ViolationType.PRIVACY_BREACH,
            condition=lambda data: self._check_privacy_protection(data),
            threshold=0.0,
            description="GDPR Article 25 - Data protection by design and by default",
            severity=ViolationSeverity.CRITICAL
        ))
        
        # Transparency rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="gdpr_transparency",
            framework=ComplianceFramework.GDPR,
            rule_type=ViolationType.TRANSPARENCY,
            condition=lambda data: self._check_decision_transparency(data),
            threshold=0.8,  # transparency score
            description="GDPR Article 22 - Automated decision-making transparency",
            severity=ViolationSeverity.MEDIUM
        ))
    
    def _add_sox_rules(self):
        """Add SOX-specific compliance rules."""
        # Internal controls rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="sox_internal_controls",
            framework=ComplianceFramework.SOX,
            rule_type=ViolationType.AUDIT_TRAIL,
            condition=lambda data: self._check_internal_controls(data),
            threshold=1.0,  # strict compliance
            description="SOX Section 404 - Internal control over financial reporting",
            severity=ViolationSeverity.HIGH
        ))
        
        # Audit trail completeness
        self.add_compliance_rule(ComplianceRule(
            rule_id="sox_audit_completeness",
            framework=ComplianceFramework.SOX,
            rule_type=ViolationType.AUDIT_TRAIL,
            condition=lambda data: self._check_audit_completeness(data),
            threshold=0.95,  # 95% completeness required
            description="SOX Section 302 - Disclosure controls and procedures",
            severity=ViolationSeverity.CRITICAL
        ))
    
    def _add_hipaa_rules(self):
        """Add HIPAA-specific compliance rules."""
        # Access control rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="hipaa_access_control",
            framework=ComplianceFramework.HIPAA,
            rule_type=ViolationType.ACCESS_CONTROL,
            condition=lambda data: self._check_access_controls(data),
            threshold=1.0,  # strict access control
            description="HIPAA Security Rule - Access Control (164.312(a))",
            severity=ViolationSeverity.HIGH
        ))
        
        # Audit controls
        self.add_compliance_rule(ComplianceRule(
            rule_id="hipaa_audit_controls",
            framework=ComplianceFramework.HIPAA,
            rule_type=ViolationType.AUDIT_TRAIL,
            condition=lambda data: self._check_audit_controls(data),
            threshold=1.0,
            description="HIPAA Security Rule - Audit Controls (164.312(b))",
            severity=ViolationSeverity.HIGH
        ))
    
    def _add_faa_rules(self):
        """Add FAA-specific compliance rules."""
        # Operational safety rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="faa_operational_safety",
            framework=ComplianceFramework.FAA,
            rule_type=ViolationType.OPERATIONAL_SAFETY,
            condition=lambda data: self._check_operational_safety(data),
            threshold=0.95,  # 95% safety compliance
            description="FAA Part 107 - Small Unmanned Aircraft Systems",
            severity=ViolationSeverity.CRITICAL
        ))
        
        # Flight data retention
        self.add_compliance_rule(ComplianceRule(
            rule_id="faa_flight_data_retention",
            framework=ComplianceFramework.FAA,
            rule_type=ViolationType.DATA_RETENTION,
            condition=lambda data: self._check_flight_data_retention(data),
            threshold=90.0,  # days
            description="FAA flight data recording requirements",
            severity=ViolationSeverity.MEDIUM
        ))
    
    def _add_fda_rules(self):
        """Add FDA-specific compliance rules."""
        # Quality system rule
        self.add_compliance_rule(ComplianceRule(
            rule_id="fda_quality_system",
            framework=ComplianceFramework.FDA,
            rule_type=ViolationType.AUDIT_TRAIL,
            condition=lambda data: self._check_quality_system(data),
            threshold=1.0,
            description="FDA Quality System Regulation (21 CFR Part 820)",
            severity=ViolationSeverity.HIGH
        ))
    
    def _add_custom_rules(self):
        """Add custom compliance rules."""
        # Risk threshold monitoring
        self.add_compliance_rule(ComplianceRule(
            rule_id="custom_risk_threshold",
            framework=ComplianceFramework.CUSTOM,
            rule_type=ViolationType.RISK_THRESHOLD,
            condition=lambda data: self._check_risk_threshold(data),
            threshold=0.8,  # 80% risk threshold
            description="Custom risk threshold monitoring",
            severity=ViolationSeverity.MEDIUM
        ))
        
        # Bias detection
        self.add_compliance_rule(ComplianceRule(
            rule_id="custom_bias_detection",
            framework=ComplianceFramework.CUSTOM,
            rule_type=ViolationType.BIAS_DETECTION,
            condition=lambda data: self._check_bias_indicators(data),
            threshold=0.2,  # 20% bias threshold
            description="AI bias detection and monitoring",
            severity=ViolationSeverity.HIGH
        ))
    
    def add_compliance_rule(self, rule: ComplianceRule):
        """Add a new compliance rule to the monitoring system."""
        with self._lock:
            self.compliance_rules[rule.rule_id] = rule
    
    def remove_compliance_rule(self, rule_id: str):
        """Remove a compliance rule from monitoring."""
        with self._lock:
            self.compliance_rules.pop(rule_id, None)
    
    def start_monitoring(self):
        """Start the compliance monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.audit_logger.log_audit_event(
            event_type="compliance_monitoring",
            actor="RegulatoryMonitor",
            action="start_monitoring",
            resource="compliance_system",
            outcome="success",
            details={
                "enabled_frameworks": [f.value for f in self.enabled_frameworks],
                "active_rules": len(self.compliance_rules)
            },
            audit_level=AuditLevel.CRITICAL
        )
    
    def stop_monitoring(self):
        """Stop the compliance monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.audit_logger.log_audit_event(
            event_type="compliance_monitoring",
            actor="RegulatoryMonitor",
            action="stop_monitoring",
            resource="compliance_system",
            outcome="success",
            audit_level=AuditLevel.CRITICAL
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs compliance checks."""
        last_periodic_check = 0
        last_daily_check = 0
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Real-time checks
                self._perform_realtime_checks()
                
                # Periodic checks (every minute)
                if current_time - last_periodic_check >= self.monitoring_intervals['periodic']:
                    self._perform_periodic_checks()
                    last_periodic_check = current_time
                
                # Daily checks
                if current_time - last_daily_check >= self.monitoring_intervals['daily']:
                    self._perform_daily_checks()
                    last_daily_check = current_time
                
                # Update stats
                self.monitoring_stats['last_check_time'] = current_time
                self.monitoring_stats['checks_performed'] += 1
                
                # Sleep before next iteration
                time.sleep(self.monitoring_intervals['real_time'])
                
            except Exception as e:
                print(f"Error in compliance monitoring loop: {e}")
                time.sleep(5.0)  # Wait before retrying
    
    def _perform_realtime_checks(self):
        """Perform real-time compliance checks."""
        # Check recent audit entries for violations
        recent_entries = self.audit_logger.get_audit_entries(
            start_time=time.time() - 60,  # last minute
            end_time=time.time()
        )
        
        for entry in recent_entries:
            self._check_entry_compliance(entry)
        
        # Check recent decisions
        recent_decisions = self.decision_tracker.get_decisions_by_timerange(
            start_time=time.time() - 60,
            end_time=time.time()
        )
        
        for decision in recent_decisions:
            self._check_decision_compliance(decision)
    
    def _perform_periodic_checks(self):
        """Perform periodic compliance checks."""
        # Data retention checks
        self._check_data_retention_compliance()
        
        # Access pattern analysis
        self._check_access_patterns()
        
        # Risk threshold monitoring
        self._check_risk_thresholds()
    
    def _perform_daily_checks(self):
        """Perform daily compliance checks."""
        # Comprehensive audit review
        self._perform_comprehensive_audit()
        
        # Generate compliance reports
        self._generate_daily_compliance_report()
        
        # Clean up old data per retention policies
        self._enforce_data_retention()
    
    def _check_entry_compliance(self, entry):
        """Check an audit entry for compliance violations."""
        for rule in self.compliance_rules.values():
            if not rule.enabled:
                continue
            
            try:
                if not rule.condition(entry):
                    self._create_violation(
                        violation_type=rule.rule_type,
                        severity=rule.severity,
                        framework=rule.framework,
                        description=f"Rule violation: {rule.description}",
                        affected_resources=[entry.entry_id],
                        detection_method=f"rule_check_{rule.rule_id}",
                        metadata={"rule_id": rule.rule_id, "entry_data": asdict(entry)}
                    )
            except Exception as e:
                print(f"Error checking rule {rule.rule_id}: {e}")
    
    def _check_decision_compliance(self, decision):
        """Check a decision for compliance violations."""
        # Check decision transparency
        if not self._check_decision_transparency(decision):
            self._create_violation(
                violation_type=ViolationType.TRANSPARENCY,
                severity=ViolationSeverity.MEDIUM,
                framework=ComplianceFramework.CUSTOM,
                description="Decision lacks sufficient transparency",
                affected_resources=[decision.decision_id],
                detection_method="transparency_check"
            )
        
        # Check for bias indicators
        if not self._check_bias_indicators(decision):
            self._create_violation(
                violation_type=ViolationType.BIAS_DETECTION,
                severity=ViolationSeverity.HIGH,
                framework=ComplianceFramework.CUSTOM,
                description="Potential bias detected in decision process",
                affected_resources=[decision.decision_id],
                detection_method="bias_detection"
            )
    
    def _create_violation(self, 
                         violation_type: ViolationType,
                         severity: ViolationSeverity,
                         framework: ComplianceFramework,
                         description: str,
                         affected_resources: List[str],
                         detection_method: str,
                         metadata: Dict[str, Any] = None):
        """Create and register a compliance violation."""
        violation_id = f"violation_{int(time.time())}_{len(self.violations)}"
        
        violation = ComplianceViolation(
            violation_id=violation_id,
            timestamp=time.time(),
            violation_type=violation_type,
            severity=severity,
            framework=framework,
            description=description,
            affected_resources=affected_resources,
            detection_method=detection_method,
            recommended_actions=self._get_recommended_actions(violation_type, framework),
            auto_remediation_possible=self._check_auto_remediation_possible(violation_type),
            metadata=metadata or {}
        )
        
        # Store violation
        with self._lock:
            self.violations.append(violation)
            self.violation_history[framework.value].append(violation)
            self.violation_counts[violation_type] += 1
            self.monitoring_stats['violations_detected'] += 1
        
        # Trigger alerts
        self._trigger_alerts(violation)
        
        # Attempt auto-remediation
        if violation.auto_remediation_possible:
            self._attempt_auto_remediation(violation)
        
        # Log the violation
        self.audit_logger.log_audit_event(
            event_type="compliance_violation",
            actor="RegulatoryMonitor",
            action="detect_violation",
            resource=violation_id,
            outcome="violation_detected",
            details=asdict(violation),
            audit_level=AuditLevel.CRITICAL
        )
    
    def _get_recommended_actions(self, violation_type: ViolationType, 
                                framework: ComplianceFramework) -> List[str]:
        """Get recommended actions for a violation type."""
        actions = {
            ViolationType.DATA_RETENTION: [
                "Review data retention policies",
                "Archive or delete old data",
                "Update retention schedules"
            ],
            ViolationType.ACCESS_CONTROL: [
                "Review access permissions",
                "Implement additional access controls",
                "Conduct access audit"
            ],
            ViolationType.PRIVACY_BREACH: [
                "Immediate containment of data exposure",
                "Notify affected parties",
                "Implement additional privacy controls"
            ],
            ViolationType.TRANSPARENCY: [
                "Enhance decision documentation",
                "Implement explainability features",
                "Provide clear decision rationale"
            ],
            ViolationType.BIAS_DETECTION: [
                "Review decision criteria",
                "Implement bias mitigation",
                "Conduct fairness assessment"
            ]
        }
        
        return actions.get(violation_type, ["Review compliance requirements"])
    
    def _check_auto_remediation_possible(self, violation_type: ViolationType) -> bool:
        """Check if auto-remediation is possible for a violation type."""
        auto_remediable = {
            ViolationType.DATA_RETENTION,
            ViolationType.RISK_THRESHOLD
        }
        return violation_type in auto_remediable
    
    def _attempt_auto_remediation(self, violation: ComplianceViolation):
        """Attempt automatic remediation of a violation."""
        try:
            if violation.violation_type == ViolationType.DATA_RETENTION:
                self._auto_remediate_data_retention(violation)
            elif violation.violation_type == ViolationType.RISK_THRESHOLD:
                self._auto_remediate_risk_threshold(violation)
            
            self.monitoring_stats['auto_remediations'] += 1
            
        except Exception as e:
            print(f"Auto-remediation failed for {violation.violation_id}: {e}")
    
    def _trigger_alerts(self, violation: ComplianceViolation):
        """Trigger alerts for a compliance violation."""
        for handler in self.alert_handlers:
            try:
                handler(violation)
            except Exception as e:
                print(f"Alert handler failed: {e}")
    
    def add_alert_handler(self, handler: Callable[[ComplianceViolation], None]):
        """Add an alert handler for compliance violations."""
        self.alert_handlers.append(handler)
    
    # Compliance check methods
    def _check_data_retention(self, data: Any, framework: ComplianceFramework) -> bool:
        """Check data retention compliance."""
        if hasattr(data, 'timestamp'):
            retention_policy = self.retention_policies.get(framework, {})
            max_retention_days = retention_policy.get('audit_logs', 365)
            age_days = (time.time() - data.timestamp) / 86400
            return age_days <= max_retention_days
        return True
    
    def _check_privacy_protection(self, data: Any) -> bool:
        """Check privacy protection compliance."""
        # Check for sensitive data exposure
        if hasattr(data, 'details') and isinstance(data.details, dict):
            sensitive_keywords = ['personal', 'private', 'confidential', 'sensitive']
            data_str = json.dumps(data.details).lower()
            return not any(keyword in data_str for keyword in sensitive_keywords)
        return True
    
    def _check_decision_transparency(self, data: Any) -> bool:
        """Check decision transparency compliance."""
        if hasattr(data, 'reasoning_chain'):
            # Require at least some reasoning steps
            return len(data.reasoning_chain) > 0
        return True
    
    def _check_internal_controls(self, data: Any) -> bool:
        """Check internal controls compliance."""
        # Verify proper authorization and documentation
        if hasattr(data, 'actor') and hasattr(data, 'action'):
            return data.actor != 'unknown' and data.action != 'unknown'
        return True
    
    def _check_audit_completeness(self, data: Any) -> bool:
        """Check audit trail completeness."""
        required_fields = ['timestamp', 'actor', 'action', 'resource', 'outcome']
        if hasattr(data, '__dict__'):
            return all(hasattr(data, field) for field in required_fields)
        return True
    
    def _check_access_controls(self, data: Any) -> bool:
        """Check access control compliance."""
        # Verify proper access controls are in place
        if hasattr(data, 'event_type') and 'access' in data.event_type:
            return hasattr(data, 'actor') and data.actor != 'anonymous'
        return True
    
    def _check_audit_controls(self, data: Any) -> bool:
        """Check audit controls compliance."""
        # Verify audit controls are functioning
        return hasattr(data, 'entry_hash') and data.entry_hash is not None
    
    def _check_operational_safety(self, data: Any) -> bool:
        """Check operational safety compliance."""
        # Check for safety-related violations
        if hasattr(data, 'risk_score'):
            return data.risk_score < 0.8  # Risk threshold
        return True
    
    def _check_flight_data_retention(self, data: Any) -> bool:
        """Check flight data retention compliance."""
        return self._check_data_retention(data, ComplianceFramework.FAA)
    
    def _check_quality_system(self, data: Any) -> bool:
        """Check quality system compliance."""
        # Verify quality controls are in place
        if hasattr(data, 'event_type') and 'quality' in data.event_type:
            return hasattr(data, 'details') and 'quality_check' in data.details
        return True
    
    def _check_risk_threshold(self, data: Any) -> bool:
        """Check risk threshold compliance."""
        if hasattr(data, 'risk_score'):
            return data.risk_score <= 0.8
        return True
    
    def _check_bias_indicators(self, data: Any) -> bool:
        """Check for bias indicators."""
        # Simple bias detection based on decision patterns
        if hasattr(data, 'reasoning_chain'):
            reasoning_text = ' '.join(data.reasoning_chain).lower()
            bias_keywords = ['always', 'never', 'all', 'none', 'definitely']
            bias_count = sum(1 for keyword in bias_keywords if keyword in reasoning_text)
            return bias_count <= 1  # Allow some absolute language
        return True
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status."""
        recent_violations = [v for v in self.violations if time.time() - v.timestamp <= 86400]
        
        status = {
            "monitoring_active": self.monitoring_active,
            "enabled_frameworks": [f.value for f in self.enabled_frameworks],
            "total_violations": len(self.violations),
            "recent_violations_24h": len(recent_violations),
            "violation_breakdown": dict(self.violation_counts),
            "monitoring_stats": self.monitoring_stats.copy(),
            "compliance_score": self._calculate_compliance_score(),
            "last_check": datetime.fromtimestamp(self.monitoring_stats['last_check_time']).isoformat()
        }
        
        return status
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)."""
        if not self.monitoring_stats['checks_performed']:
            return 100.0
        
        violation_rate = self.monitoring_stats['violations_detected'] / self.monitoring_stats['checks_performed']
        compliance_score = max(0.0, 100.0 - (violation_rate * 100))
        return round(compliance_score, 2)
    
    def generate_compliance_report(self, framework: ComplianceFramework = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        if framework:
            violations = [v for v in self.violations if v.framework == framework]
        else:
            violations = self.violations
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "framework": framework.value if framework else "all",
            "summary": {
                "total_violations": len(violations),
                "by_severity": defaultdict(int),
                "by_type": defaultdict(int)
            },
            "recent_violations": [],
            "compliance_trends": self._analyze_compliance_trends(violations),
            "recommendations": self._generate_compliance_recommendations(violations)
        }
        
        # Aggregate violation statistics
        for violation in violations:
            report["summary"]["by_severity"][violation.severity.value] += 1
            report["summary"]["by_type"][violation.violation_type.value] += 1
        
        # Recent violations (last 7 days)
        recent_cutoff = time.time() - (7 * 86400)
        report["recent_violations"] = [
            asdict(v) for v in violations if v.timestamp >= recent_cutoff
        ]
        
        return report
    
    def _enforce_data_retention(self):
        """Enforce data retention policies by cleaning up old data."""
        try:
            # Get old entries for cleanup
            cutoff_time = time.time() - (365 * 24 * 3600)  # 1 year default
            old_entries = self.audit_logger.get_audit_entries(end_time=cutoff_time)
            
            # In a real implementation, this would actually delete old data
            if old_entries:
                print(f"Data retention: {len(old_entries)} old entries identified for cleanup")
        except Exception as e:
            print(f"Data retention enforcement failed: {e}")
    
    def _perform_comprehensive_audit(self):
        """Perform comprehensive audit review."""
        try:
            # Get all recent audit entries
            entries = self.audit_logger.get_audit_entries(
                start_time=time.time() - 86400  # Last 24 hours
            )
            
            # Analyze for patterns
            risk_events = [e for e in entries if getattr(e, 'risk_score', 0) > 0.7]
            
            if risk_events:
                print(f"Comprehensive audit: {len(risk_events)} high-risk events identified")
        except Exception as e:
            print(f"Comprehensive audit failed: {e}")
    
    def _generate_daily_compliance_report(self):
        """Generate daily compliance report."""
        try:
            for framework in self.enabled_frameworks:
                report = self.generate_compliance_report(framework)
                print(f"Daily report for {framework.value}: {report['summary']['total_violations']} violations")
        except Exception as e:
            print(f"Daily report generation failed: {e}")
    
    def _auto_remediate_data_retention(self, violation):
        """Auto-remediate data retention violations."""
        print(f"Auto-remediating data retention violation: {violation.violation_id}")
        # In a real implementation, this would archive or delete old data
    
    def _auto_remediate_risk_threshold(self, violation):
        """Auto-remediate risk threshold violations."""
        print(f"Auto-remediating risk threshold violation: {violation.violation_id}")
        # In a real implementation, this would adjust system parameters
    
    def _analyze_compliance_trends(self, violations):
        """Analyze compliance trends."""
        if not violations:
            return {"trend": "stable", "change": 0}
        
        # Simple trend analysis
        recent_count = len([v for v in violations if time.time() - v.timestamp <= 86400])
        older_count = len([v for v in violations if 86400 < time.time() - v.timestamp <= 172800])
        
        if older_count > 0:
            change = (recent_count - older_count) / older_count * 100
        else:
            change = 0
        
        return {
            "trend": "increasing" if change > 10 else "decreasing" if change < -10 else "stable",
            "change_percent": change
        }
    
    def _generate_compliance_recommendations(self, violations):
        """Generate compliance recommendations."""
        recommendations = []
        
        severity_counts = defaultdict(int)
        for violation in violations:
            severity_counts[violation.severity] += 1
        
        if severity_counts[ViolationSeverity.CRITICAL] > 0:
            recommendations.append("Immediate review required for critical violations")
        
        if severity_counts[ViolationSeverity.HIGH] > 5:
            recommendations.append("Consider implementing additional preventive controls")
        
        if len(violations) > 20:
            recommendations.append("Review and update compliance policies")
        
        return recommendations
    
    def _check_data_retention_compliance(self):
        """Check data retention compliance."""
        try:
            # Check audit log retention
            old_entries = self.audit_logger.get_audit_entries(
                end_time=time.time() - (365 * 24 * 3600)  # Older than 1 year
            )
            
            if old_entries:
                print(f"Data retention check: {len(old_entries)} entries exceed retention period")
        except Exception as e:
            print(f"Data retention check failed: {e}")
    
    def _check_access_patterns(self):
        """Check access patterns for anomalies."""
        try:
            recent_entries = self.audit_logger.get_audit_entries(
                start_time=time.time() - 3600  # Last hour
            )
            
            # Simple access pattern analysis
            access_counts = defaultdict(int)
            for entry in recent_entries:
                if hasattr(entry, 'actor'):
                    access_counts[entry.actor] += 1
            
            # Check for excessive access
            for actor, count in access_counts.items():
                if count > 50:  # Threshold
                    print(f"Access pattern alert: {actor} has {count} accesses in last hour")
        except Exception as e:
            print(f"Access pattern check failed: {e}")
    
    def _check_risk_thresholds(self):
        """Check risk thresholds."""
        try:
            recent_entries = self.audit_logger.get_audit_entries(
                start_time=time.time() - 3600  # Last hour
            )
            
            high_risk_events = [
                e for e in recent_entries 
                if hasattr(e, 'risk_score') and e.risk_score > 0.8
            ]
            
            if high_risk_events:
                print(f"Risk threshold alert: {len(high_risk_events)} high-risk events detected")
        except Exception as e:
            print(f"Risk threshold check failed: {e}")


# Global regulatory monitor instance
_global_monitor = None

def get_global_monitor() -> RegulatoryMonitor:
    """Get the global regulatory monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RegulatoryMonitor()
    return _global_monitor

def start_compliance_monitoring(frameworks: List[ComplianceFramework] = None):
    """Start global compliance monitoring."""
    monitor = get_global_monitor()
    if frameworks:
        monitor.enabled_frameworks = frameworks
        monitor._initialize_compliance_rules()
    monitor.start_monitoring()

def get_compliance_status() -> Dict[str, Any]:
    """Get current compliance status."""
    return get_global_monitor().get_compliance_status()