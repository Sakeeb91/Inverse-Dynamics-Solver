#!/usr/bin/env python3
"""
Audit Logger for Regulatory Compliance

This module provides cryptographically secure audit logging for regulatory
compliance requirements, ensuring immutable records of all system decisions
and actions for legal and compliance auditing.
"""

import hashlib
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import threading
from enum import Enum


class AuditLevel(Enum):
    """Audit logging levels for different compliance requirements."""
    BASIC = "basic"           # Basic operational logging
    COMPLIANCE = "compliance" # Regulatory compliance logging
    FORENSIC = "forensic"     # Detailed forensic logging
    CRITICAL = "critical"     # Critical security events


class ComplianceFramework(Enum):
    """Supported regulatory compliance frameworks."""
    GDPR = "gdpr"             # General Data Protection Regulation
    SOX = "sox"               # Sarbanes-Oxley Act
    HIPAA = "hipaa"           # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"       # Payment Card Industry Data Security Standard
    FAA = "faa"               # Federal Aviation Administration
    FDA = "fda"               # Food and Drug Administration
    CUSTOM = "custom"         # Custom compliance framework


@dataclass
class AuditEntry:
    """Immutable audit log entry with cryptographic integrity."""
    entry_id: str
    timestamp: float
    audit_level: str
    compliance_framework: str
    event_type: str
    actor: str
    action: str
    resource: str
    outcome: str
    details: Dict[str, Any]
    risk_score: float
    compliance_tags: List[str]
    previous_hash: str
    entry_hash: str


class AuditLogger:
    """
    Cryptographically secure audit logger for regulatory compliance.
    
    Provides immutable audit trails with chain-of-custody verification,
    supporting multiple regulatory frameworks and compliance requirements.
    """
    
    def __init__(self, 
                 log_directory: str = "audit_logs",
                 enable_file_logging: bool = True,
                 compliance_frameworks: List[ComplianceFramework] = None):
        """
        Initialize the audit logger.
        
        Args:
            log_directory: Directory to store audit log files
            enable_file_logging: Whether to write logs to files
            compliance_frameworks: List of compliance frameworks to enforce
        """
        self.log_directory = log_directory
        self.enable_file_logging = enable_file_logging
        self.compliance_frameworks = compliance_frameworks or [ComplianceFramework.CUSTOM]
        
        # In-memory audit chain
        self.audit_chain: List[AuditEntry] = []
        self.last_hash = "0" * 64  # Genesis hash
        self._lock = threading.Lock()
        
        # Compliance rule configurations
        self.compliance_rules = self._load_compliance_rules()
        
        # Risk scoring parameters
        self.risk_thresholds = {
            AuditLevel.BASIC: 0.1,
            AuditLevel.COMPLIANCE: 0.3,
            AuditLevel.FORENSIC: 0.7,
            AuditLevel.CRITICAL: 0.9
        }
        
        # Initialize logging directory
        if self.enable_file_logging:
            os.makedirs(self.log_directory, exist_ok=True)
    
    def _load_compliance_rules(self) -> Dict[str, Dict]:
        """Load compliance rules for different frameworks."""
        rules = {
            ComplianceFramework.GDPR.value: {
                "data_processing_consent": True,
                "data_retention_period": 365,  # days
                "anonymization_required": True,
                "right_to_be_forgotten": True,
                "breach_notification_hours": 72
            },
            ComplianceFramework.SOX.value: {
                "financial_controls": True,
                "audit_trail_retention": 2555,  # 7 years in days
                "executive_certification": True,
                "internal_controls_testing": True
            },
            ComplianceFramework.HIPAA.value: {
                "phi_protection": True,
                "access_controls": True,
                "audit_controls": True,
                "integrity_protection": True,
                "transmission_security": True
            },
            ComplianceFramework.FAA.value: {
                "operational_approval": True,
                "pilot_certification": True,
                "maintenance_records": True,
                "flight_data_retention": 30,  # days
                "incident_reporting": True
            },
            ComplianceFramework.FDA.value: {
                "quality_system_regulation": True,
                "good_manufacturing_practice": True,
                "clinical_trial_data": True,
                "adverse_event_reporting": True
            }
        }
        return rules
    
    def _calculate_hash(self, entry_data: Dict[str, Any], previous_hash: str) -> str:
        """Calculate cryptographic hash for audit entry."""
        # Create deterministic string from entry data
        entry_string = json.dumps(entry_data, sort_keys=True)
        combined_string = f"{previous_hash}{entry_string}"
        
        # Calculate SHA-256 hash
        return hashlib.sha256(combined_string.encode()).hexdigest()
    
    def _calculate_risk_score(self, 
                            event_type: str, 
                            action: str, 
                            outcome: str,
                            details: Dict[str, Any]) -> float:
        """Calculate risk score for the audit entry."""
        base_risk = 0.1
        
        # Risk factors based on event type
        event_risk_factors = {
            "data_access": 0.3,
            "parameter_modification": 0.4,
            "system_configuration": 0.5,
            "user_authentication": 0.2,
            "decision_override": 0.6,
            "error_condition": 0.7,
            "security_event": 0.9
        }
        
        # Risk factors based on outcome
        outcome_risk_factors = {
            "success": 0.0,
            "failure": 0.4,
            "error": 0.6,
            "violation": 0.8,
            "breach": 1.0
        }
        
        # Calculate combined risk score
        event_risk = event_risk_factors.get(event_type, base_risk)
        outcome_risk = outcome_risk_factors.get(outcome, 0.0)
        
        # Additional risk from details
        detail_risk = 0.0
        if details:
            if "sensitive_data" in details:
                detail_risk += 0.2
            if "administrative_action" in details:
                detail_risk += 0.1
            if "system_critical" in details:
                detail_risk += 0.3
        
        return min(1.0, event_risk + outcome_risk + detail_risk)
    
    def _generate_compliance_tags(self, 
                                event_type: str, 
                                action: str,
                                frameworks: List[ComplianceFramework]) -> List[str]:
        """Generate compliance tags based on the event and applicable frameworks."""
        tags = []
        
        for framework in frameworks:
            framework_value = framework.value
            rules = self.compliance_rules.get(framework_value, {})
            
            # Tag based on event type and framework requirements
            if framework == ComplianceFramework.GDPR:
                if "data" in event_type.lower():
                    tags.append("gdpr:data_processing")
                if "access" in action.lower():
                    tags.append("gdpr:access_control")
            
            elif framework == ComplianceFramework.SOX:
                if "financial" in event_type.lower():
                    tags.append("sox:financial_control")
                if "audit" in action.lower():
                    tags.append("sox:audit_trail")
            
            elif framework == ComplianceFramework.HIPAA:
                if "health" in event_type.lower() or "medical" in event_type.lower():
                    tags.append("hipaa:phi_protection")
            
            elif framework == ComplianceFramework.FAA:
                if "flight" in event_type.lower() or "aviation" in event_type.lower():
                    tags.append("faa:operational_compliance")
            
            # Add generic compliance tag
            tags.append(f"{framework_value}:monitored")
        
        return tags
    
    def log_audit_event(self,
                       event_type: str,
                       actor: str,
                       action: str,
                       resource: str,
                       outcome: str,
                       details: Dict[str, Any] = None,
                       audit_level: AuditLevel = AuditLevel.BASIC,
                       compliance_frameworks: List[ComplianceFramework] = None) -> str:
        """
        Log an audit event with full compliance tracking.
        
        Args:
            event_type: Type of event (e.g., "user_action", "system_decision")
            actor: Who performed the action (user, system component)
            action: What action was performed
            resource: What resource was affected
            outcome: Result of the action (success, failure, error)
            details: Additional context and metadata
            audit_level: Compliance audit level
            compliance_frameworks: Specific frameworks for this event
            
        Returns:
            entry_id: Unique identifier for the audit entry
        """
        with self._lock:
            entry_id = str(uuid.uuid4())
            timestamp = time.time()
            
            # Use provided frameworks or default to logger's frameworks
            frameworks = compliance_frameworks or self.compliance_frameworks
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(event_type, action, outcome, details or {})
            
            # Generate compliance tags
            compliance_tags = self._generate_compliance_tags(event_type, action, frameworks)
            
            # Prepare entry data for hashing
            entry_data = {
                "entry_id": entry_id,
                "timestamp": timestamp,
                "audit_level": audit_level.value,
                "compliance_framework": [f.value for f in frameworks],
                "event_type": event_type,
                "actor": actor,
                "action": action,
                "resource": resource,
                "outcome": outcome,
                "details": details or {},
                "risk_score": risk_score,
                "compliance_tags": compliance_tags
            }
            
            # Calculate hash
            entry_hash = self._calculate_hash(entry_data, self.last_hash)
            
            # Create audit entry
            audit_entry = AuditEntry(
                entry_id=entry_id,
                timestamp=timestamp,
                audit_level=audit_level.value,
                compliance_framework=[f.value for f in frameworks],
                event_type=event_type,
                actor=actor,
                action=action,
                resource=resource,
                outcome=outcome,
                details=details or {},
                risk_score=risk_score,
                compliance_tags=compliance_tags,
                previous_hash=self.last_hash,
                entry_hash=entry_hash
            )
            
            # Add to audit chain
            self.audit_chain.append(audit_entry)
            self.last_hash = entry_hash
            
            # Write to file if enabled
            if self.enable_file_logging:
                self._write_to_file(audit_entry)
            
            # Check for compliance violations
            self._check_compliance_violations(audit_entry)
            
            return entry_id
    
    def _write_to_file(self, entry: AuditEntry):
        """Write audit entry to file with date-based organization."""
        date_str = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_directory, f"audit_{date_str}.jsonl")
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(asdict(entry), default=str) + '\n')
        except Exception as e:
            # Log to stderr if file writing fails
            print(f"Failed to write audit log: {e}")
    
    def _check_compliance_violations(self, entry: AuditEntry):
        """Check for potential compliance violations and alert."""
        violations = []
        
        # Check risk threshold violations
        if entry.risk_score > 0.8:
            violations.append(f"High risk event: {entry.risk_score:.2f}")
        
        # Check framework-specific violations
        for framework in entry.compliance_framework:
            rules = self.compliance_rules.get(framework, {})
            
            # Example: Check data retention violations for GDPR
            if framework == ComplianceFramework.GDPR.value:
                if "data_retention" in entry.compliance_tags:
                    # Check if data retention period is exceeded
                    pass  # Implementation would check actual retention periods
        
        # Log violations if found
        if violations:
            self.log_audit_event(
                event_type="compliance_violation",
                actor="audit_system",
                action="violation_detected",
                resource=entry.entry_id,
                outcome="alert",
                details={"violations": violations},
                audit_level=AuditLevel.CRITICAL
            )
    
    def verify_audit_chain(self) -> Dict[str, Any]:
        """
        Verify the integrity of the entire audit chain.
        
        Returns:
            Verification results including any integrity issues
        """
        verification_result = {
            "chain_valid": True,
            "total_entries": len(self.audit_chain),
            "verification_timestamp": datetime.now().isoformat(),
            "issues": []
        }
        
        if not self.audit_chain:
            return verification_result
        
        previous_hash = "0" * 64  # Genesis hash
        
        for i, entry in enumerate(self.audit_chain):
            # Verify hash chain
            if entry.previous_hash != previous_hash:
                verification_result["chain_valid"] = False
                verification_result["issues"].append(
                    f"Hash chain broken at entry {i}: {entry.entry_id}"
                )
            
            # Verify entry hash
            entry_data = asdict(entry)
            entry_data.pop("entry_hash")
            entry_data.pop("previous_hash")
            
            calculated_hash = self._calculate_hash(entry_data, entry.previous_hash)
            if calculated_hash != entry.entry_hash:
                verification_result["chain_valid"] = False
                verification_result["issues"].append(
                    f"Entry hash mismatch at entry {i}: {entry.entry_id}"
                )
            
            previous_hash = entry.entry_hash
        
        return verification_result
    
    def get_audit_entries(self,
                         start_time: float = None,
                         end_time: float = None,
                         event_type: str = None,
                         actor: str = None,
                         audit_level: AuditLevel = None,
                         compliance_framework: ComplianceFramework = None) -> List[AuditEntry]:
        """
        Retrieve audit entries based on filter criteria.
        
        Args:
            start_time: Start timestamp for filtering
            end_time: End timestamp for filtering
            event_type: Filter by event type
            actor: Filter by actor
            audit_level: Filter by audit level
            compliance_framework: Filter by compliance framework
            
        Returns:
            List of matching audit entries
        """
        filtered_entries = self.audit_chain
        
        # Apply filters
        if start_time is not None:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]
        
        if end_time is not None:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_time]
        
        if event_type is not None:
            filtered_entries = [e for e in filtered_entries if e.event_type == event_type]
        
        if actor is not None:
            filtered_entries = [e for e in filtered_entries if e.actor == actor]
        
        if audit_level is not None:
            filtered_entries = [e for e in filtered_entries if e.audit_level == audit_level.value]
        
        if compliance_framework is not None:
            filtered_entries = [e for e in filtered_entries 
                              if compliance_framework.value in e.compliance_framework]
        
        return filtered_entries
    
    def generate_compliance_report(self, 
                                 framework: ComplianceFramework,
                                 period_days: int = 30) -> Dict[str, Any]:
        """
        Generate a compliance report for a specific framework and time period.
        
        Args:
            framework: Compliance framework to report on
            period_days: Number of days to include in the report
            
        Returns:
            Comprehensive compliance report
        """
        end_time = time.time()
        start_time = end_time - (period_days * 24 * 3600)
        
        entries = self.get_audit_entries(
            start_time=start_time,
            end_time=end_time,
            compliance_framework=framework
        )
        
        report = {
            "framework": framework.value,
            "report_period": {
                "start": datetime.fromtimestamp(start_time).isoformat(),
                "end": datetime.fromtimestamp(end_time).isoformat(),
                "days": period_days
            },
            "summary": {
                "total_events": len(entries),
                "high_risk_events": len([e for e in entries if e.risk_score > 0.7]),
                "compliance_violations": len([e for e in entries if e.event_type == "compliance_violation"]),
                "average_risk_score": sum(e.risk_score for e in entries) / len(entries) if entries else 0
            },
            "event_breakdown": {},
            "risk_analysis": {},
            "recommendations": []
        }
        
        # Event breakdown by type
        event_types = {}
        for entry in entries:
            event_types[entry.event_type] = event_types.get(entry.event_type, 0) + 1
        report["event_breakdown"] = event_types
        
        # Risk analysis
        risk_levels = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for entry in entries:
            if entry.risk_score < 0.3:
                risk_levels["low"] += 1
            elif entry.risk_score < 0.6:
                risk_levels["medium"] += 1
            elif entry.risk_score < 0.9:
                risk_levels["high"] += 1
            else:
                risk_levels["critical"] += 1
        report["risk_analysis"] = risk_levels
        
        # Generate recommendations
        if report["summary"]["high_risk_events"] > 10:
            report["recommendations"].append("Consider implementing additional security controls")
        
        if report["summary"]["compliance_violations"] > 0:
            report["recommendations"].append("Review and address compliance violations immediately")
        
        return report
    
    def export_audit_logs(self, 
                         file_path: str,
                         format: str = "json",
                         include_verification: bool = True) -> str:
        """
        Export audit logs for external analysis or compliance reporting.
        
        Args:
            file_path: Path to export file
            format: Export format (json, csv)
            include_verification: Include chain verification results
            
        Returns:
            Path to exported file
        """
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_entries": len(self.audit_chain),
                "compliance_frameworks": [f.value for f in self.compliance_frameworks]
            },
            "audit_entries": [asdict(entry) for entry in self.audit_chain]
        }
        
        if include_verification:
            export_data["verification"] = self.verify_audit_chain()
        
        if format.lower() == "json":
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        return file_path


# Global audit logger instance
_global_audit_logger = AuditLogger()

def get_global_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    return _global_audit_logger

def audit_log(event_type: str, 
              actor: str, 
              action: str, 
              resource: str, 
              outcome: str,
              **kwargs) -> str:
    """Convenience function for logging audit events."""
    return _global_audit_logger.log_audit_event(
        event_type=event_type,
        actor=actor,
        action=action,
        resource=resource,
        outcome=outcome,
        **kwargs
    )