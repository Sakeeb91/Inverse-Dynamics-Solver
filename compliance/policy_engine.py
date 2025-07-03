#!/usr/bin/env python3
"""
Policy Engine for Configurable Compliance Rules

This module provides a flexible policy engine that allows organizations
to define, customize, and enforce compliance policies across different
regulatory frameworks and business requirements.
"""

import json
from typing import Dict, List, Any, Optional, Callable, Union

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import re
import time
from pathlib import Path

from explainable_ai.audit_logger import ComplianceFramework, AuditLevel, audit_log


class PolicyOperator(Enum):
    """Operators for policy conditions."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "ge"
    LESS_EQUAL = "le"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"
    IN = "in"
    NOT_IN = "not_in"


class PolicyActionType(Enum):
    """Types of actions that can be taken when policies are violated."""
    ALERT = "alert"
    BLOCK = "block"
    LOG = "log"
    REMEDIATE = "remediate"
    ESCALATE = "escalate"
    NOTIFY = "notify"


@dataclass
class PolicyCondition:
    """Individual condition within a policy rule."""
    field: str
    operator: PolicyOperator
    value: Any
    case_sensitive: bool = True
    description: str = ""


@dataclass
class PolicyAction:
    """Action to take when a policy is triggered."""
    action_type: PolicyActionType
    parameters: Dict[str, Any]
    enabled: bool = True
    description: str = ""


@dataclass
class PolicyRule:
    """Complete policy rule definition."""
    rule_id: str
    name: str
    description: str
    framework: ComplianceFramework
    category: str
    conditions: List[PolicyCondition]
    actions: List[PolicyAction]
    severity: str = "medium"
    enabled: bool = True
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class PolicyEvaluator(ABC):
    """Abstract base class for policy evaluators."""
    
    @abstractmethod
    def evaluate(self, data: Any, condition: PolicyCondition) -> bool:
        """Evaluate a policy condition against data."""
        pass


class StandardPolicyEvaluator(PolicyEvaluator):
    """Standard policy evaluator for common data types."""
    
    def evaluate(self, data: Any, condition: PolicyCondition) -> bool:
        """Evaluate a policy condition against data."""
        try:
            field_value = self._extract_field_value(data, condition.field)
            return self._apply_operator(field_value, condition.operator, condition.value, condition.case_sensitive)
        except Exception as e:
            print(f"Policy evaluation error: {e}")
            return False
    
    def _extract_field_value(self, data: Any, field_path: str) -> Any:
        """Extract field value using dot notation (e.g., 'user.details.name')."""
        if hasattr(data, '__dict__'):
            obj = data.__dict__
        elif isinstance(data, dict):
            obj = data
        else:
            obj = {"value": data}
        
        # Navigate through nested fields
        for field in field_path.split('.'):
            if isinstance(obj, dict) and field in obj:
                obj = obj[field]
            elif hasattr(obj, field):
                obj = getattr(obj, field)
            else:
                return None
        
        return obj
    
    def _apply_operator(self, field_value: Any, operator: PolicyOperator, 
                       expected_value: Any, case_sensitive: bool) -> bool:
        """Apply the specified operator to compare values."""
        # Handle case sensitivity for strings
        if isinstance(field_value, str) and isinstance(expected_value, str) and not case_sensitive:
            field_value = field_value.lower()
            expected_value = expected_value.lower()
        
        if operator == PolicyOperator.EQUALS:
            return field_value == expected_value
        elif operator == PolicyOperator.NOT_EQUALS:
            return field_value != expected_value
        elif operator == PolicyOperator.GREATER_THAN:
            return field_value > expected_value
        elif operator == PolicyOperator.LESS_THAN:
            return field_value < expected_value
        elif operator == PolicyOperator.GREATER_EQUAL:
            return field_value >= expected_value
        elif operator == PolicyOperator.LESS_EQUAL:
            return field_value <= expected_value
        elif operator == PolicyOperator.CONTAINS:
            return expected_value in str(field_value)
        elif operator == PolicyOperator.NOT_CONTAINS:
            return expected_value not in str(field_value)
        elif operator == PolicyOperator.MATCHES:
            return bool(re.search(expected_value, str(field_value)))
        elif operator == PolicyOperator.IN:
            return field_value in expected_value
        elif operator == PolicyOperator.NOT_IN:
            return field_value not in expected_value
        else:
            return False


class PolicyEngine:
    """
    Configurable policy engine for compliance rule management.
    
    Supports loading policies from files, dynamic rule evaluation,
    and flexible action execution for compliance violations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the policy engine.
        
        Args:
            config_path: Optional path to policy configuration file
        """
        self.policies: Dict[str, PolicyRule] = {}
        self.evaluators: Dict[str, PolicyEvaluator] = {
            "standard": StandardPolicyEvaluator()
        }
        self.action_handlers: Dict[PolicyActionType, Callable] = {}
        self.policy_stats = {
            "evaluations": 0,
            "violations": 0,
            "actions_taken": 0
        }
        
        # Initialize default action handlers
        self._initialize_action_handlers()
        
        # Load configuration if provided
        if config_path:
            self.load_policies_from_file(config_path)
        else:
            self._load_default_policies()
    
    def _initialize_action_handlers(self):
        """Initialize default action handlers."""
        self.action_handlers[PolicyActionType.ALERT] = self._handle_alert_action
        self.action_handlers[PolicyActionType.LOG] = self._handle_log_action
        self.action_handlers[PolicyActionType.BLOCK] = self._handle_block_action
        self.action_handlers[PolicyActionType.REMEDIATE] = self._handle_remediate_action
        self.action_handlers[PolicyActionType.ESCALATE] = self._handle_escalate_action
        self.action_handlers[PolicyActionType.NOTIFY] = self._handle_notify_action
    
    def add_policy(self, policy: PolicyRule):
        """Add a new policy rule to the engine."""
        self.policies[policy.rule_id] = policy
        
        audit_log(
            event_type="policy_management",
            actor="PolicyEngine",
            action="add_policy",
            resource=policy.rule_id,
            outcome="success",
            details={
                "policy_name": policy.name,
                "framework": policy.framework.value,
                "category": policy.category,
                "conditions_count": len(policy.conditions),
                "actions_count": len(policy.actions)
            },
            audit_level=AuditLevel.COMPLIANCE
        )
    
    def remove_policy(self, rule_id: str):
        """Remove a policy rule from the engine."""
        if rule_id in self.policies:
            policy = self.policies.pop(rule_id)
            audit_log(
                event_type="policy_management",
                actor="PolicyEngine",
                action="remove_policy",
                resource=rule_id,
                outcome="success",
                details={"policy_name": policy.name},
                audit_level=AuditLevel.COMPLIANCE
            )
    
    def enable_policy(self, rule_id: str):
        """Enable a policy rule."""
        if rule_id in self.policies:
            self.policies[rule_id].enabled = True
    
    def disable_policy(self, rule_id: str):
        """Disable a policy rule."""
        if rule_id in self.policies:
            self.policies[rule_id].enabled = False
    
    def evaluate_policies(self, data: Any, framework: ComplianceFramework = None) -> List[Dict[str, Any]]:
        """
        Evaluate all applicable policies against provided data.
        
        Args:
            data: Data to evaluate policies against
            framework: Optional framework filter
            
        Returns:
            List of policy violations with recommended actions
        """
        violations = []
        
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            
            if framework and policy.framework != framework:
                continue
            
            self.policy_stats["evaluations"] += 1
            
            if self._evaluate_policy(data, policy):
                violation = self._create_violation(policy, data)
                violations.append(violation)
                self.policy_stats["violations"] += 1
                
                # Execute policy actions
                self._execute_policy_actions(policy, data, violation)
        
        return violations
    
    def _evaluate_policy(self, data: Any, policy: PolicyRule) -> bool:
        """Evaluate all conditions in a policy rule."""
        if not policy.conditions:
            return False
        
        evaluator = self.evaluators["standard"]  # Use standard evaluator for now
        
        # All conditions must be true for the policy to trigger
        for condition in policy.conditions:
            if not evaluator.evaluate(data, condition):
                return False
        
        return True
    
    def _create_violation(self, policy: PolicyRule, data: Any) -> Dict[str, Any]:
        """Create a violation record for a triggered policy."""
        return {
            "violation_id": f"policy_{policy.rule_id}_{int(time.time())}",
            "timestamp": time.time(),
            "policy_id": policy.rule_id,
            "policy_name": policy.name,
            "policy_description": policy.description,
            "framework": policy.framework.value,
            "category": policy.category,
            "severity": policy.severity,
            "data_snapshot": self._create_data_snapshot(data),
            "recommended_actions": [action.description for action in policy.actions if action.enabled],
            "tags": policy.tags
        }
    
    def _create_data_snapshot(self, data: Any) -> Dict[str, Any]:
        """Create a snapshot of relevant data for the violation."""
        if hasattr(data, '__dict__'):
            return {k: str(v)[:200] for k, v in data.__dict__.items()}  # Limit length
        elif isinstance(data, dict):
            return {k: str(v)[:200] for k, v in data.items()}
        else:
            return {"value": str(data)[:200]}
    
    def _execute_policy_actions(self, policy: PolicyRule, data: Any, violation: Dict[str, Any]):
        """Execute all enabled actions for a triggered policy."""
        for action in policy.actions:
            if not action.enabled:
                continue
            
            try:
                handler = self.action_handlers.get(action.action_type)
                if handler:
                    handler(action, data, violation)
                    self.policy_stats["actions_taken"] += 1
                else:
                    print(f"No handler found for action type: {action.action_type}")
            except Exception as e:
                print(f"Error executing action {action.action_type}: {e}")
    
    def _handle_alert_action(self, action: PolicyAction, data: Any, violation: Dict[str, Any]):
        """Handle alert action."""
        message = action.parameters.get("message", "Policy violation detected")
        print(f"🚨 COMPLIANCE ALERT: {message}")
        print(f"   Policy: {violation['policy_name']}")
        print(f"   Severity: {violation['severity']}")
    
    def _handle_log_action(self, action: PolicyAction, data: Any, violation: Dict[str, Any]):
        """Handle log action."""
        audit_log(
            event_type="policy_violation",
            actor="PolicyEngine",
            action="policy_triggered",
            resource=violation["policy_id"],
            outcome="violation",
            details=violation,
            audit_level=AuditLevel.CRITICAL
        )
    
    def _handle_block_action(self, action: PolicyAction, data: Any, violation: Dict[str, Any]):
        """Handle block action."""
        # In a real implementation, this would block the operation
        print(f"🛑 BLOCKED: Operation blocked due to policy violation: {violation['policy_name']}")
    
    def _handle_remediate_action(self, action: PolicyAction, data: Any, violation: Dict[str, Any]):
        """Handle automatic remediation action."""
        remediation_type = action.parameters.get("type", "unknown")
        print(f"🔧 AUTO-REMEDIATION: Attempting {remediation_type} for policy {violation['policy_name']}")
    
    def _handle_escalate_action(self, action: PolicyAction, data: Any, violation: Dict[str, Any]):
        """Handle escalation action."""
        escalation_level = action.parameters.get("level", "manager")
        print(f"📈 ESCALATION: Escalating to {escalation_level} for policy {violation['policy_name']}")
    
    def _handle_notify_action(self, action: PolicyAction, data: Any, violation: Dict[str, Any]):
        """Handle notification action."""
        recipients = action.parameters.get("recipients", ["admin"])
        print(f"📧 NOTIFICATION: Notifying {recipients} about policy violation: {violation['policy_name']}")
    
    def load_policies_from_file(self, file_path: str):
        """Load policies from a configuration file (JSON or YAML)."""
        try:
            path = Path(file_path)
            
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml'] and YAML_AVAILABLE:
                    config = yaml.safe_load(f)
                elif path.suffix.lower() in ['.yaml', '.yml'] and not YAML_AVAILABLE:
                    raise ImportError("YAML support not available. Install with: pip install pyyaml")
                else:
                    config = json.load(f)
            
            self._parse_policy_config(config)
            
            audit_log(
                event_type="policy_management",
                actor="PolicyEngine",
                action="load_policies",
                resource=file_path,
                outcome="success",
                details={"policies_loaded": len(self.policies)},
                audit_level=AuditLevel.COMPLIANCE
            )
            
        except Exception as e:
            print(f"Error loading policies from {file_path}: {e}")
    
    def _parse_policy_config(self, config: Dict[str, Any]):
        """Parse policy configuration and create PolicyRule objects."""
        policies = config.get("policies", [])
        
        for policy_data in policies:
            try:
                # Parse conditions
                conditions = []
                for cond_data in policy_data.get("conditions", []):
                    condition = PolicyCondition(
                        field=cond_data["field"],
                        operator=PolicyOperator(cond_data["operator"]),
                        value=cond_data["value"],
                        case_sensitive=cond_data.get("case_sensitive", True),
                        description=cond_data.get("description", "")
                    )
                    conditions.append(condition)
                
                # Parse actions
                actions = []
                for action_data in policy_data.get("actions", []):
                    action = PolicyAction(
                        action_type=PolicyActionType(action_data["type"]),
                        parameters=action_data.get("parameters", {}),
                        enabled=action_data.get("enabled", True),
                        description=action_data.get("description", "")
                    )
                    actions.append(action)
                
                # Create policy rule
                policy = PolicyRule(
                    rule_id=policy_data["rule_id"],
                    name=policy_data["name"],
                    description=policy_data["description"],
                    framework=ComplianceFramework(policy_data["framework"]),
                    category=policy_data.get("category", "general"),
                    conditions=conditions,
                    actions=actions,
                    severity=policy_data.get("severity", "medium"),
                    enabled=policy_data.get("enabled", True),
                    tags=policy_data.get("tags", []),
                    metadata=policy_data.get("metadata", {})
                )
                
                self.add_policy(policy)
                
            except Exception as e:
                print(f"Error parsing policy {policy_data.get('rule_id', 'unknown')}: {e}")
    
    def _load_default_policies(self):
        """Load default compliance policies."""
        # GDPR Data Protection Policy
        gdpr_policy = PolicyRule(
            rule_id="gdpr_data_retention_default",
            name="GDPR Data Retention Compliance",
            description="Ensure personal data is not retained beyond necessary period",
            framework=ComplianceFramework.GDPR,
            category="data_protection",
            conditions=[
                PolicyCondition(
                    field="event_type",
                    operator=PolicyOperator.CONTAINS,
                    value="personal_data",
                    description="Check for personal data processing events"
                ),
                PolicyCondition(
                    field="timestamp",
                    operator=PolicyOperator.LESS_THAN,
                    value=time.time() - (365 * 24 * 3600),  # 1 year ago
                    description="Check if data is older than retention period"
                )
            ],
            actions=[
                PolicyAction(
                    action_type=PolicyActionType.ALERT,
                    parameters={"message": "Personal data retention period exceeded"},
                    description="Alert on retention violation"
                ),
                PolicyAction(
                    action_type=PolicyActionType.LOG,
                    parameters={},
                    description="Log the violation for audit"
                )
            ],
            severity="high",
            tags=["gdpr", "data_protection", "retention"]
        )
        self.add_policy(gdpr_policy)
        
        # Risk Threshold Policy
        risk_policy = PolicyRule(
            rule_id="risk_threshold_default",
            name="Risk Threshold Monitoring",
            description="Monitor for high-risk operations",
            framework=ComplianceFramework.CUSTOM,
            category="risk_management",
            conditions=[
                PolicyCondition(
                    field="risk_score",
                    operator=PolicyOperator.GREATER_THAN,
                    value=0.8,
                    description="Check for high risk scores"
                )
            ],
            actions=[
                PolicyAction(
                    action_type=PolicyActionType.ALERT,
                    parameters={"message": "High risk operation detected"},
                    description="Alert on high risk"
                ),
                PolicyAction(
                    action_type=PolicyActionType.ESCALATE,
                    parameters={"level": "security_team"},
                    description="Escalate high risk events"
                )
            ],
            severity="critical",
            tags=["risk", "security", "monitoring"]
        )
        self.add_policy(risk_policy)
    
    def export_policies(self, file_path: str, format: str = "json"):
        """Export current policies to a file."""
        try:
            policies_data = {
                "policies": [
                    {
                        "rule_id": policy.rule_id,
                        "name": policy.name,
                        "description": policy.description,
                        "framework": policy.framework.value,
                        "category": policy.category,
                        "conditions": [
                            {
                                "field": cond.field,
                                "operator": cond.operator.value,
                                "value": cond.value,
                                "case_sensitive": cond.case_sensitive,
                                "description": cond.description
                            }
                            for cond in policy.conditions
                        ],
                        "actions": [
                            {
                                "type": action.action_type.value,
                                "parameters": action.parameters,
                                "enabled": action.enabled,
                                "description": action.description
                            }
                            for action in policy.actions
                        ],
                        "severity": policy.severity,
                        "enabled": policy.enabled,
                        "tags": policy.tags,
                        "metadata": policy.metadata
                    }
                    for policy in self.policies.values()
                ]
            }
            
            with open(file_path, 'w') as f:
                if format.lower() == "yaml" and YAML_AVAILABLE:
                    yaml.dump(policies_data, f, default_flow_style=False)
                elif format.lower() == "yaml" and not YAML_AVAILABLE:
                    raise ImportError("YAML support not available. Install with: pip install pyyaml")
                else:
                    json.dump(policies_data, f, indent=2)
            
            audit_log(
                event_type="policy_management",
                actor="PolicyEngine",
                action="export_policies",
                resource=file_path,
                outcome="success",
                details={"policies_exported": len(self.policies), "format": format},
                audit_level=AuditLevel.COMPLIANCE
            )
            
        except Exception as e:
            print(f"Error exporting policies to {file_path}: {e}")
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get policy engine statistics."""
        framework_counts = {}
        category_counts = {}
        
        for policy in self.policies.values():
            framework = policy.framework.value
            category = policy.category
            
            framework_counts[framework] = framework_counts.get(framework, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_policies": len(self.policies),
            "enabled_policies": len([p for p in self.policies.values() if p.enabled]),
            "framework_distribution": framework_counts,
            "category_distribution": category_counts,
            "execution_stats": self.policy_stats.copy()
        }


# Global policy engine instance
_global_policy_engine = None

def get_global_policy_engine() -> PolicyEngine:
    """Get the global policy engine instance."""
    global _global_policy_engine
    if _global_policy_engine is None:
        _global_policy_engine = PolicyEngine()
    return _global_policy_engine

def evaluate_compliance_policies(data: Any, framework: ComplianceFramework = None) -> List[Dict[str, Any]]:
    """Convenience function for evaluating compliance policies."""
    return get_global_policy_engine().evaluate_policies(data, framework)