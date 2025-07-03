#!/usr/bin/env python3
"""
Decision Tracking System for Swarm Intelligence Explainability

This module provides comprehensive decision tracking and logging capabilities
for the swarm intelligence system, enabling full explainability and audit trails.
"""

import time
import uuid
import json
import hashlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from collections import defaultdict


@dataclass
class DecisionContext:
    """Comprehensive context for a single decision."""
    decision_id: str
    timestamp: float
    decision_type: str
    input_parameters: Dict[str, Any]
    environmental_factors: Dict[str, Any]
    agent_states: Dict[str, Any]
    decision_maker: str
    reasoning_chain: List[str]
    confidence_score: float
    alternatives_considered: List[Dict[str, Any]]
    execution_time_ms: float


@dataclass 
class DecisionOutcome:
    """Result and impact of a decision."""
    decision_id: str
    outcome: Any
    success: bool
    impact_metrics: Dict[str, float]
    downstream_effects: List[str]
    performance_delta: Dict[str, float]
    learned_insights: List[str]


class DecisionTracker:
    """
    Comprehensive decision tracking system for explainable AI.
    
    Captures all decision-making processes in the swarm intelligence system
    with full context, reasoning chains, and outcome tracking.
    """
    
    def __init__(self, enable_detailed_logging: bool = True):
        self.enable_detailed_logging = enable_detailed_logging
        self.decisions: Dict[str, DecisionContext] = {}
        self.outcomes: Dict[str, DecisionOutcome] = {}
        self.decision_chains: Dict[str, List[str]] = defaultdict(list)
        self.performance_impact: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
        
        # Decision type categorization
        self.decision_categories = {
            'agent_selection': 'Agent Selection and Assignment',
            'coordination': 'Swarm Coordination Strategy',
            'parameter_optimization': 'Parameter Optimization',
            'resource_allocation': 'Resource Allocation',
            'mission_planning': 'Mission Planning',
            'learning_adaptation': 'Collective Learning'
        }
    
    def track_decision(self, 
                      decision_type: str,
                      decision_maker: str,
                      input_params: Dict[str, Any],
                      environmental_factors: Dict[str, Any] = None,
                      agent_states: Dict[str, Any] = None) -> str:
        """
        Begin tracking a new decision with full context capture.
        
        Args:
            decision_type: Category of decision being made
            decision_maker: Component/function making the decision
            input_params: Parameters influencing the decision
            environmental_factors: Environmental conditions
            agent_states: Current state of relevant agents
            
        Returns:
            decision_id: Unique identifier for this decision
        """
        decision_id = str(uuid.uuid4())
        
        with self._lock:
            context = DecisionContext(
                decision_id=decision_id,
                timestamp=time.time(),
                decision_type=decision_type,
                input_parameters=input_params.copy() if input_params else {},
                environmental_factors=environmental_factors.copy() if environmental_factors else {},
                agent_states=agent_states.copy() if agent_states else {},
                decision_maker=decision_maker,
                reasoning_chain=[],
                confidence_score=0.0,
                alternatives_considered=[],
                execution_time_ms=0.0
            )
            
            self.decisions[decision_id] = context
            
        return decision_id
    
    def add_reasoning_step(self, decision_id: str, reasoning: str, confidence: float = None):
        """Add a step to the reasoning chain for a decision."""
        with self._lock:
            if decision_id in self.decisions:
                self.decisions[decision_id].reasoning_chain.append(reasoning)
                if confidence is not None:
                    self.decisions[decision_id].confidence_score = confidence
    
    def add_alternative(self, decision_id: str, alternative: Dict[str, Any]):
        """Record an alternative that was considered but not chosen."""
        with self._lock:
            if decision_id in self.decisions:
                self.decisions[decision_id].alternatives_considered.append(alternative)
    
    def complete_decision(self, 
                         decision_id: str, 
                         outcome: Any,
                         success: bool,
                         execution_time_ms: float,
                         impact_metrics: Dict[str, float] = None):
        """
        Complete a decision with its outcome and impact assessment.
        
        Args:
            decision_id: Decision identifier
            outcome: The actual decision result
            success: Whether the decision was successful
            execution_time_ms: Time taken to execute decision
            impact_metrics: Quantified impact of the decision
        """
        with self._lock:
            if decision_id in self.decisions:
                self.decisions[decision_id].execution_time_ms = execution_time_ms
                
                outcome_record = DecisionOutcome(
                    decision_id=decision_id,
                    outcome=outcome,
                    success=success,
                    impact_metrics=impact_metrics or {},
                    downstream_effects=[],
                    performance_delta={},
                    learned_insights=[]
                )
                
                self.outcomes[decision_id] = outcome_record
    
    def link_decisions(self, parent_decision_id: str, child_decision_id: str):
        """Create a causal link between decisions for chain analysis."""
        with self._lock:
            self.decision_chains[parent_decision_id].append(child_decision_id)
    
    def get_decision_context(self, decision_id: str) -> Optional[DecisionContext]:
        """Retrieve full context for a specific decision."""
        return self.decisions.get(decision_id)
    
    def get_decision_outcome(self, decision_id: str) -> Optional[DecisionOutcome]:
        """Retrieve outcome for a specific decision."""
        return self.outcomes.get(decision_id)
    
    def get_decisions_by_type(self, decision_type: str) -> List[DecisionContext]:
        """Get all decisions of a specific type."""
        return [decision for decision in self.decisions.values() 
                if decision.decision_type == decision_type]
    
    def get_decisions_by_timerange(self, start_time: float, end_time: float) -> List[DecisionContext]:
        """Get decisions within a specific time range."""
        return [decision for decision in self.decisions.values()
                if start_time <= decision.timestamp <= end_time]
    
    def get_decision_chain(self, root_decision_id: str) -> List[str]:
        """Get the complete chain of decisions stemming from a root decision."""
        chain = [root_decision_id]
        children = self.decision_chains.get(root_decision_id, [])
        
        for child_id in children:
            chain.extend(self.get_decision_chain(child_id))
        
        return chain
    
    def analyze_decision_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in decision-making for insights and optimization.
        
        Returns:
            Dictionary containing pattern analysis results
        """
        analysis = {
            'total_decisions': len(self.decisions),
            'decisions_by_type': defaultdict(int),
            'average_confidence': 0.0,
            'success_rate': 0.0,
            'average_execution_time': 0.0,
            'decision_frequency': defaultdict(int),
            'reasoning_complexity': defaultdict(list)
        }
        
        if not self.decisions:
            return analysis
        
        total_confidence = 0
        total_successes = 0
        total_execution_time = 0
        
        for decision in self.decisions.values():
            analysis['decisions_by_type'][decision.decision_type] += 1
            total_confidence += decision.confidence_score
            total_execution_time += decision.execution_time_ms
            analysis['reasoning_complexity'][decision.decision_type].append(
                len(decision.reasoning_chain)
            )
            
            # Analyze success rate
            outcome = self.outcomes.get(decision.decision_id)
            if outcome and outcome.success:
                total_successes += 1
        
        analysis['average_confidence'] = total_confidence / len(self.decisions)
        analysis['success_rate'] = total_successes / len(self.decisions)
        analysis['average_execution_time'] = total_execution_time / len(self.decisions)
        
        return analysis
    
    def generate_decision_report(self, decision_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report for a specific decision.
        
        Returns:
            Detailed report including context, reasoning, and impact
        """
        context = self.get_decision_context(decision_id)
        outcome = self.get_decision_outcome(decision_id)
        
        if not context:
            return {"error": "Decision not found"}
        
        report = {
            "decision_summary": {
                "id": decision_id,
                "type": context.decision_type,
                "decision_maker": context.decision_maker,
                "timestamp": datetime.fromtimestamp(context.timestamp).isoformat(),
                "confidence": context.confidence_score,
                "execution_time_ms": context.execution_time_ms
            },
            "input_context": {
                "parameters": context.input_parameters,
                "environmental_factors": context.environmental_factors,
                "agent_states": context.agent_states
            },
            "reasoning_process": {
                "reasoning_chain": context.reasoning_chain,
                "alternatives_considered": context.alternatives_considered,
                "reasoning_steps": len(context.reasoning_chain)
            },
            "outcome_analysis": {},
            "related_decisions": self.decision_chains.get(decision_id, [])
        }
        
        if outcome:
            report["outcome_analysis"] = {
                "result": str(outcome.outcome),
                "success": outcome.success,
                "impact_metrics": outcome.impact_metrics,
                "downstream_effects": outcome.downstream_effects,
                "performance_delta": outcome.performance_delta,
                "learned_insights": outcome.learned_insights
            }
        
        return report
    
    def export_decisions(self, file_path: str = None) -> str:
        """
        Export all decision data to JSON format for external analysis.
        
        Args:
            file_path: Optional file path to save the export
            
        Returns:
            JSON string of all decision data
        """
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_decisions": len(self.decisions),
                "total_outcomes": len(self.outcomes),
                "decision_categories": self.decision_categories
            },
            "decisions": {
                decision_id: asdict(context) 
                for decision_id, context in self.decisions.items()
            },
            "outcomes": {
                decision_id: asdict(outcome)
                for decision_id, outcome in self.outcomes.items()
            },
            "decision_chains": dict(self.decision_chains),
            "analysis": self.analyze_decision_patterns()
        }
        
        json_data = json.dumps(export_data, indent=2, default=str)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_data)
        
        return json_data
    
    def clear_old_decisions(self, older_than_hours: int = 24):
        """Remove decisions older than specified hours to manage memory."""
        cutoff_time = time.time() - (older_than_hours * 3600)
        
        with self._lock:
            old_decision_ids = [
                decision_id for decision_id, context in self.decisions.items()
                if context.timestamp < cutoff_time
            ]
            
            for decision_id in old_decision_ids:
                self.decisions.pop(decision_id, None)
                self.outcomes.pop(decision_id, None)
                self.decision_chains.pop(decision_id, None)


def decision_tracker_decorator(tracker: DecisionTracker, decision_type: str):
    """
    Decorator to automatically track decisions made by functions.
    
    Usage:
        @decision_tracker_decorator(tracker, "agent_selection")
        def select_agents(self, targets, constraints):
            # Function implementation
            return selected_agents
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Extract decision maker information
            decision_maker = f"{func.__module__}.{func.__name__}"
            
            # Prepare input parameters
            input_params = {
                'args': str(args)[:200],  # Limit length for memory
                'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}
            }
            
            # Start tracking
            start_time = time.time()
            decision_id = tracker.track_decision(
                decision_type=decision_type,
                decision_maker=decision_maker,
                input_params=input_params
            )
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                # Complete the decision tracking
                tracker.complete_decision(
                    decision_id=decision_id,
                    outcome=str(result)[:200],  # Limit length
                    success=True,
                    execution_time_ms=execution_time
                )
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                tracker.complete_decision(
                    decision_id=decision_id,
                    outcome=str(e),
                    success=False,
                    execution_time_ms=execution_time
                )
                raise
        
        return wrapper
    return decorator


# Global decision tracker instance
_global_tracker = DecisionTracker()

def get_global_tracker() -> DecisionTracker:
    """Get the global decision tracker instance."""
    return _global_tracker

def reset_global_tracker():
    """Reset the global decision tracker (useful for testing)."""
    global _global_tracker
    _global_tracker = DecisionTracker()