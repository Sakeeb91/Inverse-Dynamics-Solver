#!/usr/bin/env python3
"""
Test script for explainable AI infrastructure.
"""

import sys
sys.path.append('.')

from explainable_ai.decision_tracker import DecisionTracker, get_global_tracker
from explainable_ai.audit_logger import AuditLogger, audit_log, AuditLevel, ComplianceFramework
from swarm_intelligence import SwarmIntelligenceSystem

def test_decision_tracking():
    """Test basic decision tracking functionality."""
    print("🧪 Testing Decision Tracking Infrastructure...")
    
    # Test basic decision tracking
    tracker = DecisionTracker()
    
    decision_id = tracker.track_decision(
        decision_type="test_decision",
        decision_maker="test_function",
        input_params={"param1": "value1", "param2": 42},
        environmental_factors={"temperature": 20.5, "pressure": 1013.25}
    )
    
    tracker.add_reasoning_step(decision_id, "Step 1: Analyzed input parameters", 0.8)
    tracker.add_reasoning_step(decision_id, "Step 2: Considered environmental factors", 0.9)
    
    tracker.complete_decision(
        decision_id=decision_id,
        outcome="test_result",
        success=True,
        execution_time_ms=15.5,
        impact_metrics={"accuracy": 0.95, "efficiency": 0.87}
    )
    
    # Test analysis
    patterns = tracker.analyze_decision_patterns()
    print(f"✅ Decision patterns analysis: {patterns['total_decisions']} decisions tracked")
    
    # Test report generation
    report = tracker.generate_decision_report(decision_id)
    print(f"✅ Decision report generated with {len(report['reasoning_process']['reasoning_chain'])} reasoning steps")
    
    return True

def test_audit_logging():
    """Test audit logging functionality."""
    print("🔐 Testing Audit Logging Infrastructure...")
    
    # Test basic audit logging
    audit_logger = AuditLogger()
    
    entry_id = audit_logger.log_audit_event(
        event_type="test_event",
        actor="test_user",
        action="perform_test",
        resource="test_resource",
        outcome="success",
        details={"test_data": "sample_value"},
        audit_level=AuditLevel.COMPLIANCE,
        compliance_frameworks=[ComplianceFramework.GDPR]
    )
    
    print(f"✅ Audit entry created: {entry_id}")
    
    # Test chain verification
    verification = audit_logger.verify_audit_chain()
    print(f"✅ Audit chain verification: {'Valid' if verification['chain_valid'] else 'Invalid'}")
    
    # Test compliance report
    report = audit_logger.generate_compliance_report(ComplianceFramework.GDPR, 1)
    print(f"✅ Compliance report generated: {report['summary']['total_events']} events")
    
    return True

def test_swarm_integration():
    """Test integration with swarm intelligence system."""
    print("🤖 Testing Swarm Integration...")
    
    try:
        # Create swarm system (should initialize with explainability)
        swarm = SwarmIntelligenceSystem(n_agents=10)
        print(f"✅ Swarm system initialized with {len(swarm.agents)} agents")
        
        # Test agent selection with decision tracking
        targets = [(100, 100), (200, 150)]
        selected_agents = swarm._select_agents_for_mission(targets, "coordinated_strike")
        print(f"✅ Agent selection completed: {len(selected_agents)} agents selected")
        
        # Check if decisions were tracked
        tracker = get_global_tracker()
        decisions = tracker.get_decisions_by_type("agent_selection")
        print(f"✅ Decision tracking verified: {len(decisions)} agent selection decisions recorded")
        
        # Display decision details
        if decisions:
            latest_decision = decisions[-1]
            print(f"   - Decision ID: {latest_decision.decision_id}")
            print(f"   - Reasoning steps: {len(latest_decision.reasoning_chain)}")
            print(f"   - Confidence: {latest_decision.confidence_score}")
        
        return True
        
    except Exception as e:
        print(f"❌ Swarm integration test failed: {e}")
        return False

def main():
    """Run all explainable AI tests."""
    print("🎯 EXPLAINABLE AI INFRASTRUCTURE TESTING")
    print("=" * 50)
    
    tests = [
        ("Decision Tracking", test_decision_tracking),
        ("Audit Logging", test_audit_logging),
        ("Swarm Integration", test_swarm_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name}: FAILED - {e}")
        print()
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("📊 TEST SUMMARY")
    print("-" * 20)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! Explainable AI infrastructure is ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)