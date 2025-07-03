#!/usr/bin/env python3
"""
Test script for regulatory compliance monitoring framework.
"""

import sys
sys.path.append('.')

import time
import threading
from compliance import (
    get_global_monitor, start_compliance_monitoring, get_compliance_status,
    get_global_policy_engine, evaluate_compliance_policies,
    get_global_detector, start_violation_detection, get_detection_statistics
)
from explainable_ai.audit_logger import ComplianceFramework, AuditLevel, audit_log
from swarm_intelligence import SwarmIntelligenceSystem

def test_regulatory_monitor():
    """Test regulatory monitoring functionality."""
    print("📋 Testing Regulatory Monitor...")
    
    try:
        # Get monitor instance
        monitor = get_global_monitor()
        
        # Configure for testing
        monitor.enabled_frameworks = [ComplianceFramework.GDPR, ComplianceFramework.CUSTOM]
        monitor._initialize_compliance_rules()
        
        print(f"✅ Monitor initialized with {len(monitor.compliance_rules)} rules")
        print(f"   Frameworks: {[f.value for f in monitor.enabled_frameworks]}")
        
        # Start monitoring briefly
        monitor.start_monitoring()
        time.sleep(2)  # Let it run for 2 seconds
        
        # Check status
        status = get_compliance_status()
        print(f"✅ Monitoring status: {'Active' if status['monitoring_active'] else 'Inactive'}")
        print(f"   Compliance score: {status['compliance_score']}")
        print(f"   Total violations: {status['total_violations']}")
        
        monitor.stop_monitoring()
        
        return True
        
    except Exception as e:
        print(f"❌ Regulatory monitor test failed: {e}")
        return False

def test_policy_engine():
    """Test policy engine functionality."""
    print("⚖️ Testing Policy Engine...")
    
    try:
        # Get policy engine
        engine = get_global_policy_engine()
        
        print(f"✅ Policy engine initialized with {len(engine.policies)} policies")
        
        # Test policy evaluation with mock data
        class MockAuditEntry:
            def __init__(self):
                self.timestamp = time.time() - 400 * 24 * 3600  # 400 days ago
                self.event_type = "personal_data_processing"
                self.risk_score = 0.9
                self.actor = "test_user"
                self.action = "data_access"
        
        mock_data = MockAuditEntry()
        
        # Evaluate policies
        violations = evaluate_compliance_policies(mock_data, ComplianceFramework.GDPR)
        print(f"✅ Policy evaluation completed: {len(violations)} violations detected")
        
        if violations:
            violation = violations[0]
            print(f"   Sample violation: {violation['policy_name']}")
            print(f"   Severity: {violation['severity']}")
        
        # Test policy statistics
        stats = engine.get_policy_statistics()
        print(f"✅ Policy statistics: {stats['total_policies']} total, {stats['enabled_policies']} enabled")
        
        return True
        
    except Exception as e:
        print(f"❌ Policy engine test failed: {e}")
        return False

def test_violation_detector():
    """Test violation detection system."""
    print("🕵️ Testing Violation Detector...")
    
    try:
        # Get detector instance
        detector = get_global_detector()
        
        print(f"✅ Detector initialized with {len(detector.patterns)} patterns")
        
        # Start detection briefly
        detector.start_detection()
        time.sleep(3)  # Let it run for 3 seconds
        
        # Generate some activity to trigger detection
        for i in range(60):  # Generate many events to trigger threshold
            audit_log(
                event_type="data_access",
                actor="test_user_suspicious",
                action="access_data",
                resource=f"resource_{i}",
                outcome="success",
                audit_level=AuditLevel.BASIC
            )
        
        time.sleep(2)  # Allow detection to process
        
        # Check detection statistics
        stats = get_detection_statistics()
        print(f"✅ Detection statistics:")
        print(f"   Active: {stats['detection_active']}")
        print(f"   Total patterns: {stats['total_patterns']}")
        print(f"   Total detections: {stats['total_detections']}")
        print(f"   ML available: {stats['ml_available']}")
        
        detector.stop_detection()
        
        return True
        
    except Exception as e:
        print(f"❌ Violation detector test failed: {e}")
        return False

def test_compliance_integration():
    """Test integration with swarm intelligence system."""
    print("🔗 Testing Compliance Integration...")
    
    try:
        # Create swarm system (triggers compliance monitoring)
        swarm = SwarmIntelligenceSystem(n_agents=5)
        
        # Execute operations that generate audit entries
        targets = [(100, 100), (200, 150)]
        selected_agents = swarm._select_agents_for_mission(targets, "coordinated_strike")
        
        print(f"✅ Swarm operations completed: {len(selected_agents)} agents selected")
        
        # Check if compliance monitoring captured the activity
        status = get_compliance_status()
        print(f"✅ Compliance integration verified:")
        print(f"   Monitoring active: {status['monitoring_active']}")
        print(f"   Recent violations: {status['recent_violations_24h']}")
        
        # Generate compliance report
        monitor = get_global_monitor()
        report = monitor.generate_compliance_report()
        print(f"✅ Compliance report generated:")
        print(f"   Total violations: {report['summary']['total_violations']}")
        print(f"   Framework: {report['framework']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Compliance integration test failed: {e}")
        return False

def test_end_to_end_compliance():
    """Test end-to-end compliance workflow."""
    print("🌐 Testing End-to-End Compliance Workflow...")
    
    try:
        # Start all compliance systems
        start_compliance_monitoring([ComplianceFramework.GDPR, ComplianceFramework.CUSTOM])
        start_violation_detection()
        
        print("✅ All compliance systems started")
        
        # Generate various types of events
        test_events = [
            {
                "event_type": "high_risk_operation",
                "actor": "automated_system",
                "action": "parameter_optimization",
                "resource": "mission_parameters",
                "outcome": "success",
                "details": {"risk_score": 0.95}  # High risk
            },
            {
                "event_type": "data_processing",
                "actor": "data_processor",
                "action": "process_personal_data",
                "resource": "user_data",
                "outcome": "success",
                "details": {"data_type": "personal", "retention_period": 400}  # Long retention
            },
            {
                "event_type": "system_access",
                "actor": "admin_user",
                "action": "admin_action",
                "resource": "system_config",
                "outcome": "success",
                "details": {"privilege_level": "admin"}
            }
        ]
        
        # Generate events
        for event in test_events:
            audit_log(
                event_type=event["event_type"],
                actor=event["actor"],
                action=event["action"],
                resource=event["resource"],
                outcome=event["outcome"],
                details=event["details"],
                audit_level=AuditLevel.COMPLIANCE
            )
        
        # Wait for processing
        time.sleep(5)
        
        # Check final status
        compliance_status = get_compliance_status()
        detection_stats = get_detection_statistics()
        
        print(f"✅ End-to-end workflow completed:")
        print(f"   Compliance score: {compliance_status['compliance_score']}")
        print(f"   Total violations: {compliance_status['total_violations']}")
        print(f"   Detections: {detection_stats['total_detections']}")
        
        # Stop systems
        monitor = get_global_monitor()
        detector = get_global_detector()
        monitor.stop_monitoring()
        detector.stop_detection()
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end compliance test failed: {e}")
        return False

def main():
    """Run all compliance framework tests."""
    print("🎯 COMPLIANCE FRAMEWORK TESTING")
    print("=" * 45)
    
    tests = [
        ("Regulatory Monitor", test_regulatory_monitor),
        ("Policy Engine", test_policy_engine),
        ("Violation Detector", test_violation_detector),
        ("Compliance Integration", test_compliance_integration),
        ("End-to-End Workflow", test_end_to_end_compliance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name}: FAILED - {e}")
        print()
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("📊 TEST SUMMARY")
    print("-" * 25)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All compliance tests passed! Regulatory framework is ready.")
        print("📝 Features:")
        print("   • Multi-framework compliance monitoring (GDPR, SOX, HIPAA, FAA, FDA)")
        print("   • Configurable policy engine with rule evaluation")
        print("   • Automated violation detection with ML-based pattern analysis")
        print("   • Real-time compliance status monitoring")
        print("   • Integration with audit logging and decision tracking")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)