#!/usr/bin/env python3
"""
Simple test script for compliance framework components.
"""

import sys
sys.path.append('.')

import time
from compliance import (
    get_global_monitor, get_compliance_status,
    get_global_policy_engine, evaluate_compliance_policies,
    get_global_detector, get_detection_statistics
)
from explainable_ai.audit_logger import ComplianceFramework, AuditLevel, audit_log

def test_basic_compliance():
    """Test basic compliance framework functionality."""
    print("🧪 Testing Basic Compliance Framework...")
    
    try:
        # Test 1: Monitor initialization
        monitor = get_global_monitor()
        monitor.enabled_frameworks = [ComplianceFramework.CUSTOM]
        monitor._initialize_compliance_rules()
        print(f"✅ Monitor: {len(monitor.compliance_rules)} rules initialized")
        
        # Test 2: Policy engine
        engine = get_global_policy_engine()
        print(f"✅ Policy Engine: {len(engine.policies)} policies loaded")
        
        # Test 3: Violation detector
        detector = get_global_detector()
        print(f"✅ Detector: {len(detector.patterns)} patterns loaded")
        
        # Test 4: Simple policy evaluation
        class MockData:
            def __init__(self):
                self.risk_score = 0.9  # High risk
                self.event_type = "test_event"
                self.actor = "test_user"
        
        violations = evaluate_compliance_policies(MockData())
        print(f"✅ Policy Evaluation: {len(violations)} violations detected")
        
        # Test 5: Compliance status
        status = get_compliance_status()
        print(f"✅ Status: Score {status['compliance_score']}, {status['total_violations']} violations")
        
        # Test 6: Detection statistics
        stats = get_detection_statistics()
        print(f"✅ Detection Stats: {stats['total_patterns']} patterns, ML: {stats['ml_available']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic compliance test failed: {e}")
        return False

def test_audit_integration():
    """Test integration with audit logging."""
    print("🔗 Testing Audit Integration...")
    
    try:
        # Generate test audit entries
        audit_log(
            event_type="test_compliance",
            actor="test_system",
            action="compliance_test",
            resource="test_resource",
            outcome="success",
            details={"risk_score": 0.95},  # High risk
            audit_level=AuditLevel.COMPLIANCE
        )
        
        print("✅ Test audit entry created")
        
        # Check if compliance monitoring captured it
        status = get_compliance_status()
        print(f"✅ Compliance status updated: {status['total_violations']} violations")
        
        return True
        
    except Exception as e:
        print(f"❌ Audit integration test failed: {e}")
        return False

def main():
    """Run simple compliance tests."""
    print("🎯 SIMPLE COMPLIANCE TESTING")
    print("=" * 35)
    
    tests = [
        ("Basic Framework", test_basic_compliance),
        ("Audit Integration", test_audit_integration)
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
    print("-" * 20)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 Compliance framework basic functionality verified!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)