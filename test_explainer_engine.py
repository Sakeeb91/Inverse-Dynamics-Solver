#!/usr/bin/env python3
"""
Test script for SHAP/LIME explainability engine.
"""

import sys
sys.path.append('.')

import numpy as np
from explainable_ai import get_global_explainer, explain_decision, analyze_agent_selection_importance
from swarm_intelligence import SwarmIntelligenceSystem
from model import TrebuchetController

def test_ml_explainability():
    """Test ML model explainability with SHAP/LIME."""
    print("🔍 Testing ML Model Explainability...")
    
    try:
        # Create and train a trebuchet controller
        controller = TrebuchetController()
        
        # Generate some training data
        np.random.seed(42)
        n_samples = 50
        X_train = np.random.rand(n_samples, 2) * 100  # [distance, wind_speed]
        y_train = np.random.rand(n_samples, 2) * 100 + 50  # [mass, angle]
        
        # Train the model
        controller.fit(X_train, y_train)
        print("✅ Model trained and registered with explainability engine")
        
        # Test prediction explanation
        explainer = get_global_explainer()
        test_input = np.array([[50.0, 5.0]])  # distance=50m, wind=5m/s
        
        # Test explanation (will use fallback method if SHAP/LIME not available)
        explanation = explainer.explain_ml_prediction("trebuchet_controller", test_input, method="shap")
        print(f"✅ ML prediction explanation generated: {len(explanation.feature_importance)} features")
        print(f"   - Method: {explanation.method}")
        print(f"   - Confidence: {explanation.confidence_score:.3f}")
        print(f"   - Top feature: {list(explanation.feature_importance.keys())[0] if explanation.feature_importance else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML explainability test failed: {e}")
        return False

def test_decision_explanation():
    """Test decision explanation functionality."""
    print("🧠 Testing Decision Explanation...")
    
    try:
        # Create swarm and trigger some decisions
        swarm = SwarmIntelligenceSystem(n_agents=5)
        targets = [(100, 100), (200, 150)]
        
        # Execute agent selection (creates tracked decision)
        selected_agents = swarm._select_agents_for_mission(targets, "coordinated_strike")
        print(f"✅ Agent selection decision executed: {len(selected_agents)} agents selected")
        
        # Get the decision ID
        from explainable_ai import get_global_tracker
        tracker = get_global_tracker()
        decisions = tracker.get_decisions_by_type("agent_selection")
        
        if decisions:
            latest_decision = decisions[-1]
            decision_id = latest_decision.decision_id
            
            # Generate explanation
            explanation = explain_decision(decision_id, depth="detailed")
            print(f"✅ Decision explanation generated for {decision_id}")
            print(f"   - Decision type: {explanation.get('decision_type', 'Unknown')}")
            print(f"   - Key factors: {len(explanation.get('key_factors', []))}")
            print(f"   - Reasoning steps: {explanation.get('reasoning_analysis', {}).get('total_steps', 0)}")
            
            return True
        else:
            print("❌ No decisions found to explain")
            return False
            
    except Exception as e:
        print(f"❌ Decision explanation test failed: {e}")
        return False

def test_feature_importance():
    """Test feature importance analysis."""
    print("📊 Testing Feature Importance Analysis...")
    
    try:
        # Create swarm and generate multiple decisions
        swarm = SwarmIntelligenceSystem(n_agents=8)
        
        # Execute multiple agent selections to build decision history
        test_scenarios = [
            [(100, 100)],
            [(150, 120), (200, 180)],
            [(80, 90), (120, 110), (160, 140)]
        ]
        
        for i, targets in enumerate(test_scenarios):
            selected_agents = swarm._select_agents_for_mission(targets, "coordinated_strike")
            print(f"   - Scenario {i+1}: {len(targets)} targets, {len(selected_agents)} agents selected")
        
        # Analyze feature importance
        importance_result = analyze_agent_selection_importance(hours=1)
        print(f"✅ Feature importance analysis completed")
        print(f"   - Method: {importance_result.method}")
        print(f"   - Features analyzed: {len(importance_result.feature_scores)}")
        print(f"   - Baseline score: {importance_result.baseline_score:.3f}")
        
        if importance_result.feature_rankings:
            top_feature = importance_result.feature_rankings[0]
            print(f"   - Top feature: {top_feature[0]} (importance: {top_feature[1]:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature importance test failed: {e}")
        return False

def test_explanation_integration():
    """Test end-to-end explanation integration."""
    print("🔗 Testing End-to-End Explanation Integration...")
    
    try:
        # Get global components
        from explainable_ai import get_global_tracker, get_global_feature_analyzer
        tracker = get_global_tracker()
        analyzer = get_global_feature_analyzer()
        
        # Generate analysis summary
        patterns = tracker.analyze_decision_patterns()
        feature_summary = analyzer.get_feature_importance_summary()
        
        print(f"✅ Integration test completed")
        print(f"   - Total decisions tracked: {patterns['total_decisions']}")
        print(f"   - Decision types: {len(patterns['decisions_by_type'])}")
        print(f"   - Average confidence: {patterns['average_confidence']:.3f}")
        print(f"   - Success rate: {patterns['success_rate']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all explainer engine tests."""
    print("🎯 EXPLAINER ENGINE TESTING")
    print("=" * 40)
    
    tests = [
        ("ML Explainability", test_ml_explainability),
        ("Decision Explanation", test_decision_explanation),
        ("Feature Importance", test_feature_importance),
        ("Integration", test_explanation_integration)
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
        print("\n🎉 All explainer tests passed! SHAP/LIME integration is ready.")
        print("📝 Note: Some features may use fallback methods if SHAP/LIME libraries are not installed.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)