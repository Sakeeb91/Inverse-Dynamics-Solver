#!/usr/bin/env python3
"""
Basic functionality test for swarm intelligence system.
"""

def test_swarm_imports():
    """Test that swarm modules can be imported."""
    try:
        # Test basic imports first
        import numpy as np
        import pandas as pd
        print("✅ Core dependencies available")
        
        # Test swarm imports
        from swarm_intelligence import SwarmIntelligenceSystem, CommercialSwarmSystem
        from swarm_visualizer import SwarmVisualizationEngine
        print("✅ Swarm modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_swarm_basic_functionality():
    """Test basic swarm functionality."""
    try:
        from swarm_intelligence import SwarmIntelligenceSystem, CommercialSwarmSystem
        
        # Create small swarm for testing
        swarm = SwarmIntelligenceSystem(n_agents=5)
        commercial = CommercialSwarmSystem(swarm)
        
        print(f"✅ Swarm created with {len(swarm.agents)} agents")
        
        # Test status retrieval
        status = swarm.get_swarm_status()
        print(f"✅ Status retrieved: {status['agents']['total_agents']} total agents")
        
        # Test simple mission
        targets = [(100, 100), (150, 150)]
        mission_result = swarm.execute_swarm_mission(targets)
        print(f"✅ Mission executed with {mission_result['performance']['success_rate']:.1%} success")
        
        # Test ROI calculation
        roi = commercial.calculate_mission_roi(mission_result)
        print(f"✅ ROI calculated: {roi['roi_percentage']:.1f}%")
        
        return True
    except Exception as e:
        print(f"❌ Swarm test failed: {e}")
        return False

def test_visualization_system():
    """Test visualization system."""
    try:
        from swarm_intelligence import SwarmIntelligenceSystem, CommercialSwarmSystem
        from swarm_visualizer import SwarmVisualizationEngine
        
        swarm = SwarmIntelligenceSystem(n_agents=5)
        commercial = CommercialSwarmSystem(swarm)
        visualizer = SwarmVisualizationEngine(swarm, commercial)
        
        # Test basic visualization creation (without saving)
        dashboard = visualizer.create_swarm_overview_dashboard()
        print("✅ Dashboard visualization created")
        
        # Test business case visualization
        business_case = commercial.generate_business_case()
        business_viz = visualizer.create_business_case_visualization(business_case)
        print("✅ Business case visualization created")
        
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

if __name__ == "__main__":
    print("🕸️ Testing Swarm Intelligence System...")
    print()
    
    tests = [
        ("Module Imports", test_swarm_imports),
        ("Basic Functionality", test_swarm_basic_functionality),
        ("Visualization System", test_visualization_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"Testing {name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All swarm intelligence tests passed! System ready for deployment.")
    else:
        print("⚠️ Some tests failed. Check dependencies and implementation.")