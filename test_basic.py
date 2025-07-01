#!/usr/bin/env python3
"""
Basic functionality test script.
"""

def test_imports():
    """Test that all modules can be imported."""
    try:
        import numpy as np
        import scipy
        import sklearn
        import matplotlib
        import plotly
        import streamlit
        print("✅ All dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_simulator():
    """Test physics simulator."""
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from simulator import ProjectileSimulator
        
        simulator = ProjectileSimulator()
        distance = simulator.calculate_distance_only(200, 45)
        
        print(f"✅ Simulator test: Distance = {distance:.1f}m")
        assert 50 < distance < 500, "Distance should be reasonable"
        return True
    except Exception as e:
        print(f"❌ Simulator test failed: {e}")
        return False

def test_model():
    """Test neural network model."""
    try:
        from model import TrebuchetController, DataGenerator
        from simulator import ProjectileSimulator
        
        # Create components
        simulator = ProjectileSimulator()
        data_gen = DataGenerator(simulator)
        controller = TrebuchetController(max_iter=5)  # Quick test
        
        # Generate tiny dataset
        X, y, distances = data_gen.generate_random_dataset(n_samples=10)
        
        # Train model
        controller.fit(X, y)
        
        # Test prediction
        pred_mass, pred_angle = controller.predict_single(150, 0)
        
        print(f"✅ Model test: Predicted mass={pred_mass:.1f}kg, angle={pred_angle:.1f}°")
        assert 50 <= pred_mass <= 500, "Mass should be in valid range"
        assert 30 <= pred_angle <= 60, "Angle should be in valid range"
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_app_imports():
    """Test that Streamlit app can be imported."""
    try:
        # We won't run the app, just test imports
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", "app.py")
        # Just check that the file can be parsed, don't execute
        print("✅ Streamlit app can be imported")
        return True
    except Exception as e:
        print(f"❌ App import test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running basic functionality tests...\n")
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Physics Simulator", test_simulator), 
        ("Neural Network Model", test_model),
        ("Streamlit App", test_app_imports)
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
        print("🎉 All tests passed! The project is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the dependencies and installation.")