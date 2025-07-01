# 🚀 Differentiable Trebuchet: An Inverse Dynamics Solver

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project demonstrates the power of **Differentiable Programming** by building a Physics-Encoded Neural Network (PeNN) that solves the inverse dynamics problem for a trebuchet. Given a target distance and wind conditions, the system learns to predict optimal trebuchet parameters (counterweight mass and release angle) through differentiable physics simulation.

## 🎯 Problem Statement

A trebuchet is a complex mechanical system where calculating optimal parameters analytically is extremely difficult. This project solves this **inverse problem** by:

1. **The Learner (Neural Network)**: Proposes optimal parameters based on target conditions
2. **The Simulator (Physics Engine)**: Simulates projectile motion using real physics laws
3. **Differentiable Bridge**: Backpropagates gradients through the physics simulation to train the learner

## 🏗️ Architecture

```
Input: [target_distance, wind_speed] 
    ↓
Neural Network Controller
    ↓
Parameters: [counterweight_mass, release_angle]
    ↓
Physics Simulation (ODE Solver)
    ↓
Output: predicted_distance
    ↓
Loss: (predicted_distance - target_distance)²
```

## 📁 Project Structure

```
differentiable_trebuchet/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── app.py                     # Streamlit web application
├── model.py                   # Neural network controller
├── simulator.py               # Physics simulation engine
├── trainer.py                 # Training logic
├── visualizations/            # Generated plots and charts
│   ├── plots/                # Training plots
│   ├── animations/           # Simulation animations
│   └── visualization_log.md  # Visualization documentation
└── tests/                    # Unit tests
```

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Sakeeb91/Inverse-Dynamics-Solver.git
cd Inverse-Dynamics-Solver
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

## 🔬 Key Features

- **Real-time Physics Simulation**: Accurate projectile motion with air resistance and wind
- **Differentiable Programming**: Gradients flow through the physics simulation
- **Interactive Interface**: Streamlit web app for experimentation
- **Visualization Suite**: Comprehensive plotting and animation capabilities
- **Modular Design**: Clean separation of concerns for easy extension

## 🧮 Physics Model

The simulator incorporates:
- Gravitational acceleration (9.81 m/s²)
- Air resistance (quadratic drag model)
- Wind effects (constant horizontal wind speed)
- Realistic trebuchet mechanics (energy conversion from counterweight)

## 📊 Applications

This differentiable programming approach extends to:
- **Robotics Control**: Learning optimal motor commands
- **Drug Discovery**: Molecular optimization through simulation
- **Circuit Design**: Component value optimization
- **Climate Modeling**: Parameter estimation in weather models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Citation

This project implements concepts from:

> Innes, M., Edelman, A., Fischer, K., Rackauckas, C., Saba, E., Shah, V. B., & Tebbutt, W. (2019). 
> *A differentiable programming system to bridge machine learning and scientific computing*. 
> arXiv preprint arXiv:1907.07587.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔧 Technical Details

- **Backend**: Python with NumPy for numerical computations
- **ML Framework**: Scikit-learn for neural network implementation
- **Physics**: Custom ODE solver with Runge-Kutta integration
- **Frontend**: Streamlit for interactive web interface
- **Visualization**: Plotly for interactive charts and animations

---

*Built with ❤️ by [Sakeeb Rahman](https://github.com/Sakeeb91)*