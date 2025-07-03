"""
Swarm Intelligence System: Coordinated Multi-Agent Physics
A sophisticated distributed system that demonstrates emergent collective behavior
with clear commercial applications in autonomous systems and distributed optimization.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from concurrent.futures import ThreadPoolExecutor
import json

# Import our existing components
from simulator import ProjectileSimulator
from model import TrebuchetController
from trainer import TrebuchetTrainer

# Import explainable AI components
from explainable_ai.decision_tracker import DecisionTracker, get_global_tracker, decision_tracker_decorator
from explainable_ai.audit_logger import AuditLogger, get_global_audit_logger, audit_log, AuditLevel, ComplianceFramework


@dataclass
class SwarmAgent:
    """Individual agent in the swarm with unique capabilities and state."""
    id: int
    position: Tuple[float, float]  # (x, y) coordinates
    capabilities: Dict[str, float]  # mass_range, angle_precision, etc.
    energy_level: float = 1.0
    specialization: str = "generalist"  # scout, heavy_hitter, precision, coordinator
    local_knowledge: Dict = None
    performance_history: List[float] = None
    
    def __post_init__(self):
        if self.local_knowledge is None:
            self.local_knowledge = {}
        if self.performance_history is None:
            self.performance_history = []


class SwarmCommunicationProtocol:
    """Handles information sharing and coordination between swarm agents."""
    
    def __init__(self, communication_range: float = 100.0):
        self.communication_range = communication_range
        self.message_queue = []
        self.shared_knowledge_base = {}
        
    def can_communicate(self, agent1: SwarmAgent, agent2: SwarmAgent) -> bool:
        """Check if two agents are within communication range."""
        distance = np.sqrt((agent1.position[0] - agent2.position[0])**2 + 
                          (agent1.position[1] - agent2.position[1])**2)
        return distance <= self.communication_range
    
    def broadcast_knowledge(self, sender: SwarmAgent, knowledge: Dict):
        """Share knowledge with nearby agents."""
        message = {
            'sender_id': sender.id,
            'timestamp': time.time(),
            'knowledge': knowledge,
            'message_type': 'knowledge_share'
        }
        self.message_queue.append(message)
    
    def coordinate_action(self, agents: List[SwarmAgent], targets: List[Tuple[float, float]]):
        """Coordinate multi-agent action for complex targets."""
        coordination_plan = {
            'timestamp': time.time(),
            'participants': [agent.id for agent in agents],
            'targets': targets,
            'strategy': self._generate_coordination_strategy(agents, targets)
        }
        return coordination_plan
    
    def _generate_coordination_strategy(self, agents: List[SwarmAgent], targets: List[Tuple[float, float]]) -> Dict:
        """Generate optimal coordination strategy based on agent capabilities."""
        # Assign targets based on agent specialization and capabilities
        assignments = {}
        
        # Priority assignment algorithm
        for i, target in enumerate(targets):
            best_agent = max(agents, key=lambda a: self._calculate_agent_suitability(a, target))
            assignments[target] = best_agent.id
            
        return {
            'target_assignments': assignments,
            'timing_coordination': self._calculate_optimal_timing(agents, targets),
            'formation_strategy': self._determine_formation(agents)
        }
    
    def _calculate_agent_suitability(self, agent: SwarmAgent, target: Tuple[float, float]) -> float:
        """Calculate how suitable an agent is for a specific target."""
        distance_to_target = np.sqrt((agent.position[0] - target[0])**2 + 
                                   (agent.position[1] - target[1])**2)
        
        # Factor in agent capabilities and specialization
        suitability = agent.energy_level / (1 + distance_to_target * 0.01)
        
        if agent.specialization == "precision" and distance_to_target > 200:
            suitability *= 1.5  # Precision agents better for long range
        elif agent.specialization == "heavy_hitter" and distance_to_target < 100:
            suitability *= 1.3  # Heavy hitters better for close range
            
        return suitability
    
    def _calculate_optimal_timing(self, agents: List[SwarmAgent], targets: List[Tuple[float, float]]) -> Dict:
        """Calculate synchronized timing for maximum impact."""
        # Simple synchronization model - can be enhanced with flight time calculations
        base_delay = 0.0
        timing_plan = {}
        
        for agent in agents:
            # Calculate flight time to target (simplified)
            agent_target_dist = 150.0  # Default assumption
            flight_time = agent_target_dist / 50.0  # Rough flight time estimate
            timing_plan[agent.id] = base_delay + flight_time
            
        return timing_plan
    
    def _determine_formation(self, agents: List[SwarmAgent]) -> str:
        """Determine optimal formation based on number of agents and their types."""
        if len(agents) <= 3:
            return "line_formation"
        elif len(agents) <= 8:
            return "wedge_formation"
        else:
            return "distributed_swarm"


class CollectiveLearningEngine:
    """Implements distributed learning across the swarm."""
    
    def __init__(self):
        self.global_knowledge_base = {}
        self.learning_rate = 0.1
        self.experience_buffer = []
        
    def update_collective_knowledge(self, agent_experiences: List[Dict]):
        """Update global knowledge based on individual agent experiences."""
        for experience in agent_experiences:
            self._integrate_experience(experience)
            
    def _integrate_experience(self, experience: Dict):
        """Integrate a single experience into collective knowledge."""
        situation_key = self._generate_situation_key(experience)
        
        if situation_key not in self.global_knowledge_base:
            self.global_knowledge_base[situation_key] = {
                'success_rate': 0.0,
                'optimal_parameters': experience.get('parameters', {}),
                'sample_count': 0
            }
        
        # Update using exponential moving average
        knowledge = self.global_knowledge_base[situation_key]
        knowledge['sample_count'] += 1
        alpha = min(self.learning_rate, 1.0 / knowledge['sample_count'])
        
        knowledge['success_rate'] = (1 - alpha) * knowledge['success_rate'] + alpha * experience['success']
        
        # Update optimal parameters
        for param, value in experience.get('parameters', {}).items():
            if param not in knowledge['optimal_parameters']:
                knowledge['optimal_parameters'][param] = value
            else:
                knowledge['optimal_parameters'][param] = (1 - alpha) * knowledge['optimal_parameters'][param] + alpha * value
    
    def _generate_situation_key(self, experience: Dict) -> str:
        """Generate a key to categorize similar situations."""
        # Discretize continuous values for pattern matching
        target_distance = round(experience.get('target_distance', 0), -1)  # Round to nearest 10
        wind_speed = round(experience.get('wind_speed', 0), 1)  # Round to nearest 0.1
        
        return f"dist_{target_distance}_wind_{wind_speed}"
    
    def query_collective_knowledge(self, situation: Dict) -> Dict:
        """Query the collective knowledge for guidance on a situation."""
        situation_key = self._generate_situation_key(situation)
        
        if situation_key in self.global_knowledge_base:
            return self.global_knowledge_base[situation_key]
        
        # Find similar situations if exact match not found
        similar_situations = self._find_similar_situations(situation_key)
        if similar_situations:
            return similar_situations[0]  # Return most similar
        
        return {'success_rate': 0.5, 'optimal_parameters': {}, 'sample_count': 0}
    
    def _find_similar_situations(self, situation_key: str) -> List[Dict]:
        """Find similar situations in the knowledge base."""
        # Simplified similarity search - can be enhanced with ML
        similar = []
        for key, knowledge in self.global_knowledge_base.items():
            if self._calculate_situation_similarity(situation_key, key) > 0.7:
                similar.append(knowledge)
        
        return sorted(similar, key=lambda x: x['success_rate'], reverse=True)
    
    def _calculate_situation_similarity(self, key1: str, key2: str) -> float:
        """Calculate similarity between two situation keys."""
        # Simple string similarity - can be enhanced
        return 1.0 if key1 == key2 else 0.5


class SwarmIntelligenceSystem:
    """Main swarm intelligence system integrating all components."""
    
    def __init__(self, n_agents: int = 50, field_size: Tuple[float, float] = (1000, 1000)):
        self.n_agents = n_agents
        self.field_size = field_size
        
        # Initialize components
        self.agents = self._initialize_agents()
        self.communication = SwarmCommunicationProtocol()
        self.collective_learning = CollectiveLearningEngine()
        
        # Integration with existing system
        self.base_simulator = ProjectileSimulator()
        self.base_controller = TrebuchetController()
        
        # Performance tracking
        self.mission_history = []
        self.performance_metrics = {
            'success_rate': 0.0,
            'efficiency_score': 0.0,
            'adaptation_speed': 0.0,
            'coordination_quality': 0.0
        }
        
        # Initialize explainability components
        self.decision_tracker = get_global_tracker()
        self.audit_logger = get_global_audit_logger()
        
        # Log system initialization
        audit_log(
            event_type="system_initialization",
            actor="SwarmIntelligenceSystem",
            action="initialize_swarm",
            resource=f"swarm_{n_agents}_agents",
            outcome="success",
            details={
                "n_agents": n_agents,
                "field_size": field_size,
                "initialization_timestamp": time.time()
            },
            audit_level=AuditLevel.BASIC
        )
        
    def _initialize_agents(self) -> List[SwarmAgent]:
        """Initialize swarm agents with diverse capabilities."""
        agents = []
        specializations = ["scout", "heavy_hitter", "precision", "coordinator", "generalist"]
        
        for i in range(self.n_agents):
            # Random positioning
            position = (
                np.random.uniform(0, self.field_size[0]),
                np.random.uniform(0, self.field_size[1])
            )
            
            # Assign specialization (weighted towards generalists)
            specialization = np.random.choice(
                specializations, 
                p=[0.15, 0.15, 0.15, 0.1, 0.45]  # More generalists
            )
            
            # Generate capabilities based on specialization
            capabilities = self._generate_capabilities(specialization)
            
            agent = SwarmAgent(
                id=i,
                position=position,
                capabilities=capabilities,
                specialization=specialization,
                energy_level=np.random.uniform(0.8, 1.0)
            )
            
            agents.append(agent)
        
        return agents
    
    def _generate_capabilities(self, specialization: str) -> Dict[str, float]:
        """Generate agent capabilities based on specialization."""
        base_capabilities = {
            'max_mass': 300.0,
            'angle_precision': 1.0,
            'reload_speed': 1.0,
            'communication_range': 100.0,
            'energy_efficiency': 1.0
        }
        
        if specialization == "scout":
            base_capabilities.update({
                'communication_range': 150.0,
                'energy_efficiency': 1.3,
                'reload_speed': 1.5
            })
        elif specialization == "heavy_hitter":
            base_capabilities.update({
                'max_mass': 500.0,
                'angle_precision': 0.8,
                'energy_efficiency': 0.7
            })
        elif specialization == "precision":
            base_capabilities.update({
                'angle_precision': 2.0,
                'max_mass': 200.0,
                'reload_speed': 0.8
            })
        elif specialization == "coordinator":
            base_capabilities.update({
                'communication_range': 200.0,
                'energy_efficiency': 1.1
            })
        
        return base_capabilities
    
    def execute_swarm_mission(self, targets: List[Tuple[float, float]], 
                            mission_type: str = "coordinated_strike") -> Dict:
        """Execute a coordinated swarm mission."""
        mission_start_time = time.time()
        
        # Phase 1: Mission Planning
        selected_agents = self._select_agents_for_mission(targets, mission_type)
        coordination_plan = self.communication.coordinate_action(selected_agents, targets)
        
        # Phase 2: Collective Knowledge Query
        mission_guidance = self._get_mission_guidance(targets)
        
        # Phase 3: Execution Simulation
        results = self._simulate_mission_execution(selected_agents, targets, coordination_plan)
        
        # Phase 4: Learning and Adaptation
        mission_experience = self._generate_mission_experience(results, targets)
        self.collective_learning.update_collective_knowledge([mission_experience])
        
        # Phase 5: Performance Analysis
        mission_duration = time.time() - mission_start_time
        performance_analysis = self._analyze_mission_performance(results, mission_duration)
        
        mission_report = {
            'mission_id': len(self.mission_history),
            'timestamp': mission_start_time,
            'targets': targets,
            'agents_deployed': len(selected_agents),
            'coordination_plan': coordination_plan,
            'results': results,
            'performance': performance_analysis,
            'duration': mission_duration,
            'lessons_learned': mission_experience
        }
        
        self.mission_history.append(mission_report)
        self._update_global_performance_metrics(performance_analysis)
        
        return mission_report
    
    def _select_agents_for_mission(self, targets: List[Tuple[float, float]], 
                                 mission_type: str) -> List[SwarmAgent]:
        """Select optimal agents for the mission based on targets and requirements."""
        # Track this decision for explainability
        decision_id = self.decision_tracker.track_decision(
            decision_type="agent_selection",
            decision_maker="SwarmIntelligenceSystem._select_agents_for_mission",
            input_params={
                "targets": targets,
                "mission_type": mission_type,
                "available_agents": len(self.agents)
            },
            environmental_factors={
                "field_size": self.field_size,
                "communication_range": self.communication.communication_range
            },
            agent_states={
                "specializations": {spec: len([a for a in self.agents if a.specialization == spec]) 
                                 for spec in ["scout", "heavy_hitter", "precision", "coordinator", "generalist"]},
                "average_energy": np.mean([a.energy_level for a in self.agents])
            }
        )
        
        if mission_type == "coordinated_strike":
            # Select diverse team with good coverage
            selected = []
            
            # Always include a coordinator if available
            coordinators = [a for a in self.agents if a.specialization == "coordinator"]
            if coordinators:
                best_coordinator = max(coordinators, key=lambda a: a.energy_level)
                selected.append(best_coordinator)
                self.decision_tracker.add_reasoning_step(
                    decision_id, 
                    f"Selected coordinator agent {best_coordinator.id} with energy {best_coordinator.energy_level:.2f}"
                )
            
            # Select agents based on target characteristics
            remaining_agents = [a for a in self.agents if a not in selected]
            
            for i, target in enumerate(targets):
                # Find best agent for each target
                suitability_scores = {
                    a.id: self.communication._calculate_agent_suitability(a, target) 
                    for a in remaining_agents
                }
                best_agent = max(remaining_agents, 
                               key=lambda a: suitability_scores[a.id])
                
                if best_agent not in selected:
                    selected.append(best_agent)
                    remaining_agents.remove(best_agent)
                    self.decision_tracker.add_reasoning_step(
                        decision_id,
                        f"Assigned agent {best_agent.id} ({best_agent.specialization}) to target {i} with suitability {suitability_scores[best_agent.id]:.2f}"
                    )
            
            # Add additional agents based on mission complexity
            additional_needed = min(len(targets) * 2, len(remaining_agents))
            selected.extend(remaining_agents[:additional_needed])
            
            if additional_needed > 0:
                self.decision_tracker.add_reasoning_step(
                    decision_id,
                    f"Added {additional_needed} additional agents for mission complexity"
                )
            
            # Complete decision tracking
            self.decision_tracker.complete_decision(
                decision_id=decision_id,
                outcome=f"Selected {len(selected)} agents for mission",
                success=True,
                execution_time_ms=0.0,  # Would be measured in real implementation
                impact_metrics={
                    "agents_selected": len(selected),
                    "coverage_ratio": len(selected) / len(targets),
                    "coordinator_included": any(a.specialization == "coordinator" for a in selected)
                }
            )
            
            # Log audit event
            audit_log(
                event_type="agent_selection",
                actor="SwarmIntelligenceSystem",
                action="select_mission_agents",
                resource=f"mission_{mission_type}",
                outcome="success",
                details={
                    "selected_agents": [a.id for a in selected],
                    "agent_specializations": [a.specialization for a in selected],
                    "targets_count": len(targets),
                    "selection_criteria": mission_type
                },
                audit_level=AuditLevel.COMPLIANCE
            )
            
            return selected
        
        # Default selection with tracking
        default_selected = self.agents[:len(targets) * 2]
        self.decision_tracker.complete_decision(
            decision_id=decision_id,
            outcome=f"Used default selection: {len(default_selected)} agents",
            success=True,
            execution_time_ms=0.0,
            impact_metrics={"agents_selected": len(default_selected)}
        )
        
        return default_selected
    
    def _get_mission_guidance(self, targets: List[Tuple[float, float]]) -> Dict:
        """Get guidance from collective knowledge for mission planning."""
        guidance = {}
        
        for i, target in enumerate(targets):
            situation = {
                'target_distance': np.sqrt(target[0]**2 + target[1]**2),
                'wind_speed': 0.0  # Simplified for now
            }
            
            knowledge = self.collective_learning.query_collective_knowledge(situation)
            guidance[f'target_{i}'] = knowledge
        
        return guidance
    
    def _simulate_mission_execution(self, agents: List[SwarmAgent], 
                                  targets: List[Tuple[float, float]], 
                                  coordination_plan: Dict) -> Dict:
        """Simulate the execution of the coordinated mission."""
        results = {
            'individual_results': [],
            'coordination_effectiveness': 0.0,
            'target_coverage': 0.0,
            'resource_efficiency': 0.0
        }
        
        # Simulate individual agent actions
        for agent in agents:
            # Get target assignment from coordination plan
            assigned_target = self._get_agent_target_assignment(agent, targets, coordination_plan)
            
            if assigned_target:
                # Use existing physics simulation with agent-specific parameters
                agent_result = self._simulate_agent_action(agent, assigned_target)
                results['individual_results'].append(agent_result)
        
        # Calculate coordination metrics
        results['coordination_effectiveness'] = self._calculate_coordination_effectiveness(results['individual_results'])
        results['target_coverage'] = len([r for r in results['individual_results'] if r['success']]) / len(targets)
        results['resource_efficiency'] = self._calculate_resource_efficiency(agents, results['individual_results'])
        
        return results
    
    def _get_agent_target_assignment(self, agent: SwarmAgent, targets: List[Tuple[float, float]], 
                                   coordination_plan: Dict) -> Optional[Tuple[float, float]]:
        """Get the target assigned to a specific agent."""
        assignments = coordination_plan.get('strategy', {}).get('target_assignments', {})
        
        for target, assigned_agent_id in assignments.items():
            if assigned_agent_id == agent.id:
                return target
        
        # Fallback: assign closest target
        if targets:
            return min(targets, key=lambda t: np.sqrt((agent.position[0] - t[0])**2 + (agent.position[1] - t[1])**2))
        
        return None
    
    def _simulate_agent_action(self, agent: SwarmAgent, target: Tuple[float, float]) -> Dict:
        """Simulate an individual agent's action using physics simulation."""
        # Calculate required parameters using existing system
        target_distance = np.sqrt(target[0]**2 + target[1]**2)
        wind_speed = 0.0  # Simplified
        
        # Get prediction from existing controller (if trained)
        try:
            if hasattr(self.base_controller, 'is_fitted') and self.base_controller.is_fitted:
                pred_mass, pred_angle = self.base_controller.predict_single(target_distance, wind_speed)
            else:
                # Use analytical estimation as fallback
                from simulator import calculate_optimal_parameters_analytical
                pred_mass, pred_angle = calculate_optimal_parameters_analytical(target_distance, wind_speed)
        except:
            # Fallback parameters
            pred_mass, pred_angle = 200.0, 45.0
        
        # Modify parameters based on agent capabilities
        actual_mass = min(pred_mass, agent.capabilities['max_mass'])
        actual_angle = pred_angle + np.random.normal(0, 1.0 / agent.capabilities['angle_precision'])
        
        # Simulate shot using existing physics
        self.base_simulator.wind_speed = wind_speed
        achieved_distance = self.base_simulator.calculate_distance_only(actual_mass, actual_angle)
        
        # Calculate success (within 10m tolerance)
        error = abs(achieved_distance - target_distance)
        success = error <= 10.0
        
        return {
            'agent_id': agent.id,
            'target': target,
            'target_distance': target_distance,
            'predicted_params': (pred_mass, pred_angle),
            'actual_params': (actual_mass, actual_angle),
            'achieved_distance': achieved_distance,
            'error': error,
            'success': success,
            'energy_used': actual_mass / agent.capabilities['max_mass']  # Simplified energy model
        }
    
    def _calculate_coordination_effectiveness(self, individual_results: List[Dict]) -> float:
        """Calculate how well the swarm coordinated their actions."""
        if not individual_results:
            return 0.0
        
        # Measure timing synchronization (simplified)
        success_count = sum(1 for r in individual_results if r['success'])
        total_agents = len(individual_results)
        
        base_effectiveness = success_count / total_agents
        
        # Bonus for multiple simultaneous successes (coordination bonus)
        if success_count > 1:
            coordination_bonus = min(0.2, success_count * 0.05)
            base_effectiveness += coordination_bonus
        
        return min(1.0, base_effectiveness)
    
    def _calculate_resource_efficiency(self, agents: List[SwarmAgent], results: List[Dict]) -> float:
        """Calculate how efficiently resources were used."""
        if not results:
            return 0.0
        
        total_energy_used = sum(r.get('energy_used', 0) for r in results)
        total_energy_available = sum(a.energy_level for a in agents)
        
        successful_shots = sum(1 for r in results if r['success'])
        total_shots = len(results)
        
        # Efficiency = (success rate) / (energy utilization)
        energy_utilization = total_energy_used / total_energy_available if total_energy_available > 0 else 1.0
        success_rate = successful_shots / total_shots if total_shots > 0 else 0.0
        
        return success_rate / max(energy_utilization, 0.1)  # Avoid division by zero
    
    def _generate_mission_experience(self, results: Dict, targets: List[Tuple[float, float]]) -> Dict:
        """Generate learning experience from mission results."""
        # Aggregate experience for collective learning
        avg_target_distance = np.mean([np.sqrt(t[0]**2 + t[1]**2) for t in targets])
        overall_success = results['target_coverage'] > 0.7  # 70% success threshold
        
        experience = {
            'target_distance': avg_target_distance,
            'wind_speed': 0.0,  # Simplified
            'success': overall_success,
            'parameters': {
                'coordination_effectiveness': results['coordination_effectiveness'],
                'resource_efficiency': results['resource_efficiency'],
                'agents_deployed': len(results['individual_results'])
            }
        }
        
        return experience
    
    def _analyze_mission_performance(self, results: Dict, duration: float) -> Dict:
        """Analyze overall mission performance."""
        return {
            'success_rate': results['target_coverage'],
            'coordination_score': results['coordination_effectiveness'],
            'efficiency_score': results['resource_efficiency'],
            'response_time': duration,
            'scalability_factor': len(results['individual_results']) / max(1, duration),
            'adaptability_score': 0.8  # Placeholder - would measure adaptation to new situations
        }
    
    def _update_global_performance_metrics(self, performance: Dict):
        """Update global performance metrics based on mission results."""
        alpha = 0.1  # Learning rate for metrics update
        
        for metric, value in performance.items():
            if metric in self.performance_metrics:
                self.performance_metrics[metric] = (1 - alpha) * self.performance_metrics[metric] + alpha * value
    
    def get_swarm_status(self) -> Dict:
        """Get comprehensive status of the swarm system."""
        agent_status = {
            'total_agents': len(self.agents),
            'active_agents': len([a for a in self.agents if a.energy_level > 0.1]),
            'specialization_distribution': self._get_specialization_distribution(),
            'average_energy': np.mean([a.energy_level for a in self.agents]),
            'knowledge_base_size': len(self.collective_learning.global_knowledge_base)
        }
        
        return {
            'agents': agent_status,
            'performance_metrics': self.performance_metrics,
            'mission_count': len(self.mission_history),
            'collective_knowledge_entries': len(self.collective_learning.global_knowledge_base)
        }
    
    def _get_specialization_distribution(self) -> Dict[str, int]:
        """Get distribution of agent specializations."""
        distribution = {}
        for agent in self.agents:
            distribution[agent.specialization] = distribution.get(agent.specialization, 0) + 1
        return distribution
    
    def evolve_swarm(self, generations: int = 10) -> Dict:
        """Evolve the swarm over multiple generations for improved performance."""
        evolution_results = []
        
        for generation in range(generations):
            # Run multiple missions to evaluate current generation
            generation_performance = []
            
            for _ in range(5):  # 5 missions per generation
                # Generate random targets
                targets = [(np.random.uniform(50, 300), np.random.uniform(50, 300)) 
                          for _ in range(np.random.randint(2, 6))]
                
                mission_result = self.execute_swarm_mission(targets)
                generation_performance.append(mission_result['performance']['success_rate'])
            
            avg_performance = np.mean(generation_performance)
            evolution_results.append({
                'generation': generation,
                'average_performance': avg_performance,
                'best_performance': max(generation_performance),
                'swarm_status': self.get_swarm_status()
            })
            
            # Simple evolution: improve low-performing agents
            self._evolve_agent_capabilities(avg_performance)
        
        return {
            'evolution_history': evolution_results,
            'final_performance': evolution_results[-1]['average_performance'],
            'improvement': evolution_results[-1]['average_performance'] - evolution_results[0]['average_performance']
        }
    
    def _evolve_agent_capabilities(self, current_performance: float):
        """Evolve agent capabilities based on performance."""
        # Simple evolution: enhance capabilities of successful agents
        for agent in self.agents:
            if len(agent.performance_history) > 0:
                agent_avg_performance = np.mean(agent.performance_history[-5:])  # Last 5 missions
                
                if agent_avg_performance > current_performance:
                    # Successful agent - slightly enhance capabilities
                    for capability in agent.capabilities:
                        if capability != 'max_mass':  # Don't modify mass limits
                            agent.capabilities[capability] *= 1.02  # 2% improvement
                
                # Reset performance history periodically
                if len(agent.performance_history) > 20:
                    agent.performance_history = agent.performance_history[-10:]


# Commercial Integration Wrapper
class CommercialSwarmSystem:
    """Commercial wrapper that demonstrates business value and ROI."""
    
    def __init__(self, swarm_system: SwarmIntelligenceSystem):
        self.swarm = swarm_system
        self.cost_model = {
            'agent_operational_cost': 0.10,  # $ per agent per mission
            'coordination_overhead': 0.02,   # $ per coordination event
            'energy_cost': 0.05,            # $ per energy unit
            'success_value': 100.0,         # $ value per successful target hit
            'failure_cost': 10.0            # $ cost per failure
        }
        
    def calculate_mission_roi(self, mission_report: Dict) -> Dict:
        """Calculate return on investment for a mission."""
        # Costs
        agent_costs = len(mission_report['agents_deployed']) * self.cost_model['agent_operational_cost']
        energy_costs = sum(r.get('energy_used', 0) for r in mission_report['results']['individual_results']) * self.cost_model['energy_cost']
        coordination_costs = self.cost_model['coordination_overhead']
        
        total_costs = agent_costs + energy_costs + coordination_costs
        
        # Benefits
        successful_hits = sum(1 for r in mission_report['results']['individual_results'] if r['success'])
        failures = len(mission_report['results']['individual_results']) - successful_hits
        
        benefits = successful_hits * self.cost_model['success_value']
        failure_costs = failures * self.cost_model['failure_cost']
        
        net_benefits = benefits - failure_costs
        roi = (net_benefits - total_costs) / total_costs if total_costs > 0 else 0
        
        return {
            'total_costs': total_costs,
            'total_benefits': benefits,
            'failure_costs': failure_costs,
            'net_benefits': net_benefits,
            'roi_percentage': roi * 100,
            'cost_per_success': total_costs / max(1, successful_hits),
            'efficiency_ratio': successful_hits / len(mission_report['agents_deployed'])
        }
    
    def generate_business_case(self) -> Dict:
        """Generate comprehensive business case with projections."""
        # Run sample missions to establish baseline metrics
        sample_missions = []
        for _ in range(10):
            targets = [(np.random.uniform(100, 250), np.random.uniform(100, 250)) 
                      for _ in range(np.random.randint(3, 7))]
            
            mission = self.swarm.execute_swarm_mission(targets)
            roi_analysis = self.calculate_mission_roi(mission)
            sample_missions.append((mission, roi_analysis))
        
        # Calculate averages
        avg_success_rate = np.mean([m[0]['performance']['success_rate'] for m in sample_missions])
        avg_roi = np.mean([m[1]['roi_percentage'] for m in sample_missions])
        avg_efficiency = np.mean([m[1]['efficiency_ratio'] for m in sample_missions])
        
        # Project scaling benefits
        scaling_projections = self._calculate_scaling_projections(avg_success_rate, avg_roi)
        
        business_case = {
            'current_performance': {
                'average_success_rate': avg_success_rate * 100,
                'average_roi': avg_roi,
                'average_efficiency': avg_efficiency,
                'cost_per_success': np.mean([m[1]['cost_per_success'] for m in sample_missions])
            },
            'scaling_projections': scaling_projections,
            'competitive_advantages': [
                'Autonomous coordination reduces human oversight costs',
                'Adaptive learning improves performance over time',
                'Distributed redundancy ensures mission reliability',
                'Scalable architecture supports growth without linear cost increase'
            ],
            'market_applications': [
                'Autonomous logistics and delivery networks',
                'Distributed sensor and monitoring systems',
                'Precision agriculture and targeted interventions',
                'Emergency response and disaster management',
                'Manufacturing quality control and optimization'
            ],
            'risk_mitigation': {
                'redundancy_factor': len(self.swarm.agents) / 10,  # Agents per critical mission
                'adaptation_capability': self.swarm.performance_metrics['adaptability_score'],
                'knowledge_persistence': len(self.swarm.collective_learning.global_knowledge_base)
            }
        }
        
        return business_case
    
    def _calculate_scaling_projections(self, baseline_success_rate: float, baseline_roi: float) -> Dict:
        """Calculate projections for different scale scenarios."""
        scenarios = {
            'current_scale': {
                'agent_count': len(self.swarm.agents),
                'projected_success_rate': baseline_success_rate * 100,
                'projected_roi': baseline_roi,
                'operational_cost_factor': 1.0
            },
            'double_scale': {
                'agent_count': len(self.swarm.agents) * 2,
                'projected_success_rate': min(95, baseline_success_rate * 100 * 1.15),  # 15% improvement
                'projected_roi': baseline_roi * 1.25,  # Coordination benefits
                'operational_cost_factor': 1.8  # Less than linear cost increase
            },
            'enterprise_scale': {
                'agent_count': len(self.swarm.agents) * 10,
                'projected_success_rate': min(98, baseline_success_rate * 100 * 1.35),  # 35% improvement
                'projected_roi': baseline_roi * 1.8,  # Significant coordination benefits
                'operational_cost_factor': 7.5  # Economies of scale
            }
        }
        
        return scenarios


if __name__ == "__main__":
    # Demonstration of the swarm intelligence system
    print("🕸️ Initializing Swarm Intelligence System...")
    
    # Create swarm system
    swarm = SwarmIntelligenceSystem(n_agents=20)  # Smaller for demo
    commercial_system = CommercialSwarmSystem(swarm)
    
    print(f"✅ Swarm initialized with {len(swarm.agents)} agents")
    print(f"Agent specializations: {swarm._get_specialization_distribution()}")
    
    # Run demonstration mission
    print("\n🎯 Executing demonstration mission...")
    targets = [(150, 100), (200, 150), (120, 180)]
    mission_result = swarm.execute_swarm_mission(targets)
    
    print(f"Mission success rate: {mission_result['performance']['success_rate']:.1%}")
    print(f"Coordination effectiveness: {mission_result['performance']['coordination_score']:.1%}")
    print(f"Resource efficiency: {mission_result['performance']['efficiency_score']:.2f}")
    
    # Generate business case
    print("\n💼 Generating business case...")
    business_case = commercial_system.generate_business_case()
    
    print(f"Average success rate: {business_case['current_performance']['average_success_rate']:.1f}%")
    print(f"Average ROI: {business_case['current_performance']['average_roi']:.1f}%")
    print(f"Cost per success: ${business_case['current_performance']['cost_per_success']:.2f}")
    
    print("\n🚀 Swarm Intelligence System demonstration completed!")