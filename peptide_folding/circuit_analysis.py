"""
Módulo para análisis de circuitos y métricas de calidad
"""
from typing import Dict, List, Any

def analyze_circuit(circuit_transpiled):
    """Analiza el circuito transpilado y retorna métricas"""
    circuit_metrics = {
        'depth': circuit_transpiled.depth(),
        'width': circuit_transpiled.width(),
        'size': circuit_transpiled.size(),
        'gate_counts': circuit_transpiled.count_ops(),
        'nonlocal_gates': sum(1 for inst in circuit_transpiled.data if len(inst.qubits) > 1),
        'measurement_counts': sum(1 for inst in circuit_transpiled.data if inst.operation.name == 'measure'),
        'two_qubit_gate_depth': _calculate_two_qubit_depth(circuit_transpiled)
    }
    return circuit_metrics

def _calculate_two_qubit_depth(circuit):
    """Calcula la profundidad de puertas de dos qubits"""
    two_qubit_gates = [inst for inst in circuit.data if len(inst.qubits) > 1]
    return len(two_qubit_gates)

def _calculate_qubit_distance(q1: int, q2: int, connectivity: Dict[int, List[int]]) -> float:
    """Calcula la distancia mínima entre dos qubits en el grafo de conectividad"""
    if q2 in connectivity[q1]:
        return 1
    visited = {q1}
    queue = [(q1, 0)]
    while queue:
        current, dist = queue.pop(0)
        for neighbor in connectivity[current]:
            if neighbor == q2:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf')

def _estimate_success_probability(circuit, qubit_metrics) -> Dict[str, float]:
    """Estima la probabilidad de éxito del circuito"""
    depth = circuit.depth()
    total_gates = len(circuit.data)
    n_qubits = circuit.num_qubits
    
    # Calcular probabilidades basadas en métricas reales
    gate_success = 1.0
    readout_success = 1.0
    for i in range(n_qubits):
        gate_success *= (1 - qubit_metrics[i]['quality_metrics']['iro']) ** total_gates
        readout_success *= (1 - qubit_metrics[i]['quality_metrics']['readout_error']/10000)
    
    total_success = gate_success * readout_success
    
    return {
        'total_success_probability': total_success,
        'gate_success_probability': gate_success,
        'readout_success_probability': readout_success
    }

def _calculate_error_budget(circuit, qubit_metrics) -> Dict[str, float]:
    """Calcula el presupuesto de error para diferentes componentes"""
    single_qubit_error = 0.0
    two_qubit_error = 0.0
    measurement_error = 0.0
    n_qubits = circuit.num_qubits
    
    # Simplificar el cálculo de errores
    for i in range(n_qubits):
        single_qubit_error += qubit_metrics[i]['quality_metrics']['iro']
        measurement_error += qubit_metrics[i]['quality_metrics']['readout_error']/10000
    
    # Para puertas de dos qubits, asumimos el peor caso
    two_qubit_error = max(qubit_metrics[i]['quality_metrics']['iro'] for i in range(n_qubits)) * circuit.depth()
    
    total_error = single_qubit_error + two_qubit_error + measurement_error
    
    return {
        'single_qubit_gates': single_qubit_error,
        'two_qubit_gates': two_qubit_error,
        'measurements': measurement_error,
        'total_error': total_error
    } 