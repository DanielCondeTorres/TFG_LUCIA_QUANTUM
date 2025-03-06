#!/usr/bin/env python
# coding: utf-8

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import TwoLocal, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from scipy.optimize import minimize
import time
import json
import logging
import argparse
from datetime import datetime
import platform
import psutil
import pandas as pd

from peptide_folding.protein_folding_problem import ProteinFoldingProblem
from peptide_folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from peptide_folding.peptide.peptide import Peptide
from peptide_folding.penalty_parameters import PenaltyParameters
from peptide_folding.peptide_folding_conf_io import *

from qmiotools.integrations.qiskitqmio import QmioBackend
backend = QmioBackend()
backend._logger = logging.getLogger(__name__)

from peptide_folding.circuit_analysis import (
    analyze_circuit, _calculate_qubit_distance,
    _estimate_success_probability, _calculate_error_budget
)
from peptide_folding.qubit_selection import QmioQubitSelector

# Configuración global
N_SHOTS = 100  # Número de shots para cada circuito

def parse_arguments():
    parser = argparse.ArgumentParser(description='Protein Folding Simulation')
    parser.add_argument('--sequence', type=str, default="APRLR",
                      help='Amino acid sequence (default: APRLRFY)')
    parser.add_argument('--axis', type=int, choices=[0, 1, 2, 3], default=2,
                      help='Axis perpendicular to interface (0=x, 1=y, 2=z, 3=t, default: 2)')
    parser.add_argument('--weight', type=float, default=5.0,
                      help='Weight for interface term (default: 5.0)')
    parser.add_argument('--displacement', type=float, default=-0.5,
                      help='Relative displacement from interface (default: 0.5)')
    parser.add_argument('--shots', type=int, default=8000,
                      help='Number of shots per circuit (default: 8000)')
    args = parser.parse_args()
    
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(aa in valid_amino_acids for aa in args.sequence):
        raise ValueError(f"Sequence contains invalid amino acids. Use only: {', '.join(sorted(valid_amino_acids))}")
    
    return args

def setup_protein_folding_problem(sequence, interface_axis=2, interface_weight=5.0, interface_displacement=0.5):
    """
    Configuración del problema usando argumentos
    """
    main_chain = sequence
    side_chains = [""] * len(main_chain)
    
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_terms = PenaltyParameters(
        penalty_chiral=10,
        penalty_back=10,
        penalty_1=10
    )
    
    peptide = Peptide(main_chain, side_chains)
    print(f"\nPeptide created with main chain: {main_chain} and side chains: {side_chains}")
    
    protein_folding_problem = ProteinFoldingProblem(
        peptide=peptide,
        interaction=mj_interaction,
        penalty_parameters=penalty_terms,
        axis=interface_axis,
        weight_interface=interface_weight,
        relative_displacement_from_interface=interface_displacement
    )
    
    return protein_folding_problem

def create_ham_str(args):
    """
    Crear lista de términos para el hamiltoniano del péptido
    """
    protein_folding_problem = setup_protein_folding_problem(
        sequence=args.sequence,
        interface_axis=args.axis,
        interface_weight=args.weight,
        interface_displacement=args.displacement
    )
    
    qubit_op = protein_folding_problem.qubit_op()
    print("\nQubit operator built successfully.")
    print(f"\nNumber of qubits used: {qubit_op.num_qubits}")
    
    ham = []
    for pauli_string in qubit_op.paulis.to_labels():
        ham.append(pauli_string)
        #print(f"Adding Pauli term: {pauli_string}")
    
    return ham, qubit_op, protein_folding_problem

def create_qwc_circuits(qubit_op, ansatz, GROUPS):
    print("\nPreparando circuitos...")
    mesurement_op = []
    n_qubits = qubit_op.num_qubits
    
    for groups in GROUPS:
        op = ['I'] * n_qubits
        for element in groups.to_labels():
            for n,pauli in enumerate(element):
                if pauli !='I':
                    op[n] = pauli
        op = ''.join(op)
        mesurement_op.append(op)
        print(f"Grupo QWC creado: {op}")

    qc = QuantumCircuit(n_qubits)
    qc.compose(ansatz, inplace=True, front=True)
    circuit=[]
    for paulis in mesurement_op:
        qp=QuantumCircuit(qubit_op.num_qubits)
        index=1
        for j in paulis:
            if j=='Y':
                qp.sdg(qubit_op.num_qubits-index)
                qp.h(qubit_op.num_qubits-index)
            if j=='X':
                qp.h(qubit_op.num_qubits-index)
            index+=1
        circuit.append(qp)
    for i in circuit:
        i.compose(qc,inplace=True, front=True)
        i.measure_all()
    
    print(f"Número de grupos QWC: {len(GROUPS)}")
    return circuit

def opa(row, group):
    """
    Calcula la contribución energética de un estado y grupo específico.
    opa--> operator pauli acumulated
    
    Args:
        row (pd.Series): Fila del DataFrame con el estado y su probabilidad
        group (list): Grupo de operadores de Pauli que conmutan
        
    Returns:
        float: Contribución energética del estado para el grupo dado
    """
    suma = 0.0
    state = row["state"]
    prob = row["probability"]
    
    for string in group:
        c = coef[paul.index(string.to_label())]
        st = ''.join('1' if p in 'XYZ' else '0' for p in string.to_label())
        step = (-1)**sum(int(a) & int(b) for a,b in zip(st, state)) * prob * c
        suma = suma + step
    return suma.real

def evaluate_expectation_shots(parameter_values):
    """
    Evalúa el valor esperado del hamiltoniano usando shots.
    """
    n_shots = args.shots
    lista = []
    last_counts = {}
    
    for n, q in enumerate(circuit_transpiled):
        circuit_assembled = q.assign_parameters(parameter_values)
        result = backend.run(circuit_assembled, shots=n_shots, repetition_period=5e-4).result()
        counts = result.get_counts()
        last_counts = counts
        
        pdf = pd.DataFrame.from_dict(counts, orient="index").reset_index()
        pdf.rename(columns={"index":"state", 0: "counts"}, inplace=True)
        pdf["probability"] = pdf["counts"] / n_shots 
        pdf["enery"] = pdf.apply(lambda x: opa(x, GROUPS[n]), axis=1)
        lista.append(sum(pdf["enery"]).real)
    
    # Guardar información detallada
    evaluate_expectation_shots.last_counts = last_counts
    evaluate_expectation_shots.last_probabilities = {
        state: count/n_shots for state, count in last_counts.items()
    }
    evaluate_expectation_shots.circuit_info = {
        'num_circuits': len(circuit_transpiled),
        'shots_per_circuit': n_shots,
        'total_shots': n_shots * len(circuit_transpiled)
    }
    evaluate_expectation_shots.last_energy = sum(lista).real
    
    return sum(lista).real

# Crear una clase simple para emular la estructura que espera interpret_and_plot_results
class OptimizerInfo:
    def __init__(self, params):
        self.settings = params

class VQEResult:
    '''
    Clase para almacenar los resultados del VQE
    '''
    def __init__(self, optimize_result, best_bitstring, selected_qubits):
        self.selected_qubits = selected_qubits
        probabilities = evaluate_expectation_shots.last_probabilities
        self.eigenstate = type('EigenState', (), {
            'binary_probabilities': lambda self: probabilities
        })()
        
        # Actualizar protein_info con los valores exactos del output
        self.protein_info = {
            "sequence": args.sequence,
            "length": len(args.sequence),
            "interface_params": {
                "interface_axis": args.axis,
                "interface_weight": args.weight,
                "interface_displacement": args.displacement
            }
        }
        
        # Resto de la inicialización
        self.optimal_value = optimize_result.fun
        self.optimal_parameters = optimize_result.x
        
        # Measurement results
        self.best_measurement = {
            "state": int(best_bitstring, 2),
            "bitstring": best_bitstring,
            "value": optimize_result.fun,
            "probability": evaluate_expectation_shots.last_probabilities[best_bitstring]
        }
        
        # Crear timestamp al inicio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Obtener best_state
        best_state = max(evaluate_expectation_shots.last_probabilities.items(), key=lambda x: x[1])
        
        # Crear measurement_results
        self.measurement_results = {
            "best_measurement": self.best_measurement,
            "aux_operators_evaluated": {
                "qubit_mapping": self.selected_qubits
            }
        }

        # Atributos básicos
        self.optimal_point = optimize_result.x
        self.optimal_value = optimize_result.fun
        self.optimal_parameters = optimize_result.x
        self.cost_function_evals = optimize_result.nfev

        # Información de entrada
        self.input_parameters = {
            "sequence": args.sequence,
            "interface_axis": args.axis,
            "interface_weight": args.weight,
            "interface_displacement": args.displacement
        }

        # Información de ejecución
        self.execution_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cost_function_evals": optimize_result.nfev,
            "num_qubits": ansatz.num_qubits,
            "execution_time_seconds": time.time() - start_time,
            "hardware_info": {
                "system": platform.system(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "selected_qubits": self.selected_qubits
            }
        }

        # Configuración del VQE
        self.vqe_configuration = {
            "ansatz_params": {
                "num_qubits": ansatz.num_qubits,
                "num_parameters": ansatz.num_parameters,
                "depth": circuit_transpiled[0].depth()
            },
            "optimizer_params": cobyla_params,
            "initial_point": str(list(initial_point))
        }

        # Resultados de la optimización
        self.optimization_results = {
            "optimal_value": {
                "value": optimize_result.fun,
                "description": "Lowest energy found for the protein conformation",
                "units": "arbitrary energy units"
            },
            "circuit_info": {
                "total_qubits": ansatz.num_qubits,
                "total_parameters": ansatz.num_parameters,
                "depth": circuit_transpiled[0].depth(),
                "type": "TwoLocal",
                "description": "TwoLocal ansatz with rx,ry rotations and cx gates",
                "ascii_file": f"circuit_ascii_{timestamp}.txt",
                "image_file": f"circuit_diagram_{timestamp}.png",
                "operations": len(circuit_transpiled[0].data),
                "gate_counts": dict(circuit_transpiled[0].count_ops()),
                "physical_qubits": self.selected_qubits,
                "transpiled_info": {
                    "depth": circuit_transpiled[0].depth(),
                    "size": len(circuit_transpiled[0].data),
                    "initial_layout": self.selected_qubits,
                    #"final_layout": circuit_transpiled[0].layout.final_layout if hasattr(circuit_transpiled[0], 'layout') else None
                }
            },
            "optimal_parameters": self._format_optimal_parameters(optimize_result.x)
        }

    def _format_optimal_parameters(self, parameters):
        """Helper method to format optimal parameters"""
        formatted_params = {}
        for i, param in enumerate(parameters):
            formatted_params[f"angle_{i}"] = {
                "value": param,
                "qubit_logical": i % ansatz.num_qubits,
                "qubit_physical": self.selected_qubits[i % ansatz.num_qubits],
                "layer": i // ansatz.num_qubits,
                "description": f"Ry rotation on logical qubit {i % ansatz.num_qubits} (physical qubit {self.selected_qubits[i % ansatz.num_qubits]}) in layer {i // ansatz.num_qubits}",
                "gate_type": "Ry",
                "radians": param,
                "degrees": param * 180 / np.pi
            }
        return formatted_params

    def _format_binary_vector(self, bitstring):
        """Helper method to format binary vector"""
        return f"{bitstring[0]}_{bitstring[1:]}{'_' * (64 - len(bitstring))}"

    def _format_coordinates(self, coords):
        """Helper method to format coordinates"""
        formatted_coords = []
        for coord in coords:
            formatted_coords.append({
                "amino_acid": coord[0],
                "position": f"[{float(coord[1]):.4f}, {float(coord[2]):.4f}, {float(coord[3]):.4f}]"
            })
        return formatted_coords
if __name__ == "__main__":
    start_time = time.time()
    args = parse_arguments()
    
    # Obtener hamiltoniano y operador cuántico
    ham_strings, qubit_op, protein_folding_problem = create_ham_str(args)
    
    # Crear hamiltoniano con los coeficientes correctos
    hamiltonian = SparsePauliOp(ham_strings, coeffs=qubit_op.coeffs)
    
    # Obtener el número de qubits necesarios del operador cuántico
    n_qubits = qubit_op.num_qubits
    print(f"\nNúmero de qubits necesarios para la secuencia {args.sequence}: {n_qubits}")
    
    # Después de seleccionar los qubits
    qubit_selector = QmioQubitSelector()
    metrics = qubit_selector._calculate_metrics()  # Obtener las métricas para usarlas después
    selected_qubits = qubit_selector.get_best_qubits(n_qubits)
    
    # Crear ansatz
    ansatz = RealAmplitudes(n_qubits, reps=1)
    
    # Crear grupos QWC y circuitos
    GROUPS = hamiltonian.paulis.group_qubit_wise_commuting()
    paul = hamiltonian.paulis.to_labels()
    coef = hamiltonian.coeffs
    circuit = create_qwc_circuits(hamiltonian, ansatz, GROUPS)
    circuit[0].draw(output='mpl', 
                         filename='circuito_completo.png',style='iqx',
                         fold=0,  # No dividir el circuito
                         scale=0.7,  # Reducir un poco el tamaño
                         justify='left')    

    # Transpile con los qubits seleccionados
    circuit_transpiled = transpile(circuit, 
                                 backend,
                                 optimization_level=2, 
                                 initial_layout=selected_qubits)

    print("\nVerificando circuito transpilado:")
    print(f"Qubits físicos usados: {circuit_transpiled[0].qubits}")
    print(f"Profundidad: {circuit_transpiled[0].depth()}")
    print(f"Operaciones: {circuit_transpiled[0].size()}")

    # Imprimir circuito transpilado
    print("\nCircuito transpilado (ASCII):")
    print(circuit_transpiled[0])


    # Guardar imagen del circuito en formato compacto (por bloques)
    circuit_transpiled[0].draw(output='mpl', filename='circuito_transpilado_compacto.png')
    print("\nImagen del circuito (formato compacto) guardada como 'circuito_transpilado_compacto.png'")

    # Guardar imagen del circuito completo (todos los qubits en una vista)
    circuit_transpiled[0].draw(output='mpl', 
                         filename='circuito_transpilado_completo.png',
                         fold=0,  # No dividir el circuito
                         scale=0.7,  # Reducir un poco el tamaño
                         justify='left')  # Alinear a la izquierda
    print("\nImagen del circuito (vista completa) guardada como 'circuito_transpilado_completo.png'")

    # Desglose detallado de operaciones
    print("\nDesglose de operaciones:")
    for instruction in circuit_transpiled[0].data:
        print(f"Operación: {instruction.operation.name}, Qubits: {[q._index for q in instruction.qubits]}")

    # Optimización
    counts = []
    values = []
    
    def store_intermediate_result(xk):
        """Almacena resultados intermedios de la optimización"""
        iteration_count = len(counts) + 1
        # Usar la energía calculada en la última llamada a evaluate_expectation_shots
        energy = evaluate_expectation_shots.last_energy
        counts.append(iteration_count)
        values.append(energy)
        if iteration_count % 1 == 0:  # Mostrar cada iteración
            print(f"Iteración {iteration_count}: Energía = {energy:.6f}")
        return False

    steps = 1
    t0 = time.time()
    for d in range(0,steps):
        start = time.time()
        initial_point = 2*np.pi*np.random.random(ansatz.num_parameters)-np.pi
        
        cobyla_params = {
            'maxiter': 10,      # Número máximo de iteraciones
            'rhobeg': 1.0,      # Tamaño inicial del radio de la región de confianza
            'disp': True,       # Mostrar mensajes de progreso
            'catol': 1e-4       # Tolerancia absoluta en la restricción
        }

        optimizer = OptimizerInfo(cobyla_params)
        raw_result = minimize(
            fun=evaluate_expectation_shots, 
            x0=list(initial_point),
            method='COBYLA',
            options=cobyla_params,
            callback=store_intermediate_result  # Usar nuestra función de callback
        )
        
        # Calculate total time
        end_time = time.time()
        execution_time = end_time - start_time

        # Después de la optimización
        best_state = max(evaluate_expectation_shots.last_probabilities.items(), key=lambda x: x[1])
        best_bitstring = best_state[0]
        best_probability = best_state[1]

        # Crear VQEResult con todos los datos necesarios
        vqe_style_result = VQEResult(
            optimize_result=raw_result,
            best_bitstring=best_bitstring,
            selected_qubits=selected_qubits
        )

        # Interpret and plot results
        result = interpret_and_plot_results(
            vqe_style_result,  # Este objeto ya tiene toda la información necesaria
            protein_folding_problem,
            args,
            qubit_op,
            ansatz,
            optimizer,
            execution_time
        )
        
        result = protein_folding_problem.interpret(raw_result=vqe_style_result)
        
        print("\nCheck if the JSON file was created in the current directory.")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"protein_folding_results_{timestamp}"
            
            plot_convergence(counts, values, f"{base_filename}_convergence.png")
            plot_state_probabilities(evaluate_expectation_shots.last_probabilities, f"{base_filename}_probabilities.png")
            
            # Diccionario de resultados
            results_dict = {
                "input_parameters": vqe_style_result.input_parameters,
                "execution_info": vqe_style_result.execution_info,
                "qubit_info": {
                    "selected_qubits": selected_qubits,
                    "metrics": {
                        str(q): {
                            "basic_metrics": {
                                "IRB": qubit_selector.qubit_properties[q][0],
                                "IRO": qubit_selector.qubit_properties[q][1],
                                "RE": qubit_selector.qubit_properties[q][2]
                            },
                            "connectivity": {
                                "total_connections": len(qubit_selector.connectivity[q]),
                                "connected_to": qubit_selector.connectivity[q]
                            },
                            "quality_metrics": {
                                "irb": metrics[q].irb,
                                "iro": metrics[q].iro,
                                "readout_error": metrics[q].readout_error,
                                "quality_metric": metrics[q].quality_metric,
                                "connectivity_count": metrics[q].connectivity_count,
                                "combined_metric": metrics[q].combined_metric
                            }
                        } for q in selected_qubits
                    },
                    "connectivity_map": {
                        f"{q1}-{q2}": {
                            "connected": q2 in qubit_selector.connectivity[q1],
                            "distance": _calculate_qubit_distance(q1, q2, qubit_selector.connectivity)
                        }
                        for i, q1 in enumerate(selected_qubits)
                        for q2 in selected_qubits[i+1:]
                    }
                },
                "circuit_analysis": {
                    "transpiled_metrics": analyze_circuit(circuit_transpiled[0]),
                    "execution_parameters": {
                        "shots_per_circuit": args.shots,
                        "repetition_period": 5e-4,
                        "optimization_level": 2
                    },
                    "qwc_groups": {
                        "number_of_groups": len(GROUPS),
                        "groups": [str(group) for group in GROUPS],
                        "pauli_terms": paul,
                        "coefficients": [{"real": float(c.real), "imag": float(c.imag)} for c in coef.tolist()]
                    },
                    "measurement_statistics": {
                        "counts": evaluate_expectation_shots.last_counts,
                        "probabilities": evaluate_expectation_shots.last_probabilities,
                        "best_bitstring": best_bitstring,
                        "best_probability": best_probability
                    },
                    "optimization_progress": {
                        "iterations": len(counts),
                        "energy_values": values,
                        "final_energy": values[-1] if values else None,
                        "best_energy": min(values) if values else None,
                        "optimization_trace": {
                            "iterations": counts,
                            "energies": values
                        },
                        "final_parameters": {
                            "angles_radians": raw_result.x.tolist(),
                            "angles_degrees": (raw_result.x * 180 / np.pi).tolist(),
                            "final_cost_value": raw_result.fun,
                            "success": raw_result.success,
                            "number_of_iterations": len(counts),
                            "number_of_evaluations": raw_result.nfev,
                            "optimization_message": str(raw_result.message) if hasattr(raw_result, 'message') else "Optimization completed"
                        }
                    }
                },
                "vqe_configuration": vqe_style_result.vqe_configuration,
                "optimization_results": vqe_style_result.optimization_results,
                "protein_info": vqe_style_result.protein_info,
                "measurement_results": vqe_style_result.measurement_results,
                "coordinates": result.protein_shape_file_gen.get_xyz_data(),
                "main_turns": result.protein_shape_decoder.main_turns,
                "side_turns": result.protein_shape_decoder.side_turns,
                "visualization_files": {
                    "convergence_plot": f"{base_filename}_convergence.png",
                    "state_probabilities": f"{base_filename}_probabilities.png",
                    "structure_plot": "structure.png",
                    "circuit_diagram": f"circuit_diagram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                }
            }
            
            # Convertir arrays de numpy a listas
            try:
                results_dict = numpy_to_list(results_dict)
            except Exception as e:
                print(f"Error al convertir resultados: {str(e)}")
                # Implementar manejo alternativo
    
            # Guardar en archivo JSON
            output_filename = f"{base_filename}.json"
            try:
                with open(output_filename, 'w') as f:
                    json.dump(results_dict, f, indent=4, default=safe_json_serialize)
                print(f"\nResults saved to {output_filename}")
            except Exception as e:
                if isinstance(e, (IOError, OSError)):
                    print(f"\nError saving JSON file: {str(e)}")
                else:
                    print(f"\nWarning: Non-critical issue while saving JSON: {str(e)}")

            # Imprimir análisis de error al final
            print("\nAnálisis de Error del Circuito:")
            print("--------------------------------")
            
            # Preparar métricas para el análisis
            qubit_metrics_dict = {i: {
                "quality_metrics": {
                    "iro": metrics[selected_qubits[i]].iro,
                    "readout_error": metrics[selected_qubits[i]].readout_error
                }
            } for i in range(len(selected_qubits))}
            
            # Calcular y mostrar probabilidad de éxito
            success_prob = _estimate_success_probability(circuit_transpiled[0], qubit_metrics_dict)
            print("\nProbabilidad de éxito estimada:")
            print(f"  Total: {success_prob['total_success_probability']:.4f}")
            print(f"  Puertas: {success_prob['gate_success_probability']:.4f}")
            print(f"  Medición: {success_prob['readout_success_probability']:.4f}")
            
            # Calcular y mostrar presupuesto de error
            error_budget = _calculate_error_budget(circuit_transpiled[0], qubit_metrics_dict)
            print("\nPresupuesto de error:")
            print(f"  Puertas de un qubit: {error_budget['single_qubit_gates']:.4f}")
            print(f"  Puertas de dos qubits: {error_budget['two_qubit_gates']:.4f}")
            print(f"  Mediciones: {error_budget['measurements']:.4f}")
            print(f"  Error total: {error_budget['total_error']:.4f}")
        except Exception as e:
            print(f"\nError saving JSON file: {str(e)}")

    print(f"\nTotal execution time: {execution_time:.2f} seconds")




