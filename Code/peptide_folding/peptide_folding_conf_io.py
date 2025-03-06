"""
Módulo para manejar la entrada/salida y visualización de datos del plegamiento de péptidos.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time 
import base64
from io import BytesIO
import psutil
import platform
import subprocess
from qiskit.visualization import circuit_drawer

def numpy_to_list(obj):
    """Convierte arrays de numpy a listas para serialización JSON"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        # Eliminar información redundante en el diccionario
        if 'circuit_info' in obj and 'transpiled_metrics' in obj.get('circuit_analysis', {}):
            # Mantener solo una copia de las métricas del circuito
            if 'gate_counts' in obj['circuit_info']:
                del obj['circuit_info']['gate_counts']
            if 'depth' in obj['circuit_info']:
                del obj['circuit_info']['depth']
            if 'operations' in obj['circuit_info']:
                del obj['circuit_info']['operations']
            if 'physical_qubits' in obj['circuit_info']:
                del obj['circuit_info']['physical_qubits']
        
        # Simplificar final_layout para mostrar solo qubits relevantes
        if 'transpiled_info' in obj and 'final_layout' in obj['transpiled_info']:
            if isinstance(obj['transpiled_info']['final_layout'], dict):
                relevant_qubits = obj.get('physical_qubits', [])
                if relevant_qubits:
                    filtered_layout = {k: v for k, v in obj['transpiled_info']['final_layout'].items() 
                                    if int(k) in relevant_qubits or int(v) in relevant_qubits}
                    obj['transpiled_info']['final_layout'] = filtered_layout
        
        # Eliminar información redundante de optimización
        if 'optimization_results' in obj and 'circuit_analysis' in obj:
            if 'final_parameters' in obj['circuit_analysis'].get('optimization_progress', {}):
                del obj['circuit_analysis']['optimization_progress']['final_parameters']
        
        # Simplificar optimization_trace
        if 'optimization_trace' in obj:
            if 'iterations' in obj['optimization_trace']:
                del obj['optimization_trace']['iterations']
            if 'energies' in obj['optimization_trace'] and 'energy_values' in obj:
                del obj['optimization_trace']
        
        return {key: numpy_to_list(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    return obj

def safe_json_serialize(obj):
    """Función auxiliar para manejar tipos no serializables"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if str(type(obj)) == "<class 'qiskit.transpiler.layout.Layout'>":
        return obj.get_physical_bits()
    if str(type(obj)) == "<class 'qiskit.circuit.quantumregister.Qubit'>":
        return obj._index
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def plot_vqe_iterations(counts, values):
    """
    Grafica los resultados de las iteraciones de VQE.
    """
    try:
        fig = plt.figure()
        plt.plot(counts, values)
        plt.ylabel("Conformation Energy")
        plt.xlabel("VQE Iterations")
        plt.title("Energy of conformation vs VQE iterations")
        plt.grid(True)
        
        # Create a subplot if there are enough iterations
        if len(counts) > 40:
            fig.add_axes([0.44, 0.51, 0.44, 0.32])
            plt.plot(counts[40:], values[40:])
            plt.ylabel("Conformation Energy")
            plt.xlabel("VQE Iterations")
            plt.grid(True)
        
        plt.savefig("VQE_iterations.png")
        plt.show()
        plt.close()
        print("\nPlot with VQE iterations saved as 'VQE_iterations.png'")
    except Exception as e:
        print(f"\nError creating the plot: {str(e)}")

def get_hardware_info():
    """
    Get detailed hardware information, with special support for Apple Silicon.
    """
    system = platform.system()
    machine = platform.machine()
    
    # Specific information for Mac
    if system == "Darwin":
        try:
            # Get Mac model
            model = subprocess.check_output(["sysctl", "-n", "hw.model"]).decode().strip()
            
            # Get processor info for Mac
            if machine == "arm64":  # Apple Silicon
                processor = "Apple Silicon"
                # Try to get the specific model (M1, M2, M3, etc.)
                try:
                    proc_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                    if "M1" in proc_info:
                        processor = "Apple M1"
                    elif "M2" in proc_info:
                        processor = "Apple M2"
                    elif "M3" in proc_info:
                        processor = "Apple M3"
                except:
                    pass
            else:
                processor = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except:
            processor = platform.processor() or "Unknown"
            model = "Unknown Mac Model"
    else:
        processor = platform.processor()
        model = platform.machine()

    # RAM memory
    memory = psutil.virtual_memory()
    total_ram = round(memory.total / (1024**3), 2)  # Convert to GB
    available_ram = round(memory.available / (1024**3), 2)  # Convert to GB
    
    return {
        "system": system,
        "model": model,
        "processor": processor,
        "machine": machine,
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=False),  # Physical CPUs
        "cpu_count_logical": psutil.cpu_count(logical=True),  # Logical CPUs
        "ram_total_gb": total_ram,
        "ram_available_gb": available_ram
    }

def safe_serialize(obj):
    """
    Transform complex objects into serializable formats.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex64, np.complex128)):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        return {k: safe_serialize(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    return str(obj)

def get_circuit_representation(ansatz):
    """
    Get representations of the quantum circuit.
    """
    # Get ASCII representation
    ascii_circuit = ansatz.draw(output='text', fold=80)
    
    # Get circuit image
    try:
        # Create circuit image
        circuit_image = BytesIO()
        circuit_fig = circuit_drawer(ansatz, output='mpl')
        circuit_fig.savefig(circuit_image, format='png', bbox_inches='tight')
        circuit_image.seek(0)
        
        # Convert to base64
        base64_circuit = base64.b64encode(circuit_image.getvalue()).decode('utf-8')
        
        return {
            "ascii_representation": ascii_circuit,
            "image_base64": base64_circuit,
            "image_format": "png",
            "image_description": "Base64 encoded PNG image of the quantum circuit"
        }
    except Exception as e:
        print(f"Error al generar imagen del circuito: {str(e)}")
        return {
            "ascii_representation": ascii_circuit
        }

def save_circuit_representations(ansatz, timestamp):
    """
    Guarda las representaciones del circuito en archivos separados.
    """
    circuit_info = {
        "total_qubits": ansatz.num_qubits,
        "total_parameters": ansatz.num_parameters,
        "depth": ansatz.reps,
        "type": "RealAmplitudes",
        "description": "RealAmplitudes ansatz with alternating Ry rotations and CNOT layers"
    }
    
    try:
        # Save ASCII representation
        ascii_filename = f"circuit_ascii_{timestamp}.txt"
        ascii_str = str(ansatz)
        with open(ascii_filename, 'w', encoding='utf-8') as f:
            f.write(ascii_str)
            f.flush()
        circuit_info["ascii_file"] = ascii_filename
            
        # Save circuit image
        try:
            image_filename = f"circuit_diagram_{timestamp}.png"
            
            # Disable interactive mode
            plt.ioff()
            
            # Create a new figure
            plt.figure(figsize=(12, 6))
            
            # Draw the circuit
            circuit_drawer(ansatz, output='mpl')
            
            # Save and close
            plt.savefig(image_filename, bbox_inches='tight')
            plt.close('all')
            
            circuit_info["image_file"] = image_filename
            print(f"\nCircuit saved successfully in {image_filename}")
            
        except Exception as e:
            print(f"\nError saving circuit image: {str(e)}")
            
        return circuit_info
        
    except Exception as e:
        print(f"\nError saving circuit representations: {str(e)}")
        return circuit_info

    finally:
        plt.close('all')

def interpret_and_plot_results(raw_result, protein_folding_problem, args, qubit_op, ansatz, optimizer, execution_time):
    """
    Interpret and save VQE results.
    """
    results_dict = {}  # Initialize the dictionary outside the try block
    try:
        # Get interpreted result for XYZ coordinates
        result = protein_folding_problem.interpret(raw_result=raw_result)
        xyz_data = result.protein_shape_file_gen.get_xyz_data()
        
        # Convert xyz_data to a serializable format
        if isinstance(xyz_data, np.ndarray):
            xyz_data = xyz_data.tolist()
        elif hasattr(xyz_data, '__dict__'):
            xyz_data = xyz_data.__dict__
        
        # Get hardware information
        hardware_info = get_hardware_info()
        
        # Create a dictionary with detailed information about the angles
        def create_angle_info(i, value, num_qubits, reps, qubit_mapping):
            layer = i // num_qubits
            qubit_logical = i % num_qubits
            qubit_physical =qubit_mapping[qubit_logical] if qubit_mapping else qubit_logical
            return {
                "value": float(value),
                "qubit_logical": qubit_logical,
                "qubit_physical": qubit_physical,
                "layer": layer,
                "description": f"Ry rotation on logical qubit {qubit_logical} (physical qubit {qubit_physical}) in layer {layer}",
                "gate_type": "Ry",
                "radians": float(value),
                "degrees": float(value) * 180 / np.pi
            }

        qubit_mapping = None
        if hasattr(raw_result, 'aux_operators_evaluated') and raw_result.aux_operators_evaluated:
            try:
                if isinstance(raw_result.aux_operators_evaluated, dict):
                    qubit_mapping = raw_result.aux_operators_evaluated.get('qubit_mapping', None)
                else:
                    # Si es una string, intentar evaluarla como diccionario
                    import ast
                    aux_dict = ast.literal_eval(str(raw_result.aux_operators_evaluated))
                    qubit_mapping = aux_dict.get('qubit_mapping', None)
            except:
                qubit_mapping = None

        # Format XYZ coordinates
        xyz_formatted = []
        raw_xyz = result.protein_shape_file_gen.get_xyz_data()
        if isinstance(raw_xyz, np.ndarray):
            for i in range(len(raw_xyz)):
                try:
                    amino = str(raw_xyz[i][0])
                    x = float(raw_xyz[i][1])
                    y = float(raw_xyz[i][2])
                    z = float(raw_xyz[i][3])
                    xyz_formatted.append({
                        "amino_acid": amino,
                        "position": f"[{x:.4f}, {y:.4f}, {z:.4f}]"
                    })
                except Exception as e:
                    print(f"Error al procesar coordenadas para aminoácido {i}: {str(e)}")
                    xyz_formatted.append({
                        "amino_acid": str(raw_xyz[i][0]),
                        "position": f"[{str(raw_xyz[i][1])}, {str(raw_xyz[i][2])}, {str(raw_xyz[i][3])}]"
                    })
        
        # Get circuit representation
        circuit_representation = get_circuit_representation(ansatz)
        
        # Generate timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save circuit representations
        circuit_info = save_circuit_representations(ansatz, timestamp)
       
        result = protein_folding_problem.interpret(raw_result=raw_result)

    except Exception as e:
        print(f"\nError processing results: {str(e)}")

    # Show results in console
    print("\n\nRaw results from VQE:\n")
    print("\nNumber of cost function evaluations: ", raw_result.cost_function_evals)
    print("\nLowest energy found for the protein conformation: ", raw_result.optimal_value)
    print("\nBest parameters found for the angles of the quantum circuit: ", raw_result.optimal_point)
    #print("\nParámetros óptimos: ", raw_result.optimal_parameters)
    #print("\nAuxiliary operators evaluated: ", raw_result.aux_operators_evaluated)
    print("\nBest measurement: ", raw_result.best_measurement)
    print("\n")

    # Interpret and show results of protein folding
    #result = protein_folding_problem.interpret(raw_result=raw_result)
    print("\nThe bit string representing the protein shape during optimization is: ",
          result.turn_sequence)
    print("\nExtended bit string:", result.get_result_binary_vector())
    print(f"\nMain sequence of turns of the folded protein: {result.protein_shape_decoder.main_turns}")
    print(f"\nSequence of turns of the side chains: {result.protein_shape_decoder.side_turns}")

    # Show XYZ data corresponding to the folded protein
    print("\nXYZ corresponding to the folded protein:")
    print(result.protein_shape_file_gen.get_xyz_data())

    # Plot the protein structure
    fig = result.get_figure(title="Protein Structure", ticks=False, grid=True)
    fig.get_axes()[0].view_init(10, 70)
    plt.savefig("structure.png")
    plt.show()
    plt.close()
    print("\nPlot with the protein structure saved as 'structure.png'")

def plot_convergence(counts, values, filename='convergence_plot.png'):
    """
    Crear gráfica de convergencia del VQE
    """
    if not counts or not values:
        print("Warning: No hay datos de convergencia para graficar")
        return
        
    with plt.style.context('seaborn'):  # Mejor estilo por defecto
        plt.figure(figsize=(10, 6))
        plt.plot(counts, values, 'b-o', label='Energía', linewidth=2, markersize=6)
        plt.xlabel('Número de iteración')
        plt.ylabel('Energía')
        plt.title('Convergencia del VQE')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        min_energy = min(values)
        plt.text(0.02, 0.98, f'Energía mínima: {min_energy:.6f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nGráfica de convergencia guardada como '{filename}'")

def plot_state_probabilities(probabilities, filename='state_probabilities.png'):
    plt.clf()  # Limpiar figura anterior
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(probabilities.keys()), y=list(probabilities.values()))
    plt.xticks(rotation=45)
    plt.xlabel('Estado')
    plt.ylabel('Probabilidad')
    plt.title('Distribución de Probabilidades de Estados')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')  # Cerrar todas las figuras
    print(f"Gráfica de probabilidades guardada como '{filename}'")

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
                    "final_layout": circuit_transpiled[0].layout.final_layout if hasattr(circuit_transpiled[0], 'layout') else None
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