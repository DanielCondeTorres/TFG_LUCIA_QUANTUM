"""
Módulo para selección y gestión de qubits físicos
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class QubitMetrics:
    irb: float
    iro: float
    readout_error: float
    quality_metric: float
    connectivity_count: int
    combined_metric: float

    def __post_init__(self) -> None:
        """Validación de tipos y valores"""
        assert 0 <= self.irb <= 1, f"IRB debe estar entre 0 y 1, got {self.irb}"
        assert 0 <= self.iro <= 1, f"IRO debe estar entre 0 y 1, got {self.iro}"
        assert self.readout_error > 0, f"Readout error debe ser positivo, got {self.readout_error}"

class QmioQubitSelector:
    def __init__(self):
        logger.info("Inicializando QmioQubitSelector")
        self.qubit_properties = {
            # Datos reales del hardware QMIO
            0: (0.996, 0.637, 10278.107),
            1: (0.999, 0.923, 9652.645),
            2: (0.995, 0.825, 10076.506),
            3: (1.000, 0.915, 9850.835),
            4: (1.000, 0.711, 9644.855),
            5: (0.991, 0.885, 9848.259),
            6: (0.999, 0.948, 10271.046),
            7: (0.988, 0.755, 9849.595),
            8: (0.999, 0.887, 9850.778),
            9: (0.933, 0.796, 10077.584),
            10: (0.999, 0.887, 10285.586),
            11: (0.999, 0.922, 9644.062),
            12: (1.000, 0.919, 10071.312),
            13: (0.995, 0.934, 10267.817),
            14: (1.000, 0.812, 9852.605),
            15: (0.999, 0.937, 9653.470),
            16: (1.000, 0.942, 10283.733),
            17: (0.768, 0.832, 10065.888),
            18: (1.000, 0.930, 10278.374),
            19: (1.000, 0.880, 9843.772),
            20: (0.999, 0.905, 9636.346),
            21: (0.999, 0.893, 10087.091),
            22: (1.000, 0.863, 10259.890),
            23: (0.997, 0.866, 9645.613),
            24: (0.995, 0.952, 10249.683),
            25: (0.982, 0.376, 9847.277),
            26: (0.999, 0.827, 10057.022),
            27: (0.995, 0.952, 10249.683),
            28: (0.982, 0.376, 9847.277),
            29: (0.999, 0.327, 10057.022),
            30: (0.998, 0.376, 10235.421),
            31: (1.000, 0.900, 9625.546)
        }
        
        # Mapa de conectividad completo del QMIO
        self.coupling_map = [
            # Conexiones verticales
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (6, 7), (7, 8), (8, 9), (9, 10),
            (11, 12), (12, 13),
            (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20), (20, 21),
            (22, 23), (23, 24),
            (25, 26), (26, 27),
            (28, 29), (29, 30), (30, 31),
            # Conexiones horizontales/diagonales
            (0, 7), (1, 8), (2, 9), (3, 10),
            (6, 11), (7, 12), (8, 13),
            (11, 14), (12, 15), (13, 16),
            (14, 17), (15, 18), (16, 19),
            (17, 22), (18, 23), (19, 24),
            (22, 25), (23, 26), (24, 27),
            (25, 28), (26, 29), (27, 30),
            (28, 31)
        ]
        self._initialize_connectivity()

    def _initialize_connectivity(self):
        """Inicializa el mapa de conectividad"""
        self.connectivity = {i: [] for i in range(32)}
        for q1, q2 in self.coupling_map:
            self.connectivity[q1].append(q2)
            self.connectivity[q2].append(q1)

    def get_best_qubits(self, n_required: int) -> List[int]:
        logger.info(f"Solicitando {n_required} qubits")
        metrics = self._calculate_metrics()
        selected = self._select_qubits(metrics, n_required)
        logger.info(f"Qubits seleccionados: {selected}")
        return selected

    def _calculate_metrics(self) -> Dict[int, QubitMetrics]:
        logger.debug("Calculando métricas para todos los qubits")
        metrics = {}
        for qubit, (irb, iro, re) in self.qubit_properties.items():
            quality_metric = (1 - irb) + iro + (re / 10000)
            connectivity_count = len(self.connectivity[qubit])
            combined_metric = quality_metric * (1 + 1/connectivity_count)
            
            metrics[qubit] = QubitMetrics(
                irb=irb,
                iro=iro,
                readout_error=re,
                quality_metric=quality_metric,
                connectivity_count=connectivity_count,
                combined_metric=combined_metric
            )
        return metrics

    def _select_qubits(self, metrics: Dict[int, QubitMetrics], n_required: int) -> List[int]:
        """Selecciona los mejores qubits basándose en las métricas"""
        selected_qubits = []
        available_qubits = sorted(metrics.items(), key=lambda x: x[1].combined_metric)
        
        # Seleccionar primer qubit (el mejor)
        selected_qubits.append(available_qubits[0][0])
        
        # Seleccionar qubits restantes
        while len(selected_qubits) < n_required:
            best_next_qubit = None
            best_connections = -1
            best_quality = float('inf')
            
            for qubit, quality in available_qubits:
                if qubit not in selected_qubits:
                    connections = sum(1 for sq in selected_qubits if qubit in self.connectivity[sq])
                    if connections > best_connections or (connections == best_connections and quality.combined_metric < best_quality):
                        best_next_qubit = qubit
                        best_connections = connections
                        best_quality = quality.combined_metric
            
            if best_next_qubit is not None:
                selected_qubits.append(best_next_qubit)
            else:
                raise ValueError(f"No se pueden encontrar {n_required} qubits con suficiente conectividad")
        
        # Imprimir información detallada
        print(f"\nQubits seleccionados: {selected_qubits}")
        print("Métricas de calidad de los qubits seleccionados:")
        for q in selected_qubits:
            m = metrics[q]
            print(f"  Qubit {q}:")
            print(f"    IRB={m.irb:.3f}, IRO={m.iro:.3f}, RE={m.readout_error:.1f}")
            print(f"    Quality={m.quality_metric:.3f}, Connectivity={m.connectivity_count}")
            print(f"    Combined Metric={m.combined_metric:.3f}")
        
        print("\nConectividad entre qubits seleccionados:")
        for i, q1 in enumerate(selected_qubits):
            for q2 in selected_qubits[i+1:]:
                if q2 in self.connectivity[q1]:
                    print(f"  {q1} <-> {q2}")
        
        return selected_qubits