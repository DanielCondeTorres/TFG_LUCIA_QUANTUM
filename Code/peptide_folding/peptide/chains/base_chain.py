# (C) Copyright IBM 2021, 2022.
#
# Este código está licenciado bajo la Licencia Apache, Versión 2.0. Puedes
# obtener una copia de esta licencia en el archivo LICENSE.txt en el directorio raíz
# de este árbol de código fuente o en http://www.apache.org/licenses/LICENSE-2.0.
#
# Cualquier modificación o trabajo derivado de este código debe retener este
# aviso de copyright, y los archivos modificados deben llevar un aviso indicando
# que han sido alterados con respecto a los originales.
"""Clase abstracta que define una cadena de un péptido."""
# base_chain.py
from qiskit.quantum_info import SparsePauliOp
from abc import ABC
from typing import List, Sequence, Optional, Set

from ..beads.base_bead import BaseBead
from ..pauli_ops_builder import (
    _build_full_identity,
    _build_pauli_z_op,
)

class BaseChain(ABC):
    """Clase abstracta que define una cadena de un péptido."""

    def __init__(self, beads_list: Sequence[BaseBead]):
        """
        Args:
            beads_list: Una lista de beads que definen la cadena.
        """
        self._beads_list = beads_list

    def __getitem__(self, item):
        return self._beads_list[item]

    def __len__(self):
        return len(self._beads_list)

    @property
    def beads_list(self) -> Sequence[BaseBead]:
        """Retorna la lista de todos los beads en la cadena."""
        return self._beads_list

    @property
    def residue_sequence(self) -> List[Optional[str]]:
        """
        Retorna la lista de todas las secuencias de residuos en la cadena.
        """
        residue_sequence = []
        for bead in self._beads_list:
            residue_sequence.append(bead.residue_type)
        return residue_sequence

    @staticmethod
    def _build_turn_qubit(chain_len: int, pauli_z_index: int) -> SparsePauliOp:
        """
        Construye un operador de Pauli que codifica el giro que sigue desde un índice dado de bead.

        Args:
            chain_len: Longitud de la cadena.
            pauli_z_index: Índice de un operador Pauli Z en un operador de giro.

        Returns:
            SparsePauliOp: Un operador que codifica el giro que sigue desde un índice dado de bead.
        """
        num_turn_qubits = 2 * (chain_len - 1)
        norm_factor = 0.5

        #print(f"[DEBUG] Construyendo turn_qubit con chain_len={chain_len}, pauli_z_index={pauli_z_index}, num_turn_qubits={num_turn_qubits}")

        # Validar que num_turn_qubits no sea excesivamente grande
        MAX_QUBITS = 32  # Ajusta este valor según tus necesidades y recursos
        if num_turn_qubits > MAX_QUBITS:
            raise ValueError(f"El número de qubits de giro {num_turn_qubits} excede el límite máximo permitido de {MAX_QUBITS}.")

        # Ajustar el índice para alinear con la versión antigua (invertir el índice)
        adjusted_pauli_z_index = num_turn_qubits - pauli_z_index - 1

        # Construir identidad completa
        identity_op = _build_full_identity(num_turn_qubits)

        # Construir Pauli Z en el índice especificado
        pauli_z_op = _build_pauli_z_op(num_turn_qubits, {adjusted_pauli_z_index})

        # Combinar los operadores con el factor de normalización
        # 'SparsePauliOp' soporta operaciones aritméticas como la suma y resta
        turn_qubit = (norm_factor * identity_op) - (norm_factor * pauli_z_op)
        #print(f"[DEBUG] Operador turn_qubit construido: {turn_qubit}")
        return turn_qubit
