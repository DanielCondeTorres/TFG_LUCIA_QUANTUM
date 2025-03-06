# (C) Copyright IBM 2021, 2022.
#
# Este código está licenciado bajo la Licencia Apache, Versión 2.0. Puedes
# obtener una copia de esta licencia en el archivo LICENSE.txt en el directorio raíz
# de este árbol de código fuente o en http://www.apache.org/licenses/LICENSE-2.0.
#
# Cualquier modificación o trabajo derivado de este código debe retener este
# aviso de copyright, y los archivos modificados deben llevar un aviso indicando
# que han sido alterados con respecto a los originales.
"""Clase abstracta que define un bead de un péptido."""
# base_bead.py
from abc import ABC
from typing import Tuple, Union, Callable, Optional

from qiskit.quantum_info import SparsePauliOp

from ..pauli_ops_builder import _build_full_identity
from ...residue_validator import _validate_residue_symbol


class BaseBead(ABC):
    """Clase abstracta que define un bead de un péptido."""

    def __init__(
        self,
        chain_type: str,
        main_index: int,
        residue_type: Optional[str],
        turn_qubits: Tuple[SparsePauliOp, SparsePauliOp],
        build_turn_indicator_fun_0: Callable[[], SparsePauliOp],
        build_turn_indicator_fun_1: Callable[[], SparsePauliOp],
        build_turn_indicator_fun_2: Callable[[], SparsePauliOp],
        build_turn_indicator_fun_3: Callable[[], SparsePauliOp],
    ):
        """
        Args:
            chain_type: Tipo de la cadena, ya sea "main_chain" o "side_chain".
            main_index: Índice del bead en la cadena principal en un péptido.
            residue_type: Un carácter que representa el tipo de residuo para el bead. Una cadena vacía en caso de bead lateral no existente.
            turn_qubits: Una tupla de dos operadores SparsePauliOp que codifican el giro que sigue a partir de un índice de bead dado.
            build_turn_indicator_fun_0: Método que construye funciones indicadoras de giro para el bead.
            build_turn_indicator_fun_1: Método que construye funciones indicadoras de giro para el bead.
            build_turn_indicator_fun_2: Método que construye funciones indicadoras de giro para el bead.
            build_turn_indicator_fun_3: Método que construye funciones indicadoras de giro para el bead.
        """
        self.chain_type = chain_type
        self.main_index = main_index
        self._residue_type = residue_type
        _validate_residue_symbol(residue_type)
        self._turn_qubits = turn_qubits

        if self._residue_type and self.turn_qubits is not None:
            '''
            if isinstance(turn_qubits[0], SparsePauliOp):
                # Obtener el número de qubits
                self._full_id = _build_full_identity(turn_qubits[0].num_qubits)
            elif isinstance(turn_qubits[0], (int, float)):
                # Usar como una cuenta aproximada de qubits
                self._full_id = _build_full_identity(int(turn_qubits[0]))
            else:
                raise TypeError("Tipo inesperado para turn_qubits[0]: {}".format(type(turn_qubits[0])))
            '''
            self._full_id = _build_full_identity(turn_qubits[0].num_qubits)
            # Construir las funciones indicadoras de giro utilizando SparsePauliOp
            self._turn_indicator_fun_0 = build_turn_indicator_fun_0()
            self._turn_indicator_fun_1 = build_turn_indicator_fun_1()
            self._turn_indicator_fun_2 = build_turn_indicator_fun_2()
            self._turn_indicator_fun_3 = build_turn_indicator_fun_3()

    @property
    def turn_qubits(self) -> Tuple[SparsePauliOp, SparsePauliOp]:
        """Retorna la tupla de dos operadores que codifican el giro que sigue desde el bead."""
        return self._turn_qubits

    @property
    def residue_type(self) -> Optional[str]:
        """Retorna el tipo de residuo."""
        return self._residue_type

    # Para el giro que conduce desde el bead
    @property
    def indicator_functions(
        self,
    ) -> Union[None, Tuple[SparsePauliOp, SparsePauliOp, SparsePauliOp, SparsePauliOp]]:
        """
        Retorna todas las funciones indicadoras de giro para el bead.
        Returns:
            Una tupla de todas las funciones indicadoras de giro para el bead.
        """
        #print(f"[DEBUG] indicator_functions, number of qubits = {self.turn_qubits[0].num_qubits}")
        if self.turn_qubits is None:
            return None
        return (
            self._turn_indicator_fun_0,
            self._turn_indicator_fun_1,
            self._turn_indicator_fun_2,
            self._turn_indicator_fun_3,
        )
