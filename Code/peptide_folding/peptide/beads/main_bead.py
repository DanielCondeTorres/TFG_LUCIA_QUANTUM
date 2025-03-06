# (C) Copyright IBM 2021, 2022.
#
# Este código está licenciado bajo la Licencia Apache, Versión 2.0. Puedes
# obtener una copia de esta licencia en el archivo LICENSE.txt en el directorio raíz
# de este árbol de código fuente o en http://www.apache.org/licenses/LICENSE-2.0.
#
# Cualquier modificación o trabajo derivado de este código debe retener este
# aviso de copyright, y los archivos modificados deben llevar un aviso indicando
# que han sido alterados con respecto a los originales.
"""Clase que define un bead principal de un péptido."""
# main_bead.py
from typing import Tuple
from qiskit.quantum_info import SparsePauliOp
from .base_bead import BaseBead
from ..chains.side_chain import SideChain


class MainBead(BaseBead):
    """Clase que define un bead principal de un péptido."""

    def __init__(
        self,
        main_index: int,
        residue_type: str,
        turn_qubits: Tuple[SparsePauliOp, SparsePauliOp],
        side_chain: SideChain,
    ):
        """
        Args:
            main_index: Índice del bead en la cadena principal de un péptido.
            residue_type: Un carácter que representa el tipo de residuo para el bead.
            turn_qubits: Una tupla de dos operadores SparsePauliOp que codifican el giro que sigue a partir de un índice de bead dado.
            side_chain: Un objeto que representa una cadena lateral adjunta a este bead principal.
        """
        #print(f"[DEBUG] Inicializando MainBead: index={main_index}, residue={residue_type}")
        super().__init__(
            "main_chain",
            main_index,
            residue_type,
            turn_qubits,
            self._build_turn_indicator_fun_0,
            self._build_turn_indicator_fun_1,
            self._build_turn_indicator_fun_2,
            self._build_turn_indicator_fun_3,
        )
        self._side_chain = side_chain

    def __str__(self):
        return self.chain_type + "_" + str(self.main_index)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, MainBead):
            return False
        return (
            self.main_index == other.main_index and self.chain_type == other.chain_type
        )

    def _build_turn_indicator_fun_0(self) -> SparsePauliOp:
        """
        Construye la función indicadora de giro 0.
        """
        
        #print("DEBUG: MainBead._build_turn_indicator_fun_0, self._full_id:", self._full_id, "self._turn_qubits[0]:", self._turn_qubits[0], "self._turn_qubits[1]:", self._turn_qubits[1])
        
        I = self._full_id
        P0 = self._turn_qubits[0]
        P1 = self._turn_qubits[1]
        
        # Tensoriza con _full_id si es necesario
        result = I.tensor((I - P0).dot(I - P1))
        
        #print("DEBUG: MainBead._build_turn_indicator_fun_0, result:", result)
        #print("DEBUG: MainBead._build_turn_indicator_fun_0, result.simplify():", result.simplify())
        
        return result.simplify()

    def _build_turn_indicator_fun_1(self) -> SparsePauliOp:
        """
        Construye la función indicadora de giro 1.
        """
        I = self._full_id
        P0 = self._turn_qubits[0]
        P1 = self._turn_qubits[1]
        
        result = I.tensor(P1.dot(P1 - P0))
        
        #print("DEBUG: MainBead._build_turn_indicator_fun_1, result:", result)
        #print("DEBUG: MainBead._build_turn_indicator_fun_1, result.simplify():", result.simplify())
        
        return result.simplify()

    def _build_turn_indicator_fun_2(self) -> SparsePauliOp:
        """
        Construye la función indicadora de giro 2.
        """
        
        I = self._full_id
        P0 = self._turn_qubits[0]
        P1 = self._turn_qubits[1]
        
        result = I.tensor(P0.dot(P0 - P1))
        
        #print("DEBUG: MainBead._build_turn_indicator_fun_2, result:", result)
        #print("DEBUG: MainBead._build_turn_indicator_fun_2, result.simplify():", result.simplify())
        
        return result.simplify()

    def _build_turn_indicator_fun_3(self) -> SparsePauliOp:
        """
        Construye la función indicadora de giro 3.
        """
        
        I = self._full_id
        P0 = self._turn_qubits[0]
        P1 = self._turn_qubits[1]
        
        result = I.tensor(P0.dot(P1))
        
        #print("DEBUG: MainBead._build_turn_indicator_fun_3, result:", result)
        #print("DEBUG: MainBead._build_turn_indicator_fun_3, result.simplify():", result.simplify())
        
        return result.simplify()

    @property
    def side_chain(self) -> SideChain:
        """Retorna una cadena lateral."""
        return self._side_chain
