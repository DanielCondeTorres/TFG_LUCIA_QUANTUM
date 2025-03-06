# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""A class defining a side bead of a peptide."""
from typing import Tuple, Optional

from qiskit.quantum_info import SparsePauliOp

from .base_bead import BaseBead


class SideBead(BaseBead):
    """A class defining a side bead of a peptide."""

    def __init__(
        self,
        main_index: int,
        side_index: int,
        residue_type: Optional[str],
        turn_qubits: Tuple[SparsePauliOp, SparsePauliOp],
    ):
        """
        Args:
            main_index: Index of the bead on the main chain in a peptide to which the side
                        chain of this side bead is attached.
            side_index: Index of the bead on the related side chain in a peptide.
            residue_type: A character representing the type of a residue for the bead. Empty
                        string if a side bead does not exist.
            turn_qubits: A tuple of two SparsePauliOp operators that encode the turn following from a given
                         bead index.
        """
        super().__init__(
            "side_chain",
            main_index,
            residue_type,
            turn_qubits,
            self._build_turn_indicator_fun_0,
            self._build_turn_indicator_fun_1,
            self._build_turn_indicator_fun_2,
            self._build_turn_indicator_fun_3,
        )
        self.side_index = side_index

    def __str__(self):
        return (
            f"{self.chain_type}_{self.side_index}_main_chain_ind_{self.main_index}"
        )

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, SideBead):
            return False
        return (
            self.main_index == other.main_index
            and self.side_index == other.side_index
            and self.chain_type == other.chain_type
        )

    def _build_turn_indicator_fun_0(self) -> SparsePauliOp:
        I = self._full_id
        P0 = self._turn_qubits[0]
        P1 = self._turn_qubits[1]
        
        result = ((I - P0).dot(I - P1)).tensor(I)
        return result.simplify()

    def _build_turn_indicator_fun_1(self) -> SparsePauliOp:
        I = self._full_id
        P0 = self._turn_qubits[0]
        P1 = self._turn_qubits[1]
        
        result = (P1.dot(P1 - P0)).tensor(I)
        return result.simplify()

    def _build_turn_indicator_fun_2(self) -> SparsePauliOp:
        I = self._full_id
        P0 = self._turn_qubits[0]
        P1 = self._turn_qubits[1]
        
        result = (P0.dot(P0 - P1)).tensor(I)
        return result.simplify()

    def _build_turn_indicator_fun_3(self) -> SparsePauliOp:
        I = self._full_id
        P0 = self._turn_qubits[0]
        P1 = self._turn_qubits[1]
        
        result = (P0.dot(P1)).tensor(I)
        return result.simplify()
