# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# protein_folding_problem.py
"""Defines a protein folding problem that can be passed to algorithms."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, Any
import json
from qiskit.quantum_info import SparsePauliOp

from .interactions.interaction import Interaction
from .penalty_parameters import PenaltyParameters
from .peptide.peptide import Peptide
from .qubit_op_builder import QubitOpBuilder
from .qubit_utils.qubit_number_reducer import remove_unused_qubits
from .sampling_problem import SamplingProblem

if TYPE_CHECKING:
    from .protein_folding_result import ProteinFoldingResult


class ProteinFoldingProblem(SamplingProblem):
    """Defines a protein folding problem that can be passed to algorithms. Example initialization:

    .. code-block:: python

        penalty_terms = PenaltyParameters(15, 15, 15)
        main_chain_residue_seq = "SAASSASAAG"
        side_chain_residue_sequences = ["", "", "A", "A", "A", "A", "A", "A", "S", ""]
        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)
        mj_interaction = MiyazawaJerniganInteraction()
        protein_folding_problem = ProteinFoldingProblem(
            peptide, 
            mj_interaction, 
            penalty_terms,
            axis=2,
            weight_interface=5.0,
            relative_displacement_from_interface=0.5
        )
        qubit_op = protein_folding_problem.qubit_op()
    """

    def __init__(
        self,
        peptide: Peptide,
        interaction: Interaction,
        penalty_parameters: PenaltyParameters,
        axis: int = 2,
        weight_interface: float = 5.0,
        relative_displacement_from_interface: float = 0.5,
    ):
        """
        Args:
            peptide: A peptide object that defines the protein subject to the folding problem.
            interaction: A type of interaction between the beads of the protein (e.g., random, Miyazawa-Jernigan).
            penalty_parameters: Parameters that define the penalties for various features (chirality, backbone, etc.).
            axis: Eje perpendicular a la interfase (default: 2).
            weight_interface: Peso para el término de la interfase (default: 5.0).
            relative_displacement_from_interface: Desplazamiento relativo desde la interfase para el bead 0 (default: 0.5).
        """
        self._peptide = peptide
        self._interaction = interaction
        self._penalty_parameters = penalty_parameters
        self._axis = axis
        self._weight_interface = weight_interface
        self._relative_displacement_from_interface = relative_displacement_from_interface

        # Calcular la matriz de energías de interacción
        self._pair_energies = interaction.calculate_energy_matrix(peptide.get_main_chain.main_chain_residue_sequence)
        #print(f"[DEBUG] Matriz de energías de pares calculada: {self._pair_energies}")

        # Crear el builder de operador cuántico con los nuevos parámetros
        self._qubit_op_builder = QubitOpBuilder(
            peptide, 
            self._pair_energies, 
            penalty_parameters,
            axis=self._axis,
            weight_interface=self._weight_interface,
            relative_displacement_from_interface=self._relative_displacement_from_interface
        )
        #print("[DEBUG] QubitOpBuilder inicializado.")

        # Construir el operador cuántico (SparsePauliOp)
        total_hamiltonian = self._qubit_op_builder.build_qubit_op()
        #print("[DEBUG] Total Hamiltonian construido.")

        # Reducir el número de qubits si es posible
        self._unused_qubits = []
        self._qubit_op_reduced, self._unused_qubits = remove_unused_qubits(total_hamiltonian)
        #print("[DEBUG] QubitOp reducido con ", len(self._unused_qubits), " qubits no utilizados.")
        # Guardar axis en un archivo JSON
        data_to_save = {
            "axis": self._axis,
            "weight_interface": self._weight_interface,
            "relative_displacement_from_interface": self._relative_displacement_from_interface
            }

        with open("config.json", "w") as file:
            json.dump(data_to_save, file)
    def qubit_op(self) -> Union[SparsePauliOp, None]:
        """
        Builds the full qubit operator for the protein folding problem.

        Returns:
            SparsePauliOp: The qubit operator representing the problem.
        """
        if self._qubit_op_reduced is not None:
            return self._qubit_op_reduced
        return self._qubit_op_full()

    def _qubit_op_full(self) -> Union[SparsePauliOp, None]:
        """
        Builds the full qubit operator without compression.

        Returns:
            SparsePauliOp: The full qubit operator.
        """
        #print("[DEBUG] Construyendo qubit operator completo.")
        qubit_operator = self._qubit_op_builder.build_qubit_op()

        if qubit_operator is None:
            print("[ERROR] _qubit_op_full: build_qubit_op() retornó None.")
            return qubit_operator

        #print(f"[DEBUG] _qubit_op_full: build_qubit_op() retornó un operador con {qubit_operator.num_qubits} qubits.")
        return qubit_operator

    def interpret(self, raw_result: Any) -> ProteinFoldingResult:
        """
        Interprets the raw algorithm result, in the context of this problem, and returns a
        ProteinFoldingResult. The returned class can plot the protein and generate a
        .xyz file with the coordinates of each of its atoms.

        Args:
            raw_result: The raw result of solving the protein folding problem.

        Returns:
            ProteinFoldingResult: An instance that contains the protein folding result.
        """
        # pylint: disable=import-outside-toplevel
        from .protein_folding_result import ProteinFoldingResult

        # Verificar que raw_result tenga los atributos esperados
        if not hasattr(raw_result, 'eigenstate') or not hasattr(raw_result.eigenstate, 'binary_probabilities'):
            raise AttributeError("raw_result debe tener el atributo 'eigenstate' con método 'binary_probabilities'.")

        probs = raw_result.eigenstate.binary_probabilities()
        best_turn_sequence = max(probs, key=probs.get)
        return ProteinFoldingResult(
            unused_qubits=self.unused_qubits,
            peptide=self.peptide,
            turn_sequence=best_turn_sequence,
        )

    @property
    def unused_qubits(self) -> List[int]:
        """Returns the list of indices for qubits in the original problem formulation that were
        removed during compression."""
        return self._unused_qubits

    @property
    def peptide(self) -> Peptide:
        """Returns the peptide defining the protein subject to the folding problem."""
        return self._peptide

    @property
    def axis(self) -> int:
        """Returns the axis perpendicular to the interface."""
        return self._axis

    @property
    def weight_interface(self) -> float:
        """Returns the weight for the interface term."""
        return self._weight_interface

    @property
    def relative_displacement_from_interface(self) -> float:
        """Returns the relative displacement from the interface for bead 0."""
        return self._relative_displacement_from_interface
