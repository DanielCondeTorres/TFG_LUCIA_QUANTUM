# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Changes certain qubits to fixed values."""

from typing import Union
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Pauli

def _fix_qubits(
    operator: Union[int, SparsePauliOp],
    has_side_chain_second_bead: bool = False,
) -> Union[int, SparsePauliOp]:
    """
    Assign predefined values to certain qubits in the main chain.
    Qubits at positions 0, 1, 2, 3, and optionally 5 are fixed to predefined values.

    Args:
        operator: An operator whose qubits shall be fixed.
        has_side_chain_second_bead: Boolean flag to determine if the fifth qubit should be fixed.

    Returns:
        An operator with relevant qubits changed to fixed values.
    """
    # Return the operator if it is not a valid SparsePauliOp type
    if not isinstance(operator, SparsePauliOp):
        return operator

    # Simplify any redundant terms in the operator
    operator = operator.simplify()

    # Extract the Pauli table and coefficients
    table_z = np.copy(operator.paulis.z)
    table_x = np.copy(operator.paulis.x)
    coeffs = np.copy(operator.coeffs)

    # Update coefficients and preset binary values for fixed qubits
    for i in range(len(coeffs)):
        coeffs[i] = _update_coeffs(coeffs[i], table_z[i], has_side_chain_second_bead)
        _preset_binary_vals(table_z[i], has_side_chain_second_bead)

    # Create the updated SparsePauliOp with new tables and coefficients
    pauli_list = [Pauli((z, x)) for z, x in zip(table_z, table_x)]
    operator_updated = SparsePauliOp(pauli_list, coeffs=coeffs).simplify()
    return operator_updated

def _update_coeffs(coeffs: complex, table_z: np.ndarray, has_side_chain_second_bead: bool) -> complex:
    """
    Update the coefficients based on fixed qubit values.

    Args:
        coeffs: The original coefficient of the operator.
        table_z: The Z part of the Pauli representation.
        has_side_chain_second_bead: Boolean flag to determine if the fifth qubit should be considered.

    Returns:
        Updated coefficient.
    """
    # If the second qubit (index 1) is True, invert the coefficient
    if len(table_z) > 1 and table_z[1]:
        coeffs = -coeffs
    # If the fifth qubit (index 5) is True and there is no side chain, invert the coefficient
    if not has_side_chain_second_bead and len(table_z) > 6 and table_z[5]:
        coeffs = -coeffs
    return coeffs

def _preset_binary_vals(table_z: np.ndarray, has_side_chain_second_bead: bool):
    """
    Set predefined binary values for specific qubits.

    Args:
        table_z: The Z part of the Pauli representation.
        has_side_chain_second_bead: Boolean flag to determine if the fifth qubit should be fixed.
    """
    # Indices of main chain qubits to be fixed
    main_beads_indices = [0, 1, 2, 3]
    if not has_side_chain_second_bead:
        main_beads_indices.append(5)
    for index in main_beads_indices:
        _preset_single_binary_val(table_z, index)

def _preset_single_binary_val(table_z: np.ndarray, index: int):
    """
    Set a specific qubit to False (0) if it exists.

    Args:
        table_z: The Z part of the Pauli representation.
        index: The index of the qubit to be set.
    """
    if index < len(table_z):
        table_z[index] = False
