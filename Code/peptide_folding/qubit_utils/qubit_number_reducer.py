# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Removes qubit registers that are not relevant for the problem."""
from typing import Union, List, Dict, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Pauli


def remove_unused_qubits(
    total_hamiltonian: SparsePauliOp
) -> Tuple[SparsePauliOp, List[int]]:
    """
    Removes those qubits from a total Hamiltonian that are equal to an identity operator across
    all terms, i.e., they are irrelevant para el problema. Reduce el número de qubits requeridos
    para codificar el problema.

    Args:
        total_hamiltonian: A full Hamiltonian for the protein folding problem as a SparsePauliOp.

    Returns:
        Tuple consisting of the total_hamiltonian compressed to an equivalent Hamiltonian and
        indices of qubits in the original Hamiltonian that were unused as optimization variables.
    """
    unused_qubits = _find_unused_qubits(total_hamiltonian)
    num_qubits = total_hamiltonian.num_qubits

    compressed_hamiltonian = _compress_sparse_pauli_op(
        num_qubits, total_hamiltonian, unused_qubits
    )

    return compressed_hamiltonian, unused_qubits


def _compress_sparse_pauli_op(
    num_qubits: int,
    total_hamiltonian: SparsePauliOp,
    unused_qubits: List[int],
) -> SparsePauliOp:
    """
    Comprime un SparsePauliOp eliminando los qubits no utilizados.

    Args:
        num_qubits: Número total de qubits en el Hamiltoniano original.
        total_hamiltonian: El Hamiltoniano a comprimir.
        unused_qubits: Lista de qubits a eliminar.

    Returns:
        Un SparsePauliOp comprimido con los qubits no utilizados eliminados.
    """
    if not unused_qubits:
        # No hay qubits por eliminar
        return total_hamiltonian

    # Crear una máscara booleana donde True indica mantener el qubit
    mask = np.ones(num_qubits, dtype=bool)
    mask[unused_qubits] = False

    # Filtrar las etiquetas de Pauli para eliminar los qubits no utilizados
    new_paulis = []
    new_coeffs = []

    for pauli, coeff in zip(total_hamiltonian.paulis, total_hamiltonian.coeffs):
        # Obtener la etiqueta del pauli (e.g., 'IXYZ')
        pauli_label = pauli.to_label()
        # Filtrar los caracteres correspondientes a qubits utilizados
        new_pauli_label = ''.join(
            char for char, keep in zip(pauli_label, mask) if keep
        )
        new_paulis.append(new_pauli_label)
        new_coeffs.append(coeff)

    # Crear un nuevo SparsePauliOp comprimido sin el argumento 'sparse'
    compressed_sparse_pauli_op = SparsePauliOp(new_paulis, coeffs=new_coeffs)
    # Combinar términos duplicados y simplificar
    compressed_sparse_pauli_op = compressed_sparse_pauli_op.simplify()

    return compressed_sparse_pauli_op


def _find_unused_qubits(total_hamiltonian: SparsePauliOp) -> List[int]:
    """
    Identifica qubits que no se usan (es decir, siempre identidad) en el Hamiltoniano.

    Args:
        total_hamiltonian: El Hamiltoniano a analizar como un SparsePauliOp.

    Returns:
        Lista de índices de qubits que no se usan.
    """
    num_qubits = total_hamiltonian.num_qubits
    used_qubits: Dict[int, bool] = {ind: False for ind in range(num_qubits)}

    for pauli in total_hamiltonian.paulis:
        pauli_label = pauli.to_label()
        for ind, char in enumerate(pauli_label):
            if char != 'I':
                used_qubits[ind] = True

    # Identificar qubits que no están marcados como utilizados
    unused = [ind for ind, used in used_qubits.items() if not used]
    return unused




'''
from typing import Union, List, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp


def remove_unused_qubits(
    total_hamiltonian: SparsePauliOp
) -> Tuple[SparsePauliOp, List[int]]:
    """
    Removes those qubits from a total Hamiltonian that are equal to an identity operator across
    all terms, i.e., they are irrelevant for the problem. It makes the number of qubits required
    for encoding the problem smaller or equal.

    Args:
        total_hamiltonian: A full Hamiltonian for the protein folding problem.

    Returns:
        Tuple consisting of the total_hamiltonian compressed to an equivalent Hamiltonian and
        indices of qubits in the original Hamiltonian that were unused as optimization variables.
    """
    unused_qubits = _find_unused_qubits(total_hamiltonian)
    num_qubits = total_hamiltonian.num_qubits

    return (
        _compress_sparse_pauli_op(num_qubits, total_hamiltonian, unused_qubits),
        unused_qubits,
    )


def _find_unused_qubits(total_hamiltonian: SparsePauliOp) -> List[int]:
    """
    Identifies qubits that are not used in any of the terms of the Hamiltonian.

    Args:
        total_hamiltonian: The SparsePauliOp representing the Hamiltonian.

    Returns:
        A list of indices representing the qubits that are not used.
    """
    num_qubits = total_hamiltonian.num_qubits
    unused_qubits = []

    for qubit in range(num_qubits):
        is_unused = True
        for pauli in total_hamiltonian.table:
            if pauli.z[qubit] or pauli.x[qubit]:
                is_unused = False
                break
        if is_unused:
            unused_qubits.append(qubit)

    return unused_qubits


def _update_used_map(pauli_table: np.ndarray, used_qubits: List[bool]) -> None:
    """
    Updates the used qubits map to indicate which qubits are used in the Pauli table.

    Args:
        pauli_table: The Pauli table representing the Hamiltonian.
        used_qubits: A list indicating whether each qubit is used or not.
    """
    for pauli in pauli_table:
        for i in range(len(used_qubits)):
            if pauli.z[i] or pauli.x[i]:
                used_qubits[i] = True


def _calc_reduced_pauli_tables(
    num_qubits: int, table_x: np.ndarray, table_z: np.ndarray, unused_qubits: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the reduced Pauli tables by removing the unused qubits.

    Args:
        num_qubits: Total number of qubits in the original Hamiltonian.
        table_x: The X part of the Pauli table.
        table_z: The Z part of the Pauli table.
        unused_qubits: List of qubits to be removed.

    Returns:
        A tuple containing the reduced X and Z parts of the Pauli table.
    """
    new_table_x = np.delete(table_x, unused_qubits, axis=1)
    new_table_z = np.delete(table_z, unused_qubits, axis=1)
    return new_table_x, new_table_z


def _compress_sparse_pauli_op(
    num_qubits: int,
    total_hamiltonian: SparsePauliOp,
    unused_qubits: List[int],
) -> SparsePauliOp:
    """
    Compresses a SparsePauliOp by removing unused qubits.

    Args:
        num_qubits: Total number of qubits in the original Hamiltonian.
        total_hamiltonian: The SparsePauliOp representing the Hamiltonian.
        unused_qubits: List of qubits to be removed.

    Returns:
        A compressed SparsePauliOp with the unused qubits removed.
    """
    pauli_table = total_hamiltonian.table
    coeffs = total_hamiltonian.coeffs

    # Create new tables excluding the unused qubits
    new_z = []
    new_x = []
    for i in range(len(pauli_table)):  # Iterate through each Pauli in the table
        z_part = pauli_table[i].z
        x_part = pauli_table[i].x

        # Remove the unused qubits from both Z and X parts
        reduced_z = np.delete(z_part, unused_qubits)
        reduced_x = np.delete(x_part, unused_qubits)

        new_z.append(reduced_z)
        new_x.append(reduced_x)

    # Convert lists to numpy arrays for SparsePauliOp creation
    new_z = np.array(new_z, dtype=bool)
    new_x = np.array(new_x, dtype=bool)

    # Create a new SparsePauliOp with the modified Pauli table and original coefficients
    compressed_operator = SparsePauliOp.from_dense((new_z, new_x), coeffs)

    return compressed_operator


def _compress_pauli_sum_op(
    num_qubits: int,
    total_hamiltonian: SparsePauliOp,
    unused_qubits: List[int],
) -> SparsePauliOp:
    """
    Compresses a SparsePauliOp by removing unused qubits from each term.

    Args:
        num_qubits: Total number of qubits in the original Hamiltonian.
        total_hamiltonian: The SparsePauliOp representing the Hamiltonian.
        unused_qubits: List of qubits to be removed.

    Returns:
        A compressed SparsePauliOp with the unused qubits removed from each term.
    """
    pauli_table = total_hamiltonian.table
    coeffs = total_hamiltonian.coeffs

    # Create new tables excluding the unused qubits
    new_z = []
    new_x = []
    new_coeffs = []
    for i in range(len(pauli_table)):  # Iterate through each Pauli in the table
        z_part = pauli_table[i].z
        x_part = pauli_table[i].x

        # Remove the unused qubits from both Z and X parts
        reduced_z = np.delete(z_part, unused_qubits)
        reduced_x = np.delete(x_part, unused_qubits)

        new_z.append(reduced_z)
        new_x.append(reduced_x)
        new_coeffs.append(coeffs[i])

    # Convert lists to numpy arrays for SparsePauliOp creation
    new_z = np.array(new_z, dtype=bool)
    new_x = np.array(new_x, dtype=bool)
    new_coeffs = np.array(new_coeffs, dtype=complex)

    # Create a new SparsePauliOp with the modified Pauli table and original coefficients
    compressed_operator = SparsePauliOp.from_dense((new_z, new_x), new_coeffs)

    return compressed_operator

# Nota: Esta transformación asegura que el código sea compatible con `SparsePauliOp`.
# Es necesario probarlo para garantizar su correcto funcionamiento en todos los casos previstos.
'''