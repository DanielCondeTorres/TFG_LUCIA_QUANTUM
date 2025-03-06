# (C) Copyright Pauli("I")BM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LPauli("I")CENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LPauli("I")CENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Builds Pauli operators of a given size."""
# pauli_ops_builder.py
from typing import Set
from qiskit.quantum_info import SparsePauliOp

def _build_full_identity(num_qubits: int) -> SparsePauliOp:
    """
    Construye un operador de identidad completa para el número especificado de qubits
    utilizando una representación basada en cadenas de Pauli.

    Args:
        num_qubits (int): Número de qubits.

    Returns:
        SparsePauliOp: Operador de identidad para el número especificado de qubits.
    """
    if num_qubits < 1:
        raise ValueError("El número de qubits debe ser al menos 1.")
    
    pauli_str = 'I' * num_qubits
    return SparsePauliOp.from_list([(pauli_str, 1.0)])

def _build_pauli_z_op(num_qubits: int, pauli_z_indices: Set[int]) -> SparsePauliOp:
    """
    Construye un operador de Pauli Z en los índices especificados y la identidad en los demás.

    Args:
        num_qubits (int): Número total de qubits.
        pauli_z_indices (Set[int]): Conjunto de índices donde se aplica Pauli Z.

    Returns:
        SparsePauliOp: Operador de Pauli Z según los índices especificados.
    """
    if num_qubits < 1:
        raise ValueError("El número de qubits debe ser al menos 1.")
    
    pauli_list = ['I'] * num_qubits
    for idx in pauli_z_indices:
        if idx < 0 or idx >= num_qubits:
            raise ValueError(f"Índice de qubit {idx} fuera de rango para {num_qubits} qubits.")
        pauli_list[idx] = 'Z'
    
    pauli_str = ''.join(pauli_list)
    return SparsePauliOp.from_list([(pauli_str, 1.0)])
