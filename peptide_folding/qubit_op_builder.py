# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# qubit_op_builder.py
"""Builds qubit operators for all Hamiltonian terms in the protein folding problem."""
from typing import Union, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .bead_contacts.contact_map import ContactMap
from .bead_distances.distance_map import DistanceMap
from .exceptions.invalid_side_chain_exception import InvalidSideChainException
from .exceptions.invalid_size_exception import InvalidSizeException
from .penalty_parameters import PenaltyParameters
from .peptide.pauli_ops_builder import _build_full_identity
from .qubit_utils.qubit_fixing import _fix_qubits
from .peptide.beads.base_bead import BaseBead
from .peptide.peptide import Peptide
from .interactions.resources.FaucherePliska import FaucherePliskaInteractions

# pylint: disable=too-few-public-methods

class QubitOpBuilder:
    """Builds qubit operators for all Hamiltonian terms in the protein folding problem."""
    def __init__(self, peptide: Peptide, pair_energies: np.ndarray, penalty_parameters: PenaltyParameters,
                 axis: int = 2, weight_interface: float = 5.0, 
                 relative_displacement_from_interface: float = 0.5):
        """Initializes the QubitOpBuilder.
        Args:
            peptide: A Peptide object that includes all information about a protein.
            pair_energies: Numpy array of pair energies for amino acids.
            penalty_parameters: A PenaltyParameters object storing the values of all penalty parameters.
            axis: Eje perpendicular a la interfase (default: 2)
            weight_interface: Peso para el término de la interfase (default: 5.0)
            relative_displacement_from_interface: Desplazamiento relativo desde la interfase para el bead 0 (default: 0.5)
        """
        self._peptide = peptide
        self._pair_energies = pair_energies
        self._penalty_parameters = penalty_parameters
        self._contact_map = ContactMap(peptide)
        self._distance_map = DistanceMap(peptide)
        _side_chain_hot_vector = self._peptide.get_side_chain_hot_vector()
        self._has_side_chain_second_bead = _side_chain_hot_vector[1] if len(_side_chain_hot_vector) > 1 else False
        
        # Nuevos atributos para la interfase
        self._axis = axis
        self._weight_interface = weight_interface
        self._relative_displacement_from_interface = relative_displacement_from_interface

    def build_qubit_op(self) -> SparsePauliOp:
        """
        Builds a qubit operator for a total Hamiltonian for a protein folding problem. It includes
        8 terms responsible for chirality, geometry, and nearest neighbors interactions.
        Returns:
            SparsePauliOp: A total Hamiltonian for the protein folding problem.
        Raises:
            InvalidSizeException: if chains of invalid/incompatible sizes provided.
            InvalidSideChainException: if side chains on forbidden indices provided.
        """
        side_chain = self._peptide.get_side_chain_hot_vector()
        main_chain_len = len(self._peptide.get_main_chain)

        if len(side_chain) != main_chain_len:
            raise InvalidSizeException("side_chain_lens size not equal main_chain_len")
        if side_chain[0] == 1 or side_chain[-1] == 1:
            raise InvalidSideChainException(
                "First and last main beads are not allowed to have a side chain. Nonempty "
                "residue provided for an invalid side chain."
            )

        # Calcula el número total de qubits necesarios
        num_qubits = 4 * (main_chain_len - 1) ** 2
        full_id = _build_full_identity(num_qubits)  # Retorna un SparsePauliOp de identidad completa
        short_id = _build_full_identity(2 * 2 * (main_chain_len - 1))  # Retorna un SparsePauliOp de identidad completa

        # Inicializa los términos del Hamiltoniano
        h_chiral = self._create_h_chiral(num_qubits)
        if h_chiral.size: h_chiral = full_id ^ h_chiral

        h_back = self._create_h_back(num_qubits)
        if h_back.size: h_back = full_id ^ h_back

        # Inicializa h_scsc y h_bbbb dependiendo de penalty_1
        if self._penalty_parameters.penalty_1:
            h_scsc = self._create_h_scsc(num_qubits)
            h_bbbb = self._create_h_bbbb(num_qubits)

        # h_short
        h_short = self._create_h_short(num_qubits)
        if h_short.size: h_short = full_id ^ h_short

        # h_bbsc, h_scbb
        if self._penalty_parameters.penalty_1:
            h_bbsc, h_scbb = self._create_h_bbsc_and_h_scbb(num_qubits)
            
        # Inicializa h_scsc, h_bbbb, h_bbsc, h_scbb, since we do know h_back has the right dimensions
        if h_scsc == 0: h_scsc = 0.0 * h_back
        if h_bbbb == 0: h_bbbb = 0.0 * h_back
        if h_bbsc == 0: h_bbsc = 0.0 * h_back
        if h_scbb == 0: h_scbb = 0.0 * h_back

        # h_interface
        h_interface = self._create_h_interface(num_qubits)
        if h_interface.size: h_interface = full_id ^ h_interface
        
        #print(f"h_chiral: {h_chiral.simplify()}, num_qubits: {h_chiral.num_qubits}\n\n")
        #print(f"h_back: {h_back.simplify()}, num_qubits: {h_back.num_qubits}\n\n")
        #print(f"h_short: {h_short.simplify()}, num_qubits: {h_short.num_qubits}\n\n")
        #print(f"h_bbbb: {h_bbbb.simplify()}, num_qubits: {h_bbbb.num_qubits}\n\n")
        #print(f"h_bbsc: {h_bbsc.simplify()}, num_qubits: {h_bbsc.num_qubits}\n\n")
        #print(f"h_scsc: {h_scsc.simplify()}, num_qubits: {h_scsc.num_qubits}\n\n")
        #print(f"h_scbb: {h_scbb.simplify()}, num_qubits: {h_scbb.num_qubits}\n\n")
        #print(f"h_interface: {h_interface.simplify()}, num_qubits: {h_interface.num_qubits}\n\n")
        
        # Suma todos los términos del Hamiltoniano
        h_total = h_chiral + h_back + h_short + h_bbbb + h_bbsc + h_scbb + h_scsc + h_interface
        #print(f"h_total: {h_total.simplify()}, num_qubits: {h_total.num_qubits}\n\n")

        return h_total.simplify()

    def _create_turn_operators(self, lower_bead: BaseBead, upper_bead: BaseBead) -> SparsePauliOp:
        """
        Creates a qubit operator for consecutive turns.
        Args:
            lower_bead: A bead with a smaller index in the chain.
            upper_bead: A bead with a bigger index in the chain.
        Returns:
            SparsePauliOp: A qubit operator for consecutive turns.
        """
        lower_indicators = lower_bead.indicator_functions
        upper_indicators = upper_bead.indicator_functions
        #print(f"lower_indicators: {lower_indicators}")
        #print(f"upper_indicators: {upper_indicators}")

        # Combina los operadores de indicadores
        term = (lower_indicators[0].dot(upper_indicators[0]) +
                lower_indicators[1].dot(upper_indicators[1]) +
                lower_indicators[2].dot(upper_indicators[2]) +
                lower_indicators[3].dot(upper_indicators[3]))
        
        #print (f"term antes de fix_qubits: {term}")

        # Aplica la corrección de qubits
        turns_operator = _fix_qubits(term, self._has_side_chain_second_bead)
        
        #print (f"\n\nturns_operator DESPUES de fix_qubits: \n {turns_operator}\n\n")
        
        return turns_operator

    def _create_h_back(self, num_qubits: int) -> SparsePauliOp:
        """
        Creates Hamiltonian that imposes the geometrical constraint wherein consecutive turns along
        the same axis are penalized by a factor, penalty_back. Note that the first two turns are
        omitted (fixed in optimization) due to symmetry degeneracy.

        Args:
            num_qubits: Total number of qubits.

        Returns:
            SparsePauliOp: Contribution to Hamiltonian that penalizes consecutive turns along the same axis.
        """
        main_chain = self._peptide.get_main_chain
        #print(f"main_chain: {main_chain}")
        penalty_back = self._penalty_parameters.penalty_back
        #print(f"penalty_back: {penalty_back}")
        #h_back = SparsePauliOp.from_list([('I' * num_qubits, 0)]) #SparsePauliOp.from_list([], num_qubits=num_qubits)  # Inicializa como operador vacío
        
        turn_op = self._create_turn_operators(main_chain[0], main_chain[1])
        h_back = SparsePauliOp.from_list([('I' * turn_op.num_qubits, 0)]) # Inicializa como operador vacío de la dimensión correcta
        h_back += penalty_back * turn_op
        #print(f"turn_op: {turn_op.simplify()} num_qubits: {turn_op.num_qubits}")
        #print(f"h_back: {h_back.simplify()}, num_qubits: {h_back.num_qubits}\n\n")
        
        for i in range(1, len(main_chain) - 2):
            turn_op = self._create_turn_operators(main_chain[i], main_chain[i + 1])
            #print(f"turn_op: {turn_op.simplify()} num_qubits: {turn_op.num_qubits}")
            #print(f"h_back: {h_back.simplify()}, num_qubits: {h_back.num_qubits}\n\n")

            h_back += penalty_back * turn_op

        h_back = _fix_qubits(h_back, self._has_side_chain_second_bead)
        #print(f"h_back: {h_back.simplify()}, num_qubits: {h_back.num_qubits}\n\n")
        return h_back

    def _create_h_chiral(self, num_qubits: int) -> SparsePauliOp:
        """
        Creates a penalty/constrain term to the total Hamiltonian that imposes that all the positions
        of all side chain beads impose the right chirality. Note that the position of the side chain
        bead at a location (i) is determined by the turn indicators at i - 1 and i. In the absence
        of side chains, this function returns zero.
        Args:
            num_qubits: Total number of qubits.
        Returns:
            SparsePauliOp: Hamiltonian term that imposes the right chirality.
        """
        main_chain = self._peptide.get_main_chain
        main_chain_len = len(main_chain)
        #h_chiral = SparsePauliOp.from_list([('I' * num_qubits, 0)]) #SparsePauliOp.from_list([], num_qubits=num_qubits)  # Inicializa como operador vacío
        # 2 qubits por giro, 2 registros de qubits principales y laterales
        full_id = _build_full_identity(2 * 2 * (main_chain_len - 1))
        h_chiral = 0.0 * full_id

        for i in range(1, len(main_chain)):
            upper_main_bead = main_chain[i]
            if upper_main_bead.side_chain is None:
                continue
            upper_side_bead = upper_main_bead.side_chain[0]
            lower_main_bead = main_chain[i - 1]

            lower_indicators = lower_main_bead.indicator_functions
            upper_indicators = upper_main_bead.indicator_functions
            upper_side_indicators = upper_side_bead.indicator_functions

            turn_coeff = int((1 - (-1) ** i) / 2)

            # Construye los términos quirales
            term1 = self._build_chiral_term(
                full_id,
                lower_indicators[1], lower_indicators[2], lower_indicators[3],
                turn_coeff,
                upper_indicators[1], upper_indicators[2], upper_indicators[3],
                upper_side_indicators[0]
            )
            term2 = self._build_chiral_term(
                full_id,
                lower_indicators[0], lower_indicators[3], lower_indicators[2],
                turn_coeff,
                upper_indicators[0], upper_indicators[3], upper_indicators[2],
                upper_side_indicators[1]
            )
            term3 = self._build_chiral_term(
                full_id,
                lower_indicators[0], lower_indicators[1], lower_indicators[3],
                turn_coeff,
                upper_indicators[0], upper_indicators[1], upper_indicators[3],
                upper_side_indicators[2]
            )
            term4 = self._build_chiral_term(
                full_id,
                lower_indicators[0], lower_indicators[2], lower_indicators[1],
                turn_coeff,
                upper_indicators[0], upper_indicators[2], upper_indicators[1],
                upper_side_indicators[3]
            )

            h_chiral += term1 + term2 + term3 + term4
            h_chiral = _fix_qubits(h_chiral, self._has_side_chain_second_bead)

        return h_chiral

    def _build_chiral_term(self, full_id: SparsePauliOp,
                           lower_main_bead_indic_b: SparsePauliOp,
                           lower_main_bead_indic_c: SparsePauliOp,
                           lower_main_bead_indic_d: SparsePauliOp,
                           turn_coeff: int,
                           upper_main_bead_indic_b: SparsePauliOp,
                           upper_main_bead_indic_c: SparsePauliOp,
                           upper_main_bead_indic_d: SparsePauliOp,
                           upper_side_bead_indic_a: SparsePauliOp) -> SparsePauliOp:
        """
        Builds the chiral term for a specific bead.
        Args:
            full_id: Full identity operator.
            lower_main_bead_indic_b: Indicator function b for lower main bead.
            lower_main_bead_indic_c: Indicator function c for lower main bead.
            lower_main_bead_indic_d: Indicator function d for lower main bead.
            turn_coeff: Turn coefficient based on position.
            upper_main_bead_indic_b: Indicator function b for upper main bead.
            upper_main_bead_indic_c: Indicator function c for upper main bead.
            upper_main_bead_indic_d: Indicator function d for upper main bead.
            upper_side_bead_indic_a: Indicator function a for upper side bead.
        Returns:
            SparsePauliOp: The chiral term operator.
        """
        lmbb=lower_main_bead_indic_b; lmbc=lower_main_bead_indic_c; lmbd=lower_main_bead_indic_d
        umbb=upper_main_bead_indic_b; umbc=upper_main_bead_indic_c; umbd=upper_main_bead_indic_d
        
        return (self._penalty_parameters.penalty_chiral * (full_id - upper_side_bead_indic_a)
            @ ((1 - turn_coeff) * (lmbb @ umbc + lmbc @ umbd + lmbd @ umbb)
                  + turn_coeff  * (lmbc @ umbb + lmbd @ umbc + lmbb @ umbd)))
        
    def _create_h_bbbb(self, num_qubits: int) -> SparsePauliOp:
        """
        Creates Hamiltonian term corresponding to a 1st neighbor interaction between main/backbone (BB) beads.
        Args:
            num_qubits: Total number of qubits.
        Returns:
            SparsePauliOp: Hamiltonian term corresponding to a 1st neighbor interaction between main/backbone (BB) beads.
        """
        penalty_1 = self._penalty_parameters.penalty_1
        main_chain_len = len(self._peptide.get_main_chain)
        #h_bbbb = SparsePauliOp.from_list([('I' * num_qubits, 0)]) #SparsePauliOp.from_list([], num_qubits=num_qubits)  # Inicializa como operador vacío
        #full_id = _build_full_identity(2 * 2 * (main_chain_len - 1))

        h_bbbb = 0

        for i in range(1, main_chain_len - 3):
            for j in range(i + 5, main_chain_len + 1):
                if (j - i) % 2 == 0:
                    continue
                # Asumiendo que lower_main_upper_main[i][j] es un SparsePauliOp
                contact_op = self._contact_map.lower_main_upper_main[i][j]
                #print(f"contact_op: {contact_op.simplify()}")
                distance_op = self._distance_map.first_neighbor(self._peptide, i, 0, j, 0, penalty_1, self._pair_energies)
                #print(f"distance_op: {distance_op.simplify()}\n\n")
                
                if h_bbbb == 0: h_bbbb = contact_op ^ distance_op
                else: h_bbbb += contact_op ^ distance_op

                # Términos de distancia adicionales con manejo de excepciones
                try:
                    distance_op = self._distance_map.second_neighbor(self._peptide, i - 1, 0, j, 0, penalty_1, self._pair_energies)
                    h_bbbb += contact_op ^ distance_op
                except (IndexError, KeyError):
                    pass
                try:
                    distance_op = self._distance_map.second_neighbor(self._peptide, i + 1, 0, j, 0, penalty_1, self._pair_energies)
                    h_bbbb += contact_op ^ distance_op
                except (IndexError, KeyError):
                    pass
                try:
                    distance_op = self._distance_map.second_neighbor(self._peptide, i, 0, j - 1, 0, penalty_1, self._pair_energies)
                    h_bbbb += contact_op ^ distance_op
                except (IndexError, KeyError):
                    pass
                try:
                    distance_op = self._distance_map.second_neighbor(self._peptide, i, 0, j + 1, 0, penalty_1, self._pair_energies)
                    h_bbbb += contact_op ^ distance_op
                except (IndexError, KeyError):
                    pass

                # Aplica la corrección de qubits después de cada interacción
                h_bbbb = _fix_qubits(h_bbbb, self._has_side_chain_second_bead)
        #print(f"h_bbbb: {h_bbbb.simplify()}, num_qubits: {h_bbbb.num_qubits}\n\n")

        return h_bbbb

    def _create_h_interface(self, num_qubits: int) -> SparsePauliOp: 
        """
        Los desplazamientos son discretos y sólo pueden tomar valores de 0, 1 o -1 entre 2 beads consecutivas.
        La diferencia con respecto al primer bead nos da el signo con respecto a la interfase.
        Args:
            num_qubits: Total number of qubits.
        Returns:
            SparsePauliOp: Hamiltonian term for interface interactions.
        """
        
        main_chain_residue_sequence = self._peptide.get_main_chain.main_chain_residue_sequence
        #print("main_chain_residue_sequence: ", main_chain_residue_sequence)
        
        hydrophobicity_value = []
        for aa in main_chain_residue_sequence:
            for key, value in FaucherePliskaInteractions.items():
                if value['label'] == aa:
                    hydrophobicity_value.append(value['hydrophobicity'])
        
        #print("hydrophobicity_value: ", hydrophobicity_value)
        
        #print("self._axis: ", self._axis)
        #print("self._weight_interface: ", self._weight_interface)
        #print("self._relative_displacement_from_interface: ", self._relative_displacement_from_interface)
        
        main_chain_len = len(self._peptide.get_main_chain)    
        h_interface = SparsePauliOp.from_list([('I' * num_qubits, 0)])
        full_id = _build_full_identity(2 * 2 * (main_chain_len - 1))
        h_interface = 0.0 * full_id
        
        displacementMap = self._distance_map.displacement_map(self._axis)  # Usar el axis del objeto
        beads = list(displacementMap.keys())
        if not beads: return h_interface

        first_bead = beads[0]
        #print("\n\nAhora vamos a imprimir el displacementMap que se utilizará en el hamiltoniano de la interfase: \n\n")
        
        full_id = _build_full_identity(2 * 2 * (main_chain_len - 1))
        one = full_id
        #print("one = ", one)
        
        TotalDisplacements = []
        i = -1
        for bead, value in displacementMap[first_bead].items(): 
            if bead: 
                i += 1
                TotalDisplacements.append((value.simplify()-self._relative_displacement_from_interface*one).simplify())
                TD = TotalDisplacements[i]
                TD2 = (TD@TD).simplify()
                TD3 = (TD@TD2).simplify()
                TD4 = (TD2@TD2).simplify()
                TD5 = (TD2@TD3).simplify()
                TD7 = (TD3@TD4).simplify()
                signDisplacement = (0.9635*TD - 0.0364*TD3 + 0.00059*TD5 - 0.00000312*TD7).simplify()
                #print("\n\n\nDisplacement between ", first_bead, " and ", bead, " along axis ", self._axis, ": \n", TotalDisplacements[i])
                h_interface += self._weight_interface * signDisplacement * hydrophobicity_value[i]

        # Muestra energías
        for i in range(1, main_chain_len):
            for j in range(i + 1, main_chain_len + 1):
                energy = self._pair_energies[i][0][j][0]
                print("bead ", i, " and bead ", j, " have energy ", energy)

        h_interface = _fix_qubits(h_interface, self._has_side_chain_second_bead)
    
        return h_interface

    def _create_h_bbsc_and_h_scbb(self, num_qubits: int) -> Tuple[SparsePauliOp, SparsePauliOp]:
        """
        Creates Hamiltonian term corresponding to 1st neighbor interaction between main/backbone (BB) and side chain (SC) beads. 
        In the absence of side chains, this function returns a tuple of identity operators with 0 coefficients.
        Args:
            num_qubits: Total number of qubits.
        Returns:
            Tuple of SparsePauliOp: Hamiltonian terms consisting of backbone and side chain interactions.
        """
        penalty_1 = self._penalty_parameters.penalty_1
        main_chain_len = len(self._peptide.get_main_chain)

        h_bbsc = 0
        h_scbb = 0
        
        side_chain = self._peptide.get_side_chain_hot_vector()
        for i in range(1, main_chain_len - 3):
            for j in range(i + 4, main_chain_len + 1):
                if (j - i) % 2 == 1:
                    continue

                if side_chain[j - 1] == 1:
                    contact_op = self._contact_map.lower_main_upper_side[i][j]
                    distance_op1 = self._distance_map.first_neighbor(self._peptide, i, 0, j, 1, penalty_1, self._pair_energies)
                    distance_op2 = self._distance_map.second_neighbor(self._peptide, i, 0, j, 0, penalty_1, self._pair_energies)
                    
                    if h_bbsc == 0: h_bbsc = contact_op ^ (distance_op1 + distance_op2)
                    else: h_bbsc += contact_op ^ (distance_op1 + distance_op2)

                    try:
                        distance_op = self._distance_map.first_neighbor(self._peptide, i, 1, j, 1, penalty_1, self._pair_energies)
                        h_bbsc += self._contact_map.lower_side_upper_side[i][j] ^ distance_op
                    except (IndexError, KeyError, TypeError):
                        pass
                    try:
                        distance_op = self._distance_map.second_neighbor(self._peptide, i + 1, 0, j, 1, penalty_1, self._pair_energies)
                        h_bbsc += self._contact_map.lower_main_upper_side[i][j] ^ distance_op
                    except (IndexError, KeyError, TypeError):
                        pass
                    try:
                        distance_op = self._distance_map.second_neighbor(self._peptide, i - 1, 0, j, 1, penalty_1, self._pair_energies)
                        h_bbsc += self._contact_map.lower_main_upper_side[i][j] ^ distance_op
                    except (IndexError, KeyError, TypeError):
                        pass

                if side_chain[i - 1] == 1:
                    contact_op = self._contact_map.lower_side_upper_main[i][j]
                    distance_op1 = self._distance_map.first_neighbor(self._peptide, i, 1, j, 0, penalty_1, self._pair_energies)
                    distance_op2 = self._distance_map.second_neighbor(self._peptide, i, 0, j, 0, penalty_1, self._pair_energies)
                    
                    if h_scbb == 0: h_scbb = contact_op ^ (distance_op1 + distance_op2)
                    else: h_scbb += contact_op ^ (distance_op1 + distance_op2)

                    try:
                        distance_op = self._distance_map.second_neighbor(self._peptide, i, 1, j, 1, penalty_1, self._pair_energies)
                        h_scbb += self._contact_map.lower_side_upper_main[i][j] ^ distance_op
                    except (IndexError, KeyError, TypeError):
                        pass
                    try:
                        distance_op = self._distance_map.second_neighbor(self._peptide, i, 1, j + 1, 0, penalty_1, self._pair_energies)
                        h_scbb += self._contact_map.lower_side_upper_main[i][j] ^ distance_op
                    except (IndexError, KeyError, TypeError):
                        pass
                    try:
                        distance_op = self._distance_map.second_neighbor(self._peptide, i, 1, j - 1, 0, penalty_1, self._pair_energies)
                        h_scbb += self._contact_map.lower_side_upper_main[i][j] ^ distance_op
                    except (IndexError, KeyError, TypeError):
                        pass

        h_bbsc = _fix_qubits(h_bbsc, self._has_side_chain_second_bead)
        h_scbb = _fix_qubits(h_scbb, self._has_side_chain_second_bead)
        return h_bbsc, h_scbb

    def _create_h_scsc(self, num_qubits: int) -> SparsePauliOp:
        """
        Creates Hamiltonian term corresponding to 1st neighbor interaction between side chain (SC) beads. 
        In the absence of side chains, this function returns an operator with 0 coefficient.
        Args:
            num_qubits: Total number of qubits.
        Returns:
            SparsePauliOp: Hamiltonian term consisting of side chain pairwise interactions.
        """
        penalty_1 = self._penalty_parameters.penalty_1
        #h_scsc = SparsePauliOp.from_list([('I' * num_qubits, 0)]) #SparsePauliOp.from_list([], num_qubits=num_qubits)  # Inicializa como operador vacío
        
        main_chain_len = len(self._peptide.get_main_chain)
        side_chain = self._peptide.get_side_chain_hot_vector()
        
        h_scsc = 0
        
        for i in range(1, main_chain_len - 3):
            for j in range(i + 5, main_chain_len + 1):
                if (j - i) % 2 == 0:
                    continue
                if side_chain[i - 1] == 0 or side_chain[j - 1] == 0:
                    continue
                contact_op = self._contact_map.lower_side_upper_side[i][j]
                distance_op1 = self._distance_map.first_neighbor(self._peptide, i, 1, j, 1, penalty_1, self._pair_energies)
                distance_op2 = self._distance_map.second_neighbor(self._peptide, i, 1, j, 0, penalty_1, self._pair_energies)
                distance_op3 = self._distance_map.second_neighbor(self._peptide, i, 0, j, 1, penalty_1, self._pair_energies)
                if h_scsc == 0:
                    h_scsc = contact_op ^ (distance_op1 + distance_op2 + distance_op3)
                else:
                    h_scsc += contact_op ^ (distance_op1 + distance_op2 + distance_op3)

        h_scsc = _fix_qubits(h_scsc, self._has_side_chain_second_bead)
        return h_scsc

    def _create_h_short(self, num_qubits: int) -> SparsePauliOp:
        """
        Creates Hamiltonian constituting interactions between beads that are no more than 4 beads apart. 
        If no side chains are present, this function returns an operator with 0 coefficient.
        Args:
            num_qubits: Total number of qubits.
        Returns:
            SparsePauliOp: Contribution to energetic Hamiltonian for interactions between beads that 
                           are no more than 4 beads apart.
        """
        main_chain_len = len(self._peptide.get_main_chain)
        side_chain = self._peptide.get_side_chain_hot_vector()
        #h_short = SparsePauliOp.from_list([('I' * num_qubits, 0)]) #SparsePauliOp.from_list([], num_qubits=num_qubits)  # Inicializa como operador vacío
        full_id = _build_full_identity(2 * 2 * (main_chain_len - 1))
        h_short = 0.0 * full_id
        
        for i in range(1, main_chain_len - 2):
            # Comprueba interacciones entre beads que no están a más de 4 beads de distancia
            if side_chain[i - 1] == 1 and side_chain[i + 2] == 1:
                op1 = self._create_turn_operators(self._peptide.get_main_chain[i + 1], self._peptide.get_main_chain[i - 1].side_chain[0])
                op2 = self._create_turn_operators(self._peptide.get_main_chain[i - 1], self._peptide.get_main_chain[i + 2].side_chain[0])
                coeff = float(self._pair_energies[i][1][i + 3][1] + 0.1 * (self._pair_energies[i][1][i + 3][0] + self._pair_energies[i][0][i + 3][1]))
                composed = op1 ^ op2
                h_short += coeff * composed  # No necesita reducir

        h_short = _fix_qubits(h_short, self._has_side_chain_second_bead)
        return h_short

