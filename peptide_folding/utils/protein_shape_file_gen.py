# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An auxiliary class that gets the coordinates of aminoacids of a molecule
 in ProteinFoldingResult."""
import os
from typing import Union, List, Optional
import numpy as np
from ..peptide.peptide import Peptide
from ..protein_folding_problem import ProteinFoldingProblem
import json

class ProteinShapeFileGen:
    """This class handles the creation of cartesian coordinates for
    each aminoacid in a protein and generates a .xyz file.
    It is used by :class:`~qiskit_research.protein_folding.ProteinFoldingResult`.
    """

    # Coordinates of the 4 edges of a tetrahedron centered at 0. The vectors are normalized.
    COORDINATES = (1.0 / np.sqrt(3)) * np.array(
        [[-1, 1, 1], [1, 1, -1], [-1, -1, -1], [1, -1, 1]]
    )

    def __init__(
        self,
        main_chain_turns: List[int],
        side_chain_turns: List[Union[None, int]],
        peptide: Peptide,
        #axis: int,
        #weight_interface: float,
        #relative_displacement_from_interface: float,
    ) -> None:
        """
        Args:
            main_chain_turns: A list of integers encoding the turns of the main chain.
            side_chain_turns: A list of integers and None encoding the turns of the main chain
                or None.
            peptide: The peptide we are getting the positions for.

        """

        self._main_chain_turns = main_chain_turns
        self._side_chain_turns = side_chain_turns

        self.main_chain_aminoacid_list = np.array(
            list(peptide.get_main_chain.main_chain_residue_sequence)
        )
        self.side_chain_aminoacid_list = np.array(
            [
                aminoacid.residue_sequence[0] if (aminoacid is not None) else None
                for aminoacid in peptide.get_side_chains()
            ]
        )
        self._main_positions = self.generate_main_positions()
        self._side_positions = self.generate_side_positions()
        # Leer el archivo JSON
        self._tetrahedron_vectors=self.generate_tetrahedron_vectors()
        self.plane_equation,self._data_plane=self.plane_equation()

    def generate_side_positions(self) -> List[Optional[np.ndarray]]:
        """
        Generates the positions of the side chain.
        Returns:
            A list of arrays with the cartesian coordinates of the side chain.
        """
        side_positions: List[Optional[np.ndarray]] = []
        counter = 1
        for mainpos, sideturn in zip(self.main_positions, self._side_chain_turns):
            if sideturn is None:
                side_positions.append(None)
            else:
                side_positions.append(
                    mainpos + (-1) ** counter * self.COORDINATES[sideturn]
                )

            counter += 1
        return side_positions

    @property
    def side_positions(self) -> List[Optional[np.ndarray]]:
        """
        Returns the xyz position for each side chain element.

        Returns:
            A list with the position of the side chain of each bead in the main chain in order.
            None in the i-th position of the list corresponds to no side chain at that
            position of the main chain.

        """

        return self._side_positions

    def generate_main_positions(self) -> np.ndarray:
        """
        Generates the positions of the main chain.

        Returns:
            An array with the cartesian coordinates of the main chain.
        """
        length_turns = len(self._main_chain_turns)
        relative_positions = np.zeros((length_turns + 1, 3), dtype=float)
       # _axis = ProteinFoldingProblem().axis
      #  print('AXIS',_axis)
        for i in range(length_turns):
            relative_positions[i + 1] = (-1) ** i * self.COORDINATES[
                self._main_chain_turns[i]
            ]

        return relative_positions.cumsum(axis=0)

    @property
    def main_positions(self) -> np.ndarray:
        """
        Returns an array with the cartesian coordinates of each aminoacid in the main chain.
        The first time called it generates the coordinates.

        Returns:
            An array with the cartesian coordinates of each aminoacid in the main chain.

        """

        return self._main_positions
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def generate_tetrahedron_vectors(self) -> np.ndarray:

        '''
        Purpose: This function appears to generate vectors representing the vertices of a tetrahedron based on some global constant self.COORDINATES.
        '''

        #The variable length_axis is set to 4, which corresponds to the number of vertices in a tetrahedron.
        length_axis = 4
        
        # It creates an empty list relative_positions_axis to store the vectors.
        relative_positions_axis = []
        
        #  A loop retrieves the vectors (or coordinates) from self.COORDINATES for each vertex and appends them to the list.
        for i in range(length_axis):
            relative_positions_axis.append(  self.COORDINATES[i] )
        
        #  Returns the list of vectors. 
        return relative_positions_axis
    
    
    def plane_equation(self):
        '''
        Purpose: Given certain parameters, this function computes the equation of a plane and returns both the coefficients of the plane and an array of coordinates that lie on the plane.
        Plane equation: Ax + By + Cx + D = 0
        '''
        with open("config.json", "r") as file:
            loaded_data = json.load(file)
        # Acceder a las propiedades
        weight_interface = loaded_data.get("weight_interface")
        relative_displacement_from_interface = loaded_data.get("relative_displacement_from_interface")
        # Obtain the selected axis
        axis =  loaded_data.get("axis")
        # Retrieve the perpendicular vector to the plane based on the selected axis
        normal_vector_to_the_plane = self._tetrahedron_vectors[axis]
        # Check if the weight of the interface is not zero, but, because there is no plane to represent
        if weight_interface != 0:
            try:
                print('Calulating plane equation ...') 
                # Extract a reference point on the plane using
                x0, y0, z0 = self._main_positions[0]+self._main_positions[1]*relative_displacement_from_interface
            
                # Extract the components of the perpendicular vector
                Nx, Ny, Nz = normal_vector_to_the_plane
            
                # Calculate the value of DD, which is part of the standard plane equation Ax+By+Cz+D=0Ax+By+Cz+D=0.
                D = -Nx * x0 - Ny * y0 - Nz * z0
            
                # Compute a string representation of the plane equation.
                ecuacion = f"{Nx}*(x - {x0}) + {Ny}*(y - {y0}) + {Nz}*(z - {z0}) + {D} = 0"
                ABCD=[Nx,Ny,Nz,D]
                A,B,C,D = ABCD 
                # Prepare a grid of x and y values, then calculate the corresponding z values for each (x, y) pair to obtain a set of points that lie on the plane.
                # Convert the calculated coordinates into a format that includes an identifier (like 'B') followed by the x, y, z coordinates, and add them to a list
                # Made in case someone wants to save the plan values in a future file, e.g. .xyz. SEE: in save_xyz_file, EXAMPLE*
                num_points=100
                x = np.linspace(-2, 2, num_points)
                y = np.linspace(-2, 2, num_points)
                x, y = np.meshgrid(x, y)
                # Calcular z a partir de la ecuación del plano
                z = (-A * x - B * y - D) / C
                # changing the plane by points
                changing_the_plane_by_points = []
                for i in range(num_points):
                    for j in range(num_points):
                        x_coord = x[i][j]
                        y_coord = y[i][j]
                        z_coord = z[i][j]
                        changing_the_plane_by_points.append(['Z', str(x_coord), str(y_coord), str(z_coord)])
                changing_the_plane_by_points = np.array(changing_the_plane_by_points)
            except ValueError:
                A = 0; B = 0; C = 0; D = 0
                ABCD =[A,B,C,D]
                changing_the_plane_by_points = np.array([0]*4)

        # No plane to represent
        else:
            print('Not plane')
            A = 0; B = 0; C = 0; D = 0
            ABCD =[A,B,C,D]
            changing_the_plane_by_points = np.array([0]*4)
        return ABCD, changing_the_plane_by_points
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def save_xyz_file(
        self, name: str, path: str = "", comment: str = "", replace: bool = False
    ) -> None:
        """
        Saves the data as an .xyz file.
        For more information about .xyz files see:
        https://en.wikipedia.org/wiki/XYZ_file_format.

        Args:
            name: The file will be called `"name".xyz`. Can overwrite files.
            path: Path under which the file will be saved. If no path is specified the file will
                be saved in the current working directory.
            comment: Comment to be added to the second line of the file. By default the line will
                be left blank.
            replace: If ``True``, the file will be overwritten if it already exists.
        Raises:
            FileExistsError: If the file already exists and ``replace`` is False.
        """
        file_path = os.path.join(path, name + ".xyz")
        if not replace and os.path.exists(file_path):
            raise FileExistsError(f"File {file_path} already exists.")
        data = self.get_xyz_data()
        number_of_particles = data.shape[0]
        header = f"{number_of_particles}\n{comment}"
        np.savetxt(
            fname=file_path,
            header=header,
            X=data,
            delimiter=" ",
            fmt="%s",
            comments="",
        )

    def get_xyz_data(self) -> np.ndarray:
        """
        Returns an array with the symbols of the atoms and their cartesian coordinates.
        Returns:
            An array with the symbols of the atoms and their cartesian coordinates.
        """
        main_data = np.column_stack(
            [self.main_chain_aminoacid_list, self.main_positions]
        )

        # We will discard the None values corresponding to empty side chains.
        side_aminoacid = np.array(self.side_chain_aminoacid_list)
        side_aminoacid = side_aminoacid[side_aminoacid != np.array(None)]
        side_aminoacid = side_aminoacid.astype("<U32")

        side_position = np.array(
            [side_pos for side_pos in self.side_positions if side_pos is not None],
            dtype="<U32",
        )
        side_data = np.column_stack([side_aminoacid, side_position])
        if side_data.size != 0:
            data = np.append(main_data, side_data, axis=0)

        else:
            data = main_data

        return data
