import json
import numpy as np
from mayavi import mlab
import argparse

# Diccionario de colores para los aminoácidos
aminoacid_info = {
    'R': {'code': 'Arg', 'color': 'blue', 'color_mayavi': (0, 0, 1)},
    'C': {'code': 'Cys', 'color': 'orange', 'color_mayavi': (1, 0.5, 0)},
    'G': {'code': 'Gly', 'color': 'cyan', 'color_mayavi': (0, 1, 1)},
    'E': {'code': 'Glu', 'color': 'magenta', 'color_mayavi': (1, 0, 1)},
    'F': {'code': 'Phe', 'color': 'yellow', 'color_mayavi': (1, 1, 0)},
    'D': {'code': 'Asp', 'color': 'red', 'color_mayavi': (1, 0, 0)},
    'H': {'code': 'His', 'color': 'pink', 'color_mayavi': (1, 0.5, 0.5)},
    'I': {'code': 'Ile', 'color': 'brown', 'color_mayavi': (0.5, 0.25, 0)},
    'K': {'code': 'Lys', 'color': 'teal', 'color_mayavi': (0, 0.5, 0.5)},
    'L': {'code': 'Leu', 'color': 'olive', 'color_mayavi': (0, 0.5, 0)},
    'M': {'code': 'Met', 'color': 'gold', 'color_mayavi': (1, 0.8, 0)},
    'N': {'code': 'Asn', 'color': 'lime', 'color_mayavi': (0, 1, 0)},
    'P': {'code': 'Pro', 'color': 'cyan', 'color_mayavi': (0, 1, 1)},
    'Q': {'code': 'Gln', 'color': 'purple', 'color_mayavi': (0.5, 0, 0.5)},
    'A': {'code': 'Ala', 'color': 'pink', 'color_mayavi': (1, 0.5, 0.5)},
    'W': {'code': 'Trp', 'color': 'brown', 'color_mayavi': (0.5, 0.25, 0)},
    'T': {'code': 'Thr', 'color': 'teal', 'color_mayavi': (0, 0.5, 0.5)},
    'V': {'code': 'Val', 'color': 'olive', 'color_mayavi': (0, 0.5, 0)},
    'S': {'code': 'Ser', 'color': 'gold', 'color_mayavi': (1, 0.8, 0)},
    'X': {'code': 'Unknown', 'color': 'brown', 'color_mayavi': (0.5, 0.25, 0)},
    'Y': {'code': 'Tyr', 'color': 'navy', 'color_mayavi': (0, 0, 0.5)}
}

# Función para cargar los datos desde el JSON
def load_representation_from_json(json_filename):
    with open(json_filename, "r") as file:
        data = json.load(file)
    
    side_chain_data = data['side_chains']
    main_chain_data = data['main_chain']
    plane_data = data['plane']
    
    x_plane_points = np.array(plane_data['x'])
    y_plane_points = np.array(plane_data['y'])
    z_plane_points = np.array(plane_data['z'])
    
    return side_chain_data, main_chain_data, x_plane_points, y_plane_points, z_plane_points

# Función para visualizar la estructura
def plot_from_json(json_filename):
    # Cargar los datos desde el JSON
    side_chain_data, main_chain_data, x_plane_points, y_plane_points, z_plane_points = load_representation_from_json(json_filename)

    # Crear la figura de Mayavi
    mlab.figure(bgcolor=(1, 1, 1))

    # Visualizar las cadenas laterales
    for chain in side_chain_data:
        x_side, y_side, z_side = chain["position"]
        mlab.points3d(x_side, y_side, z_side, mode='sphere', color=(0, 1, 0), scale_factor=0.1, opacity=1)

    # Visualizar la cadena principal
    for chain in main_chain_data:
        x_main, y_main, z_main = chain["position"]
        aminoacid = chain["aminoacid"]
        
        # Obtener el color del aminoácido
        color = aminoacid_info[aminoacid]['color_mayavi']
        
        # Mostrar los puntos de la cadena principal
        mlab.points3d(x_main, y_main, z_main, mode='sphere', color=color, scale_factor=0.1, opacity=1)
        
        # Conectar los puntos con líneas (para formar la cadena)
        if main_chain_data.index(chain) < len(main_chain_data) - 1:
            next_chain = main_chain_data[main_chain_data.index(chain) + 1]
            mlab.plot3d([x_main, next_chain["position"][0]], 
                        [y_main, next_chain["position"][1]], 
                        [z_main, next_chain["position"][2]], 
                        tube_radius=0.03, color=color, opacity=0.6)

    # Dibujar el plano si C != 0
    if len(x_plane_points) > 0 and len(y_plane_points) > 0 and len(z_plane_points) > 0:
        mlab.mesh(x_plane_points, y_plane_points, z_plane_points, color=(0.5, 0.0, 0.5), opacity=0.7)

    # Mostrar la visualización interactiva
    mlab.show()

# Función para configurar el parser de argumentos
def main():
    parser = argparse.ArgumentParser(description="Visualización 3D de proteínas usando Mayavi.")
    parser.add_argument("json_file", type=str, help="Archivo JSON con los datos de la proteína.")
    args = parser.parse_args()
    
    # Llamar a la función de visualización pasando el archivo JSON como argumento
    plot_from_json(args.json_file)

# Ejecutar el parser
if __name__ == "__main__":
    main()

