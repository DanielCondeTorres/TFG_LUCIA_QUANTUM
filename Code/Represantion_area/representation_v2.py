import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Diccionario de colores para los aminoácidos
aminoacid_info = {
    'R': {'code': 'Arg', 'color': 'blue'},
    'C': {'code': 'Cys', 'color': 'orange'},
    'G': {'code': 'Gly', 'color': 'cyan'},
    'E': {'code': 'Glu', 'color': 'magenta'},
    'F': {'code': 'Phe', 'color': 'yellow'},
    'D': {'code': 'Asp', 'color': 'red'},
    'H': {'code': 'His', 'color': 'pink'},
    'I': {'code': 'Ile', 'color': 'brown'},
    'K': {'code': 'Lys', 'color': 'teal'},
    'L': {'code': 'Leu', 'color': 'olive'},
    'M': {'code': 'Met', 'color': 'gold'},
    'N': {'code': 'Asn', 'color': 'lime'},
    'P': {'code': 'Pro', 'color': 'cyan'},
    'Q': {'code': 'Gln', 'color': 'purple'},
    'A': {'code': 'Ala', 'color': 'pink'},
    'W': {'code': 'Trp', 'color': 'brown'},
    'T': {'code': 'Thr', 'color': 'teal'},
    'V': {'code': 'Val', 'color': 'olive'},
    'S': {'code': 'Ser', 'color': 'gold'},
    'X': {'code': 'Unknown', 'color': 'brown'},
    'Y': {'code': 'Tyr', 'color': 'navy'}
}

# Función para cargar los datos desde el JSON
def load_representation_from_json(json_filename, factor=2):
    with open(json_filename, "r") as file:
        data = json.load(file)
    
    side_chain_data = data['side_chains']
    main_chain_data = data['main_chain']
    plane_data = data['plane']
    
    x_plane_points = np.array(plane_data['x'])/factor
    y_plane_points = np.array(plane_data['y'])/factor
    z_plane_points = np.array(plane_data['z'])/factor
    
    return side_chain_data, main_chain_data, x_plane_points, y_plane_points, z_plane_points

# Función para crear la visualización
def plot_from_json(json_filename, output_filename, factor=2):
    # Cargar los datos desde el JSON
    side_chain_data, main_chain_data, x_plane_points, y_plane_points, z_plane_points = load_representation_from_json(json_filename,factor)

    # Crear la figura de Matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # Diccionario para almacenar los aminoácidos presentes
    aminoacids_present = {}

    # Visualizar la cadena principal
    for i, chain in enumerate(main_chain_data):
        x_main, y_main, z_main = chain["position"]
        aminoacid = chain["aminoacid"]
        color = aminoacid_info[aminoacid]['color']

        # Añadir el aminoácido al diccionario si no está presente
        if aminoacid not in aminoacids_present:
            aminoacids_present[aminoacid] = {
                'code': aminoacid_info[aminoacid]['code'],
                'color': color
            }

        # Dibujar el punto
        ax.scatter(x_main, y_main, z_main, color=color, s=2000)

        # Conectar los puntos con líneas
        if i < len(main_chain_data) - 1:
            next_chain = main_chain_data[i + 1]
            x_next, y_next, z_next = next_chain["position"]
            ax.plot([x_main, x_next], [y_main, y_next], [z_main, z_next], color='k', linewidth=2)

    # Visualizar las cadenas laterales
    for chain in side_chain_data:
        x_side, y_side, z_side = chain["position"]
        ax.scatter(x_side, y_side, z_side, color='green', s=100)

    # Dibujar el plano si existe
    if x_plane_points.size > 0 and y_plane_points.size > 0 and z_plane_points.size > 0:
        ax.plot_surface(x_plane_points, y_plane_points, z_plane_points, color='grey', alpha=0.5)

    # Añadir la leyenda solo con los aminoácidos presentes
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=info['color'], markersize=10, label=info['code'])
                       for aa, info in aminoacids_present.items()]
    ax.legend(handles=legend_elements, title="Aminoácidos")

    # Configurar etiquetas y título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualización 3D de la proteína')
    ax.set_axis_off()
    # Guardar la imagen en archivo
    plt.savefig(output_filename, dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualización 3D de proteínas usando Matplotlib y guardado de imagen.")
    parser.add_argument("-json_file", type=str, required=True, help="Archivo JSON con los datos de la proteína.")
    parser.add_argument("-output_file", type=str, default='output.png', help="Nombre del archivo de salida (ej. output.png).")
    parser.add_argument("-factor", default=2,type=float, help="factor to scale the plane representation.")
    args = parser.parse_args()

    plot_from_json(args.json_file, args.output_file, args.factor)

if __name__ == "__main__":
    main()

