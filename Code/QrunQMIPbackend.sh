#!/bin/bash
module load qmio/hpc gcc/12.3.0 qmio-tools/0.2.0-python-3.9.9 qiskit/1.2.4-python-3.9.9
export QMIO_CALIBRATIONS=/opt/cesga/qmio/hpc/calibrations


# Nombre del directorio que desearias
OUTPUT_DIR="MiDirectorio"

# Crear directorio de salida si no existe
mkdir -p "$OUTPUT_DIR"

# Ejecutar script de Python
python pf.py --sequence ... --axis ... --weight ... --displacement ... --shots ...  --iteractions_cobyla ...

mv *png "$OUTPUT_DIR" 
mv *txt "$OUTPUT_DIR" 
mv slurm* "$OUTPUT_DIR" 
mv *json "$OUTPUT_DIR" 
