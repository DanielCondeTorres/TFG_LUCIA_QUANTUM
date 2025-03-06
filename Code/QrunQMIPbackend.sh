#!/bin/bash
module load qmio/hpc  qmio-tools/0.1.1-python-3.9.9 gcc/12.3.0 qiskit/1.0.2-python-3.9.9
export QMIO_CALIBRATIONS=/opt/cesga/qmio/hpc/calibrations

python pf.py --sequence ... --axis ...  --weight ... --displacement ... --shots ...
mkdir Output
mv *png Output
mv *txt Output
mv slurm* Output
mv *json Output
