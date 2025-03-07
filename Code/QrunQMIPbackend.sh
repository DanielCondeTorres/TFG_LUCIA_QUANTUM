#!/bin/bash
module load qmio/hpc gcc/12.3.0 qmio-tools/0.2.0-python-3.9.9 qiskit/1.2.4-python-3.9.9
export QMIO_CALIBRATIONS=/opt/cesga/qmio/hpc/calibrations

python pf.py --sequence ... --axis ...  --weight ... --displacement ... --shots ...
mkdir Output
mv *png Output
mv *txt Output
mv slurm* Output
mv *json Output
