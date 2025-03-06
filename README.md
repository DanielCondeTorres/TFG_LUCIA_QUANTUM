# TFG_LUCIA_QUANTUM
Initial Code for Protein Folding

# Send a job

```
sbatch -p qpu -c 1 -t 00:30:00 --mem=128G QrunQMIPbackend.sh
```
# Make representations:

Download the folder to your local computer:

```
scp -r usccq @qmio.cesga.es:../../Code .    
```

```
pip install mayavi
```

```
pip install configobj
```
```
pip install wxPython
```

```
pip install PyQt5
```
Make the reprepresentation, if you are in code:

```
python Represantion_area/representation.py -json_file Output/representacion.json -output_file rep.png
```