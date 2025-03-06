# TFG_LUCIA_QUANTUM
Initial Code for Protein Folding
# Para empezar
En tu github haces un fork del mío: https://github.com/DanielCondeTorres/TFG_LUCIA_QUANTUM

Luego en una terminal donde vayas a trabajar, probablemente en $LUSTRE harias:

```
git clone https://github.com/.../TFG_LUCIA_QUANTUM 
```

NO NECESARIO (NO VAS A USAR .GIT)

Pon tu correo usc
```
git config user.name "Lucia" && git config user.email "lucia.rodriguez.de@rai.usc.es"
```



# Send a job

```
sbatch -p qpu -c 1 -t 00:30:00 --mem=128G QrunQMIPbackend.sh
```

Entrar en QrunQMIPbackend.sh y añadir las flags que necesitamos en la línea:
python -u pf.py --sequence --axis  --weight --displacement --shots
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
