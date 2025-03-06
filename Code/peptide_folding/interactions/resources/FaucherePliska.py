
#Creamos un diccionario con los valores de hidrofobicidad siguiendo:  Fauchère, J., and Pliska, V. 1983. Hydrophobic parameters {pi} of amino-acid side chains from the partitioning of N-acetyl-amino-acid amides. Eur. J. Med. Chem. 8: 369–375 con enlace: https://www.researchgate.net/publication/246404378_Hydrophobic_parameters_II_of_amino_acid_side-chains_from_the_partitioning_of_N-acetyl-amino_acid_amides 
FaucherePliskaInteractions = {
      'ASP': {'label':'D','hydrophobicity': -0.77,'behavior': 'acidic'},
      'GLU': {'label':'E','hydrophobicity': -0.64,'behavior': 'acidic'},
      'LYS': {'label':'K','hydrophobicity': -0.99,'behavior': 'basic'},
      'ARG': {'label':'R','hydrophobicity': -1.01,'behavior': 'basic'},
      'HIS': {'label':'H','hydrophobicity': 0.13,'behavior': 'basic'},
      'HISH': {'label':'H','hydrophobicity': 0.13,'behavior': 'basic'},
      'GLY': {'label':'G','hydrophobicity': -0.0,'behavior': 'hydrophobic'},
      'ALA': {'label':'A','hydrophobicity': 0.31,'behavior': 'hydrophobic'},
      'VAL': {'label':'V','hydrophobicity': 1.22,'behavior': 'hydrophobic'},
      'LEU': {'label':'L','hydrophobicity': 1.70,'behavior': 'hydrophobic'},
      'ILE': {'label':'I','hydrophobicity': 1.80,'behavior': 'hydrophobic'},
      'PRO': {'label':'P','hydrophobicity': 0.72,'behavior': 'hydrophobic'},
      'PHE': {'label':'F','hydrophobicity': 1.79,'behavior': 'hydrophobic'},
      'MET': {'label':'M','hydrophobicity': 1.23,'behavior': 'hydrophobic'},
      'TRP': {'label':'W','hydrophobicity': 2.25,'behavior': 'hydrophobic'},
      'SER': {'label':'S','hydrophobicity': -0.04,'behavior': 'polar'},
      'THR': {'label':'T','hydrophobicity': 0.26,'behavior': 'polar'},
      'CYS': {'label':'C','hydrophobicity': 1.54,'behavior': 'polar'},
      'TYR': {'label':'Y','hydrophobicity': 0.96,'behavior': 'polar'},
      'ASN': {'label':'N','hydrophobicity': -0.60,'behavior': 'polar'},
      'GLN': {'label':'Q','hydrophobicity': -0.22,'behavior': 'polar'}}
#Esta pag web: https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/midas/hydrophob.html
#incluye varias tablas. por lo que se puede modificar

