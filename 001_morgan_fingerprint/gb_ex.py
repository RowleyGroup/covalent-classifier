import csv
#import matplotlib.pyplot as plt
import sys
import os
import pickle
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#from toxDataset import toxDataset
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import DataStructs
import base64
import fcntl
import rdkit.Chem.MolStandardize as rdMolStandardize

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def convert_to_indices(num, shape):
    indices = []
    for dim in reversed(shape):
        indices.append( int(num % dim) )
        num //= dim
    indices.reverse()
    return indices

def readcsv(fname, keyname):
    df=pd.read_csv(fname, sep='\t')
    data=[]
    fields=keyname.split('_')
    bitsize=int(fields[2])

    for m in df[keyname]:
        bv=DataStructs.cDataStructs.ExplicitBitVect(bitsize)
        bv.FromBase64(m)
        data.append(np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(bv), dtype=np.uint8), bitorder='little'))

    return(data)



r=3 #radii[radii_index]
b=4096 #bits[bits_index]

keyname='MorganFP_' + str(r) + '_' + str(b)

covalent_training_set=readcsv('morgan_data/trainingset_covalent_morgan.csv', keyname)
noncovalent_training_set=readcsv('morgan_data/trainingset_noncovalent_morgan.csv', keyname)

X_data=covalent_training_set + noncovalent_training_set
y_data=[1]*len(covalent_training_set) + [0]*len(noncovalent_training_set)

X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.2, random_state=1)

#clf=LogisticRegression(C=C, class_weight="balanced", solver='saga')
clf=HistGradientBoostingClassifier(max_leaf_nodes=41, max_depth=None,
                                   class_weight="balanced")

clf.fit(X_train, y_train)
pickle.dump(clf, open('gb_model.pkl', 'wb'))
quit()

clf=pickle.load(open('lr_model.pkl', 'rb'))
bits_list=[]
#print(clf.coef_)
lr_coeff=np.array(clf.coef_[0])

for (i, c) in enumerate(clf.coef_[0]):
    if(c>0.5):
        bits_list.append(i)
#        print(c)
#testset=readcsv('lr_examples.csv', keyname)

bi = {}
#fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bi, nBits=1024))
#Draw.DrawMorganBits([(mol, 31, bi), (mol, 33, bi), (mol, 64, bi),
#(mol, 175, bi), (mol, 356, bi), (mol, 389, bi),
#(mol, 698, bi), (mol, 726, bi), (mol, 799, bi), 
#(mol, 849, bi), (mol, 896, bi)], useSVG=True)

drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
drawOptions.prepareMolsBeforeDrawing = False

all_mol=[]
with open('lr_examples.csv', mode='r') as infile:
    test_data=[]
    reader = csv.DictReader(infile, delimiter='\t')
    
    for row in reader:
        if(row['InChI'] is not None):
            test_data.append(row['InChI']  )

    for (i, inchi_orig) in enumerate(test_data):

        m1 = Chem.MolFromInchi(inchi_orig)
        if(m1==None):
            continue
        mol = Chem.AddHs(m1)
        mol=Chem.MolStandardize.rdMolStandardize.CanonicalTautomer(mol)
        all_mol.append(mol)
        
morgan_tuples=[]
for (inchi, mol) in zip(test_data,all_mol):
    info={}
    fp=AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits = 8192, bitInfo=info, useChirality=False)
    np_fp=np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0')
    print(inchi)
    mol_vec=lr_coeff*np_fp
    for i in range(0, len(mol_vec)):
        if(mol_vec[i]>0.5):
            print(str(i) + ' ' + str(mol_vec[i]))
        
    tpls = [(mol ,x, info) for x in fp.GetOnBits()]

    for (i, b) in enumerate(fp):
        if(b==1 and i in bits_list):
            morgan_tuples.append( (mol, i, info) )
print(morgan_tuples)
p=Draw.DrawMorganBits(morgan_tuples, drawOptions=drawOptions)

#fhout=open('lr_morgan.svg', 'w')
#fhout.write(p)
#fhout.close()

p.save('lr_morgan.png')


#(mol, 175, bi), (mol, 356, bi), (mol, 389, bi),                                                                                                                            
#(mol, 698, bi), (mol, 726, bi), (mol, 799, bi),                                                                                                                            
#(mol, 849, bi), (mol, 896, bi)], useSVG=True)     
        
