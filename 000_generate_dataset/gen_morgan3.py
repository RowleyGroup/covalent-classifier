import csv
import glob
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.Chem import AllChem

import pubchempy as pcp
from openpyxl import load_workbook
import pandas as pd
#from rdkit.Chem import PandasTools
import time
import os
from rdkit import DataStructs

def bit2str(n):
    s=""
    for l in n:
        s+=str(l) +','
    return(s)

radii=[2,3,4,5]
bits=[512, 1024, 2048, 4096, 8192]

alkylation=[]
fhin=open('alkylation.txt', 'r')
lines=fhin.readlines()
for l in lines:
    alkylation.append(l[0:-1])
fhin.close()

insecticide=[]
fhin=open('insecticide.txt', 'r')
lines=fhin.readlines()
for l in lines:
    insecticide.append(l[0:-1])
fhin.close()

exclude=[]
fhin=open('exclude.txt', 'r')
lines=fhin.readlines()
for l in lines:
    exclude.append(l[0:-1])
fhin.close()


covalent=[]
fhin=open('covalent.txt', 'r')
lines=fhin.readlines()
for l in lines:
    covalent.append(l[0:-1])
fhin.close()

print(covalent)

covalent_data=[]

fhout=open('drugbank.csv', 'w')
morganheader=""
for r in radii:
    for b in bits:
        morganheader=morganheader + 'MorganFP_' + str(r) + '_' + str(b) + '\t'

fhout=open('drugbank.csv', 'w')
fhout.write('ID' + '\t' + 'SMILES' + '\t' + 'InChi' + '\t' + 'InChiKey' + '\t' + morganheader + '\n')

fhout_covalent=open('drugbank_covalent.csv', 'w')
fhout_covalent.write('ID' + '\t' + 'SMILES' + '\t' + 'InChi' + '\t' + 'InChiKey' + '\t' + morganheader + '\n')

if __name__ == "__main__":
    with open('drugbank_parsed.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        data=[]
        for row in reader:
            db_id=row['DrugBank ID']
            if( db_id in exclude):
                continue
            if(db_id in insecticide  or db_id in alkylation or db_id in covalent):
                covalent_data.append(  (row['DrugBank ID'], row['SMILES']))
            else:
                data.append( (row['DrugBank ID'], row['SMILES']))

    print(covalent_data)
    for (i, (id, smi)) in enumerate(data):
        m1 = Chem.MolFromSmiles(smi)
        if(m1==None):
            print('molecular generation failed')
            continue
        mol = Chem.AddHs(m1)
        info={}
        inchikey=Chem.inchi.MolToInchiKey(mol)
        inchi=Chem.inchi.MolToInchi(mol)
        morgan=""
        for r in radii:
            for b in bits:
                info={}
                morgan+=bit2str(list(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=r, nBits = b, bitInfo=info))) + '\t'
        fhout.write(str(id) + '\t' + smi + '\t' + str(inchi) + '\t' + str(inchikey) + '\t' + morgan  + '\n')
        
    for (i, (id, smi)) in enumerate(covalent_data):
        m1 = Chem.MolFromSmiles(smi)
        if(m1==None):
            print('molecular generation failed')
            continue
        mol = Chem.AddHs(m1)
        info={}
        inchikey=Chem.inchi.MolToInchiKey(mol)
        inchi=Chem.inchi.MolToInchi(mol)
        morgan=""
        for r in radii:
            for b in bits:
                info={}
                morgan+=bit2str(list(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=r, nBits = b, bitInfo=info))) + '\t'
        fhout_covalent.write(str(id) + '\t' + smi + '\t' + str(inchi) + '\t' + str(inchikey) + '\t' + morgan  + '\n')

fhout_covalent.close()
