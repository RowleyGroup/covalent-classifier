import csv
import glob
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import rdkit.Chem.MolStandardize as rdMolStandardize

import pubchempy as pcp
from openpyxl import load_workbook
import pandas as pd
#from rdkit.Chem import PandasTools
import time
import os
from rdkit import DataStructs
from rdkit import RDLogger  

RDLogger.DisableLog('rdApp.*')

organic_atoms = set([1, 5, 6, 7, 8, 9, 16, 17, 35, 53])


def reorderTautomers(m):
    enumerator = rdMolStandardize.TautomerEnumerator()
    canon = enumerator.Canonicalize(m)
    csmi = Chem.MolToSmiles(canon)
    res = [canon]
    tauts = enumerator.Enumerate(m)
    smis = [Chem.MolToSmiles(x) for x in tauts]
    stpl = sorted((x,y) for x,y in zip(smis,tauts) if x!=csmi)
    res += [y for x,y in stpl]
    return res

alkylation_drugbank_ids=[]
fhin=open('alkylation.txt', 'r')
lines=fhin.readlines()
for l in lines:
    alkylation_drugbank_ids.append(l[0:-1])
fhin.close()

insecticide_drugbank_ids=[]
fhin=open('insecticide.txt', 'r')
lines=fhin.readlines()
for l in lines:
    insecticide_drugbank_ids.append(l[0:-1])
fhin.close()

exclude_drugbank_ids=[]
fhin=open('exclude.txt', 'r')
lines=fhin.readlines()
for l in lines:
    exclude_drugbank_ids.append(l[0:-1])
fhin.close()


covalent_drugbank_ids=[]
fhin=open('covalent.txt', 'r')
lines=fhin.readlines()
for l in lines:
    covalent_drugbank_ids.append(l[0:-1])
fhin.close()


covalent_data=[]
noncovalent_data=[]

fhout_covalent=open('trainingset_covalent.csv', 'w')
fhout_noncovalent=open('trainingset_noncovalent.csv', 'w')

covalent_inchikeys=[]
noncovalent_inchikeys=[]

if __name__ == "__main__":
    with open('CovInDB.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        data=[]
        # Inhibitor_id,InChI,InChI_Key
        for row in reader:
            inchi_orig=row['InChI']
            m1=Chem.MolFromInchi(inchi_orig)
            mol = Chem.AddHs(m1)
            mol=Chem.MolStandardize.rdMolStandardize.CanonicalTautomer(mol)
            atom_num_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            is_organic = (set(atom_num_list) <= organic_atoms)

            if not is_organic:
                continue
            info={}
            inchikey=Chem.inchi.MolToInchiKey(mol)
            inchikey_connectivity=inchikey.split('-')[0]
            inchi=Chem.inchi.MolToInchi(mol)
            if(inchikey_connectivity in covalent_inchikeys):
                continue

            covalent_data.append(  (inchi, inchikey))
            covalent_inchikeys.append(inchikey_connectivity)

                
    with open('drugbank_inchi.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        data=[]
        for row in reader:
            inchi_orig=row['InChI']
            db_id=row['DrugBankID']
            if( db_id in exclude_drugbank_ids):
                continue
            
            try:
                m1=Chem.MolFromInchi(inchi_orig)
                mol = Chem.AddHs(m1)
                mol=Chem.MolStandardize.rdMolStandardize.CanonicalTautomer(mol)

                atom_num_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                is_organic = (set(atom_num_list) <= organic_atoms)

                if not is_organic:
                    continue

                info={}
                inchikey=Chem.inchi.MolToInchiKey(mol)
                inchi=Chem.inchi.MolToInchi(mol)
                inchikey_connectivity=inchikey.split('-')[0]

            except:
                continue
            
            if(inchikey_connectivity in covalent_inchikeys):
                continue
            
            if(db_id in insecticide_drugbank_ids  or db_id in alkylation_drugbank_ids or db_id in covalent_drugbank_ids):
                covalent_data.append(  (inchi, inchikey))
                covalent_inchikeys.append(inchikey_connectivity)
            else:
                noncovalent_data.append( (inchi, inchikey) )
                noncovalent_inchikeys.append(inchikey_connectivity)

    with open('BindingDB.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            inchi_orig=row['InChI']
            try:
                m1=Chem.MolFromInchi(inchi_orig)
                mol = Chem.AddHs(m1)
                mol=Chem.MolStandardize.rdMolStandardize.CanonicalTautomer(mol)
                atom_num_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                is_organic = (set(atom_num_list) <= organic_atoms)

                if not is_organic:
                    continue

                info={}
                inchikey=Chem.inchi.MolToInchiKey(mol)
                inchi=Chem.inchi.MolToInchi(mol)
                inchikey_connectivity=inchikey.split('-')[0]
            except:
                continue
            if(inchikey_connectivity in covalent_inchikeys or inchikey_connectivity in noncovalent_inchikeys):
                continue
            noncovalent_data.append( (inchi, inchikey) )

fhout_covalent.write('InChI')
for (inchi, inchikey) in covalent_data:
    fhout_covalent.write(inchi + '\n')
fhout_covalent.close()

fhout_noncovalent.write('InChI\n')
for (inchi, inchikey) in noncovalent_data:
    fhout_noncovalent.write(inchi + '\n')

fhout_noncovalent.close()


test_set_dir=['aldehyde', 'alkyne',  'aziridine', 'chlorobenzene', 'epoxides',  'haloacetamides', 'furan',
              'isothiocyanates', 'lactone',  'nitrile', 'quinone', 'atypical', 'sulfonyl', 'thioketones', 'unsaturated']
            
for group in test_set_dir:
    reader = csv.DictReader(open('test_set/RowleyTestSet-positive_' + group + '.csv', mode='r'), delimiter='\t')
    test_set_inchi=[]
    fhout=open('testset-positive_' + group + '.csv', 'w')
    fhout.write('InChI\n')

    for row in reader:
        inchi_orig=row['InChI']
        try:
            m1=Chem.MolFromInchi(inchi_orig)
            mol = Chem.AddHs(m1)
            mol=Chem.MolStandardize.rdMolStandardize.CanonicalTautomer(mol)


            atom_num_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            is_organic = (set(atom_num_list) <= organic_atoms)

            if not is_organic:
                continue

            info={}
            inchikey=Chem.inchi.MolToInchiKey(mol)
            inchi=Chem.inchi.MolToInchi(mol)
            inchikey_connectivity=inchikey.split('-')[0]
        except:
            continue
        if(inchikey_connectivity in covalent_inchikeys or inchikey_connectivity in noncovalent_inchikeys):
            continue
        fhout.write(inchi + '\n')
    fhout.close()
#RowleyTestSet-negative_first_disclosures.csv 
reader = csv.DictReader(open('test_set/RowleyTestSet-negative_first_disclosures.csv', mode='r'), delimiter='\t')
test_set_inchi=[]

fhout=open('testset-negative_first_disclosure.csv', 'w')
fhout.write('InChI\n')

for row in reader:
    inchi_orig=row['InChI']
    try:
        m1=Chem.MolFromInchi(inchi_orig)
        mol = Chem.AddHs(m1)
        mol=Chem.MolStandardize.rdMolStandardize.CanonicalTautomer(mol)

        atom_num_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        is_organic = (set(atom_num_list) <= organic_atoms)

        if not is_organic:
            continue

        info={}
        inchikey=Chem.inchi.MolToInchiKey(mol)
        inchi=Chem.inchi.MolToInchi(mol)
        inchikey_connectivity=inchikey.split('-')[0]
    except:
        continue
    if(inchikey_connectivity in covalent_inchikeys or inchikey_connectivity in noncovalent_inchikeys):
        continue
    fhout.write(inchi + '\n')
fhout.close()

reader = csv.DictReader(open('test_set/RowleyTestSet-negative_false_covalent.csv', mode='r'), delimiter='\t')
test_set_inchi=[]

fhout=open('testset-negative_false_covalent.csv', 'w')
fhout.write('InChI\n')

for row in reader:
    inchi_orig=row['InChI']
    try:
        m1=Chem.MolFromInchi(inchi_orig)
        mol = Chem.AddHs(m1)
        mol=Chem.MolStandardize.rdMolStandardize.CanonicalTautomer(mol)

        atom_num_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        is_organic = (set(atom_num_list) <= organic_atoms)
        if not is_organic:
            continue
        info={}
        inchikey=Chem.inchi.MolToInchiKey(mol)
        inchi=Chem.inchi.MolToInchi(mol)
        inchikey_connectivity=inchikey.split('-')[0]
    except:
        continue
    if(inchikey_connectivity in covalent_inchikeys or inchikey_connectivity in noncovalent_inchikeys):
        continue
    fhout.write(inchi + '\n')
fhout.close()

