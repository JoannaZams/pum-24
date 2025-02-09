import streamlit as st
import pandas as pd
import pickle as pkl 
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*') # Disabling rdkit warnings
from sklearn.svm import SVR



class Featurizer:
    def __init__(self, train_smiles):
        self.scaler = StandardScaler()
        train_descriptors = self.get_descriptors(train_smiles)
        self.scaler.fit(train_descriptors)
        
    def featurize(self, smiles):
        descriptors = self.get_descriptors(smiles)
        scaled_descriptors = self.scaler.transform(descriptors)
        return scaled_descriptors

    def get_descriptors(self, smiles):
        df = pd.DataFrame({'SMILES': smiles})
        df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
        df['mol_wt'] = df['mol'].apply(rdMolDescriptors.CalcExactMolWt)             # Molecular weight
        df['logp'] = df['mol'].apply(Crippen.MolLogP)                               # LogP (lipophilicity)
        df['num_heavy_atoms'] = df['mol'].apply(rdMolDescriptors.CalcNumHeavyAtoms) # Number of heavy atoms
        df['num_HBD'] = df['mol'].apply(rdMolDescriptors.CalcNumHBD)                # Number of hydrogen bond donors
        df['num_HBA'] = df['mol'].apply(rdMolDescriptors.CalcNumHBA)                # Number of hydrogen bond acceptors
        df['aromatic_rings'] = df['mol'].apply(rdMolDescriptors.CalcNumAromaticRings) 
        return df.drop(columns=['mol', 'SMILES'], axis=1)
    

with open('data/svr.pkl', 'rb') as f:
    svr = pkl.load(f)
    
with open('data/feat.pkl', 'rb') as f:
    feat = pkl.load(f)
        
st.title('Solubility Predictions')
st.write('Welcome to the solubility prediction app!\n You can enter you SMILES code below and the app will predict the solubility of the molecule.')

df=pd.DataFrame()

smls = st.text_area('Enter your SMILES code here:').upper()
if smls:
    X = feat.featurize(smls.strip().split())
    df['SMILES'] = smls.strip().split()

if st.button('Show me the solubility!'):
    df["Solubility"] = svr.predict(X)
    st.write(df)
