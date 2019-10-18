import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# HANDLE SALARY

df = pd.read_csv('..\Scrapping\data\complete\indeed_jobs.csv')


cols = df.columns.tolist()
del cols[0]
del cols[3]
cols.append("Salaire")

df = df[cols]

print(f"somme des salaires nuls : {df.Salaire.isnull().sum()}")

for index, x in df.iterrows():
    if type(x.Salaire) == float :
        df.loc[index, 'Salaire'] = 'vide'

print(f"Remplacement par `vide` - somme des salaires nuls : {df.Salaire.isnull().sum()}")

freq = []
for index, x in df.iterrows():
    if "an" in x.Salaire :
        freq.append('Annuel')
    elif "mois" in x.Salaire :
        freq.append('Mensuel')
    elif "semaine" in x.Salaire :
        freq.append('Hebdo')
    elif "jour" in x.Salaire :
        freq.append('Journalier')
    elif "heure" in x.Salaire :
        freq.append('Horaire')
    else :
        freq.append('NA')

freq_series = pd.Series(freq)

df["Freq"] = freq_series

print(f"somme des frequences uniques : {df.Freq.unique()}")

sal = []
sal_min = []
sal_max = []
for index, x in df.iterrows():
    if x.Salaire == 'vide' :
        sal.append('NA')

    else :
        sal.append([int(x.replace(' ', '')) for x in re.findall('(\d+\s?\d+)', x.Salaire)])


# check if NA
no_vide_sal = df.loc[(df["Salaire"] != "vide") & (df["Freq"] == "NA")]
print(f"somme des salaires non vides et sans frequence : {len(no_vide_sal)}" )

# add sal as salaire : [min, max]
sal_series = pd.Series(sal)
df["Sal"] = sal_series

sal_min = []
sal_max = []
for index, x in df.iterrows():
    if x.Sal == "NA" :
        sal_min.append("NA")
        sal_max.append("NA")
    elif len(x.Sal) == 1 :
        sal_min.append(x.Sal[0])
        sal_max.append(x.Sal[0])
    else :
        sal_min.append(x.Sal[0])
        sal_max.append(x.Sal[1])

sal_min_series = pd.Series(sal_min)
sal_max_series = pd.Series(sal_max)
df["Salaire_Min"] = sal_min_series
df["Salaire_Max"] = sal_max_series



for index, x in df.iterrows():
    if x.Freq == "Mensuel" or "mois" in x.Salaire :
        df.loc[index, ('Salaire_Min')] = df.loc[index, ('Salaire_Min')]*12
        df.loc[index, ('Salaire_Max')] = df.loc[index, ('Salaire_Max')]*12
    elif x.Freq == "Hebdo" or "semaine" in x.Salaire :
        df.loc[index, ('Salaire_Min')] = df.loc[index, ('Salaire_Min')]*52
        df.loc[index, ('Salaire_Max')] = df.loc[index, ('Salaire_Max')]*52
    elif x.Freq == "Journalier" or "jour" in x.Salaire :
        df.loc[index, ('Salaire_Min')] = df.loc[index, ('Salaire_Min')]*52*5
        df.loc[index, ('Salaire_Max')] = df.loc[index, ('Salaire_Max')]*52*5
    elif x.Freq == "Horaire" or "heure" in x.Salaire :
        df.loc[index, ('Salaire_Min')] = df.loc[index, ('Salaire_Min')]*35*52
        df.loc[index, ('Salaire_Max')] = df.loc[index, ('Salaire_Max')]*35*52


df.to_csv("data\indeed_format_processed.csv", index=False)

# HANDLE EXP

data = pd.read_csv('data\indeed_format_processed.csv')

no_exp_descr = data.loc[data["Experiences"] == "vide"]
descriptions = no_exp_descr["Descriptif_du_poste"]

def define_levels_from_desc(data) :  
    ''' 
    Fonction permettant de rajouter des niveaux expérience selon le contenu de la description
        
    '''
    for desc  in data["Descriptif_du_poste"] :
        # par termes
        junior = 'débutant|jeune|($profil|niveau)junior\b|assistant\b|jr\b'
        confirme = 'confirmé'
        senior = 'lead|senior|expert|encadrer\b|CTO|sr\b|mentore|diriger'
        # numéral
        num_junior = '(?i)(depuis (?:(?:\w* ){1,3})?[1-2]{1}(?:\D*[1-2])? an)'
        num_confirme = '(?i)(?:[2-4] ans minimum)|(?:depuis (?:(?:\w* ){1,3})?[2-4]{1}(?:\D*[1-2])? an)'
        num_senior = '(?i)(?:[5] ans minimum)|(?:depuis (?:(?:\w* ){1,3})?[5] an)'
        index = [data.index[data["Descriptif_du_poste"]== desc][0]]
        if re.findall(senior, desc) or re.findall(num_senior, desc) :
            data.loc[index, ('Experiences')]  = "senior"
        elif re.findall(confirme, desc)  or re.findall(num_confirme, desc) :
            data.loc[index, ('Experiences')]  = "confirmé"
        elif re.findall(junior, desc)  or re.findall(num_junior, desc) : 
            if re.findall('mentore encadre', desc) :
                pass
            else :
                data.loc[index, ('Experiences')]  = "junior"
    return data


# Ici, check la diminution de 'vide' pour Experiences et envoie un nouveau csv
new_data = define_levels_from_desc(no_exp_descr)
new_no_exp_descr = new_data.loc[new_data["Experiences"] == "vide"]
print(f"We now have {len(no_exp_descr) - len(new_no_exp_descr)} more data. And {len(new_data.loc[(new_data['contrat'] != 'vide') & (new_data['Experiences'] != 'vide') & (new_data['Salaire'] != 'vide')])} exploitable data.")

data.update(new_data)

if not os.path.exists('data'):
    os.mkdir('data')
data.to_csv('data\indeed_jobs_exp.csv', index=False)


# HANDLE CONTRATS

data = pd.read_csv('data\indeed_jobs_exp.csv')

no_contrat_descr = data.loc[data["contrat"] == "vide"]
descriptions = no_contrat_descr["Descriptif_du_poste"]

# À lancer après la fonction handle_exp.py

def define_contrats_from_desc(data) :
    ''' 
    Fonction permettant de rajouter des valeurs de contrats supplémentaires selon le contenu de la description
        
    '''
    for desc in data["Descriptif_du_poste"] :
        cdd = '(?i)(?:(?:contrat (?:à|a) (?:durée|duree) (?:déterminée|determinee|determine))|(?:cdd))'
        cdi = '(?i)(?:(?:contrat (?:à|a) (?:durée|duree) (?:indéterminée|indeterminee|indetermine))|(?:cdi))'
        stage = '(?i)(?:stage|stagiaire|internship|intern\b)'
        interim = '(?i)(?:intérim|interim|intérimaire|interimaire)'
        freelance = '(?i)(?:freelance|independant|indépendant|independent)'
        apprenti = '(?i)apprenti'
        contrat_pro = '(?:)(?:professionalis)'
        index = [data.index[data["Descriptif_du_poste"]== desc][0]]
        if re.findall(interim, desc) :
            data.loc[index, ('contrat')]  = "intérim"
        if re.findall(stage, desc) :
            data.loc[index, ('contrat')]  = "stage"
        if re.findall(freelance, desc) :
            data.loc[index, ('contrat')]  = "freelance"
        if re.findall(apprenti, desc) :
            data.loc[index, ('contrat')]  = "apprentissage"
        if re.findall(contrat_pro, desc) :
            data.loc[index, ('contrat')]  = "contrat_pro"
        if re.findall(cdd, desc) :
            data.loc[index, ('contrat')]  = "cdd"
        if re.findall(cdi, desc) :
            data.loc[index, ('contrat')]  = "cdi"
    return data

new_data = define_contrats_from_desc(no_contrat_descr)
new_no_contrat_descr = new_data.loc[new_data["contrat"] == "vide"]

print(f"We now have {len(no_contrat_descr) - len(new_no_contrat_descr)} more data. And {len(new_data.loc[(new_data['contrat'] != 'vide') & (new_data['Experiences'] != 'vide') & (new_data['Salaire'] != 'vide')])} exploitable data.")

data.update(new_data)

if not os.path.exists('data'):
    os.mkdir('data')
data.to_csv('data\indeed_jobs_exp_contrat.csv', index=False)


# GET PROCESSEABLE DATA

# check the amount of NaN
print(f"Our dataset NaNs : \n {data.isnull().sum()}")

# process badly created columns - should not be usefull when the csv are created correctly (handle_exp & handle_contrat) 
full_data = data.loc[(data['Salaire'] != "vide") & (data['Experiences'] != "vide") & (data['contrat'] != "vide")]

# drop if NaNs
nonan_data = full_data[pd.notnull(full_data['métier_sc'])]

print(len(nonan_data))

if not os.path.exists('data'):
    os.mkdir('data')
nonan_data.to_csv('data\complete_train.csv', index=False)