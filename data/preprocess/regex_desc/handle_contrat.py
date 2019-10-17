import pandas as pd 
import re
import os

data = pd.read_csv('..\data\indeed_jobs_exp.csv')

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
        # interim = '(?i)(?:intérim|interim)'
        freelance = '(?i)(?:freelance|independant|indépendant|independent)'
        apprenti = '(?i)apprenti'
        contrat_pro = '(?:)(?:professionalis)'
        if re.findall(stage, desc) :
            data["contrat"][data.index[data["Descriptif_du_poste"] == desc][0]] = "stage"
        if re.findall(freelance, desc) :
            data["contrat"][data.index[data["Descriptif_du_poste"] == desc][0]] = "freelance"
        if re.findall(apprenti, desc) :
            data["contrat"][data.index[data["Descriptif_du_poste"] == desc][0]] = "apprentissage"
        if re.findall(contrat_pro, desc) :
            data["contrat"][data.index[data["Descriptif_du_poste"] == desc][0]] = "contrat_pro"
        if re.findall(cdd, desc) :
            data["contrat"][data.index[data["Descriptif_du_poste"] == desc][0]] = "cdd"
        if re.findall(cdi, desc) :
            data["contrat"][data.index[data["Descriptif_du_poste"] == desc][0]] = "cdi"
    return data

new_data = define_contrats_from_desc(no_contrat_descr)
new_no_contrat_descr = new_data.loc[new_data["contrat"] == "vide"]
print(len(new_no_contrat_descr), " and before ", len(no_contrat_descr))
print(len(data))

data.update(new_data)

if not os.path.exists('..\data'):
    os.mkdir('..\data')
data.to_csv('..\data\indeed_jobs_exp_contrat.csv', index=False)