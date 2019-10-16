import pandas as pd 
import re
import os


data = pd.read_csv('..\Scrapping\data\complete\indeed_jobs.csv')

no_exp_descr = data.loc[data["Experiences"] == "vide"]
descriptions = no_exp_descr["Descriptif_du_poste"]

def define_levels_from_desc(data) :  
    ''' 
    Fonction permettant de rajouter des niveaux expérience selon le contenu de la description
        
    '''
    for desc in data["Descriptif_du_poste"] :
        junior = 'débutant | jeune | ($profil|niveau) junior\b | assistant\b | jr\b'
        confirme = 'confirmé'
        senior = 'lead | senior | expert | encadrer\b | CTO | sr\b | mentore | diriger'
        if re.findall(senior, desc) :
            data["Experiences"][data.index[data["Descriptif_du_poste"] == desc][0]]  = "senior"
        if re.findall(confirme, desc) :
            data["Experiences"][data.index[data["Descriptif_du_poste"] == desc][0]]  = "confirmé"
        if re.findall(junior, desc) : 
            if re.findall('mentore | encadre', desc) :
                pass
            else :
                data["Experiences"][data.index[data["Descriptif_du_poste"] == desc][0]]  = "junior"
    return data

# Ici, check la diminution de 'vide' pour Experiences et envoie un nouveau csv

new_data = define_levels_from_desc(no_exp_descr)
new_no_exp_descr = new_data.loc[new_data["Experiences"] == "vide"]
print(len(new_no_exp_descr), " and before ", len(no_exp_descr))
print(len(data))

data.update(new_data)

if not os.path.exists('data'):
    os.mkdir('data')
data.to_csv('data\indeed_jobs_exp.csv', index=False)