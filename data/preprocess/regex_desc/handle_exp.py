import pandas as pd 
import re
import os


data = pd.read_csv('..\data\indeed_format_processed.csv')

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

if not os.path.exists('..\data'):
    os.mkdir('..\data')
data.to_csv('..\data\indeed_jobs_exp.csv', index=False)