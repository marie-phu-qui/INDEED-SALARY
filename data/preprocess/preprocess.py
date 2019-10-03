from pymongo import MongoClient
import pandas as pd
import re

#  PRÉPROCESS DES DONNÉES SUR MONGO

# Get the not empty salary values
# Connecte à MongoDB - port par défaut 27017
client = MongoClient(host='localhost', port=27017)
# Accède à la collection "tech_jobs"
db=client.tech_jobs

data = pd.DataFrame(list(db.indeed_jobs.find()))

def find_etudes():
    list_bac = []
    for ele in data['Descriptif_du_poste'] :    
        list_bac.append(re.findall("bac ?\+ ?(\d)", ele))
    data["etudes"] = list_bac
    return data

print(find_etudes().head())

# def find_xp():
#     list_bac = []
#     for ele in df['Descriptif_du_poste'] :  
#         print(re.findall("exp ?(\d)", ele))
# #         list_bac.append(re.findall("expérience ?\+ ?(\d)", ele))
#     df["exp"] = list_bac
#     return df

# find_xp()