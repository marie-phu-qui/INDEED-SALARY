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

# Créé une dataframe qui ne s'intéressera qu'aux données complètes (possède un salaire)
salaries = data.loc[data['Salaire'] != 'vide']

def get_salary_val():
    list_salaries = []
    for ele in salaries['Salaire'] :
        if 'mois' in ele :
            salary = int(re.findall("^([0-9]+ [0-9]*)", ele)[0].replace(" ", ""))*12
            list_salaries.append(salary)
        else :
            salary = int(re.findall("^([0-9]+ [0-9]*)", ele)[0].replace(" ", ""))
            list_salaries.append(salary)
    salaries['year_pay'] = list_salaries
    return salaries

def add_year_pay(salaries):
    data["year_pay"] = 'vide'
    data.update(salaries)
    return data

add_year_pay(get_salary_val())