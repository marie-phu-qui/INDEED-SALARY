from pymongo import MongoClient
from random import randint

# Connecte à MongoDB - port par défaut 27017
client = MongoClient(host='localhost', port=27017)
# Crée la collection "tech_jobs"
db=client.tech_jobs

# Create mock-up sample data / LATER *Get the data*
titre = ['a', 'b', 'c']
company = ['d', 'e', 'f']
adress = ['1a', '2b', '3c']
salary = [1, 2, 3]
job_description = ['abab', 'blabla', 'laclac']
date = ['1/a', '2/b', '3/c']

# Met les données dans un dictionnaire 
for x in range(len(titre)):
    indeed_jobs = {
        'titre' : titre[x],
        'company' : company[x],
        'adress' : adress[x],
        'salary' : salary[x],
        'job_description' : job_description[x],
        'date' : date[x] 
    }

    # Ceci ajoute sans dropper l'ancien
    collection = db.indeed_jobs.insert_one(indeed_jobs)
#Valide l'aggrégat
print('Done')