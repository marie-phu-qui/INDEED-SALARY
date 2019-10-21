from pymongo import MongoClient
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

# Connecte à MongoDB - port par défaut 27017
client = MongoClient(host='localhost', port=27017)
# Crée la collection "tech_jobs"
db=client.tech_jobs

data = pd.DataFrame(list(db.indeed_jobs.find()))
salaries = data.loc[data['Salaire'] != 'vide']

# X = 
y = salaries['Salaire']

print(y)