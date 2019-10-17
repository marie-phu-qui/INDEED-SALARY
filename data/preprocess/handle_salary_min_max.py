import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

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

