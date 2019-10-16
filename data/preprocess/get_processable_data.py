import pandas as pd
import os

data = pd.read_csv("data\indeed_jobs_exp_contrat.csv")

# check the amount of NaN
print(f"Our dataset has {data.isnull().sum()} NaNs.")

# process badly created columns
full_data = data.loc[(data['Salaire'] != "vide") & (data['Experiences'] != "vide") & (data['contrat'] != "vide")]
full_data.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)

# drop if NaNs
nonan_data = full_data[pd.notnull(full_data['métier_sc'])]

if not os.path.exists('data'):
    os.mkdir('data')
nonan_data.to_csv('data\complete_train.csv', index=False)