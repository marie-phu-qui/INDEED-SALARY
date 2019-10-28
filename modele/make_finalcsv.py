import pandas as pd

df_tech =pd.read_csv('./data/csv_tech.csv')
df_data = pd.read_csv('./data/csv_data.csv')

data_final = pd.concat([df_tech,df_data],axis =0)

if not os.path.exists('data'):
    os.mkdir('data')
data_final.to_csv('./data/data_final.csv',index=False)