from datetime import datetime, timedelta
import re

# Doit être importé dans l'outil de scrapping
def process_dates(df) :
#     process dates
    df_days = df[df.Date_de_publication.str.contains('jour')]
    df_months = df[df.Date_de_publication.str.contains('mois')]
    d_date_list = []
    m_date_list = []
    for ele in df_days['Date_de_publication'] :
        nb_days = int(re.findall("[0-9]+", ele)[0])
        date = (now - timedelta(days=nb_days)).strftime("%d/%m/%Y")
        d_date_list.append(date)
    for ele in df_months['Date_de_publication'] :
        nb_months = int(re.findall("[0-9]+", ele)[0])
        date = (now - timedelta(nb_months*365/12)).strftime("%d/%m/%Y")
        m_date_list.append(date)
    df_days['Date'] = d_date_list
    df_months['Date'] = m_date_list
    return df_days, df_months

def add_dates(df, process_dates()):    
#     add dates to the original dataframe
    df["Date"] = 'vide'
    df_days, df_months = process_dates()
    df.update(df_days)
    df.update(df_months)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df
    
add_dates()