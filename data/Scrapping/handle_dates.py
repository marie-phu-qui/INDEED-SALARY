from datetime import datetime, timedelta
import re

# Doit être importé dans l'outil de scrapping
def add_date() :
    now = datetime.now()
    df_days = df[df.Date_de_publication.str.contains('jour')]
    date_list = []
    for ele in df_days['Date_de_publication'] :
        nb_days = int(re.findall("[0-9]+", ele)[0])
        date_list.append(now - timedelta(days=nb_days))
    return date_list

def add_recent_dates() :
    df_days['Date'] = add_date()
    df["Date"] = 'vide'
    df.update(df_days)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df
add_recent_dates()