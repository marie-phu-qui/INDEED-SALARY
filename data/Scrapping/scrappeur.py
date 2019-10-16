import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
now = datetime.now()
import pandas as pd 
import numpy as np 
from time import sleep
import re
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


def Debug_pop():
    ''' Fonction permettant le debug lors d un POP'''
    sleep(0.5)
    try:
        browser.find_element_by_xpath('//*[@id="popover-x"]/a').click()
    except:
        pass

def Metier(string):
    ''' 
    Fonction permettant de scrapper le métier choisi
    input = string
        
    '''
    Debug_pop()   
    recherche = browser.find_element_by_xpath('//*[@id="text-input-what"]')
    recherche.click()
    recherche.clear()
    recherche.send_keys(string)
    sleep(0.5)
    recherche.send_keys(Keys.ENTER)
    Debug_pop()
    browser.find_element_by_xpath('//*[@id="refineresults"]/div[1]/span[2]/a').click()
    Debug_pop()

        
def Localisation(string):
    ''' 
    Fonction permettant de scrapper la localisation choisie
    input = string
        
    '''
    Debug_pop()
    recherche = browser.find_element_by_xpath('//*[@id="where"]')
    recherche.click()
    recherche.clear()
    recherche.send_keys(string)
    sleep(2)
    recherche.send_keys(Keys.ENTER)
    Debug_pop()
    
def Type_Contrat(string):  
    ''' 
    Fonction permettant de scrapper le contrat choisi
    input = string
        
    '''
    Debug_pop()
    
    if string.lower() == 'cdi':      
        browser.find_element_by_xpath("//*[@id='JOB_TYPE_rbo']/ul/li[1]/a/span[1]").click()
    if string.lower() == 'temps plein':
        browser.find_element_by_xpath("//*[@id='JOB_TYPE_rbo']/ul/li[2]/a/span[1]").click()
    if string.lower() == 'stage':
        browser.find_element_by_xpath("//*[@id='JOB_TYPE_rbo']/ul/li[3]/a/span[1]").click()
    if string.lower() == 'cdd':
        browser.find_element_by_xpath("//*[@id='JOB_TYPE_rbo']/ul/li[4]/a/span[1]").click()
    if string.lower() == 'apprentissage':
        browser.find_element_by_xpath("//*[@id='JOB_TYPE_rbo']/ul/li[5]/a/span[1]").click()
    if string.lower() == 'contrat pro':
        browser.find_element_by_xpath("//*[@id='JOB_TYPE_rbo']/ul/li[6]/a/span[1]").click()
    if string.lower() == 'freelance / indépendant':
        browser.find_element_by_xpath("//*[@id='JOB_TYPE_rbo']/ul/li[7]/a/span[1]").click()
    if string.lower() == 'intérim':
        browser.find_element_by_xpath("//*[@id='JOB_TYPE_rbo']/ul/li[8]/a/span[1]").click()
    if string.lower() == 'temps partiel':
        browser.find_element_by_xpath("//*[@id='JOB_TYPE_rbo']/ul/li[9]/a/span[1]").click()    
    Debug_pop()
    
    
    
def Scrappeur(metier,localisation):
    ''' 
     Fonction permettant de scrapper les pages 
    '''
    
    browser.get('https://www.indeed.fr/')
    browser.maximize_window()

    Metier(metier)
    Localisation(localisation)
    

    while True:
        Debug_pop()
        annonces = browser.find_elements_by_class_name("jobsearch-SerpJobCard")
        for i in range(0,len(annonces)):
            try:
                titre = annonces[i].find_element_by_class_name('title')
                Titre.append(titre.text)
            except:
                Titre.append('vide')
            try:
                nom = annonces[i].find_element_by_class_name('company')
                Nom_Entreprise.append(nom.text)

            except:
                Nom_Entreprise.append('vide')
            try:
                adresse= annonces[i].find_element_by_class_name('location')
                Adresse.append(adresse.text)
            except:
                Adresse.append('vide')
            try:
                date = annonces[i].find_element_by_class_name('date')
                Date_de_publication.append(date.text)

            except:
                Date_de_publication.append('vide')
            try:
                salaire = annonces[i].find_element_by_class_name('salaryText')
                Salaire.append(salaire.text)
            except:
                Salaire.append('vide')
                
            try:
                annonces[i].find_element_by_class_name('title').click()
                Debug_pop()
                descriptif= browser.find_element_by_xpath("//*[@id='vjs-content']")
                Descriptif_du_poste.append(descriptif.text)


            except:
                Descriptif_du_poste.append('vide')
                
            try:
                    identifiant = browser.current_url
                    pos1 = identifiant.find("vjk")
                    identifiant = identifiant[pos1:]
                    Identifiant.append(identifiant)
            except:
                    Identifiant.append('vide')



        try:
            browser.find_element_by_xpath("//*[contains(text(), 'Suivant')]").click() 
        except:
            break
    df = pd.DataFrame(dictionnaire) 
    
    


def Relance():
    
    while True:
        Debug_pop()
        annonces = browser.find_elements_by_class_name("jobsearch-SerpJobCard")
        for i in range(0,len(annonces)):
            
            try:
                titre = annonces[i].find_element_by_class_name('title')
                Titre.append(titre.text)
            except:
                Titre.append('vide')
            try:
                nom = annonces[i].find_element_by_class_name('company')
                Nom_Entreprise.append(nom.text)

            except:
                Nom_Entreprise.append('vide')
            try:
                adresse= annonces[i].find_element_by_class_name('location')
                Adresse.append(adresse.text)
            except:
                Adresse.append('vide')
            try:
                date = annonces[i].find_element_by_class_name('date')
                Date_de_publication.append(date.text)

            except:
                Date_de_publication.append('vide')
            try:
                salaire = annonces[i].find_element_by_class_name('salaryText')
                Salaire.append(salaire.text)
            except:
                Salaire.append('vide')

            try:
                annonces[i].find_element_by_class_name('title').click()
                Debug_pop()
                descriptif= browser.find_element_by_xpath("//*[@id='vjs-content']")
                Descriptif_du_poste.append(descriptif.text)


            except:
                Descriptif_du_poste.append('vide')

            try:
                    identifiant = browser.current_url
                    pos1 = identifiant.find("vjk")
                    identifiant = identifiant[pos1:]
                    Identifiant.append(identifiant)
            except:
                    Identifiant.append('vide')

        try:
            browser.find_element_by_xpath("//*[contains(text(), 'Suivant')]").click() 
            
        except:
            break
    df = pd.DataFrame(dictionnaire) 


def Occurence_regex_des(regex):
    liste = []
    for i in df['Descriptif_du_poste']:
        liste.append(re.findall(regex,i))
    cpt = 0
    for i in liste:
        if len(i) > 0:
            cpt+= 1
    print(f"récupération de : {cpt} parmi {len(df['Descriptif_du_poste'])} = {round(cpt/len(df['Descriptif_du_poste'])*100,2) }")
    return liste     
    
def Recup_date(liste1,liste2):
    
    liste_corr = []
    for i,j in zip(liste1,liste2):
        if (i == 'vide') and (re.search('(il y a) (\d+)(.*)(jour)',j)):
            
            temp = re.findall('(il y a) (\d+)(.*)(jour)',j)[0]
            temp = ' '.join(temp)
            liste_corr.append(temp)
            
            
        elif (i == "vide") and (re.search('(il y a) (\d+)(.*)(mois)',j)):
            
            temp = re.findall('(il y a) (\d+)(.*)(mois)',j)[0]
            temp = ' '.join(temp)
            liste_corr.append(temp)


        else:
            liste_corr.append(i)
    
    liste_corr  = [str(x) for x in liste_corr]
    liste_corr  = [s.replace('+', '') for s in liste_corr]

    return liste_corr    

def Correction_date(liste):
    ''' 
    Fonction permettant de corriger les dates de publications 
    example : il y a 2 jours en 05/10/2019 si now = 07/10/2019  
    input = liste
    output = liste
        
    '''
    liste_corr = []
    for i in liste:
 
            
        jours = 0
        mois = 0
        annees = 0 
       
 
        if ("Publiée à l'instant" or "Aujourd'hui") in i:
            
            d = datetime.today()
            d =d.strftime("%d/%m/%Y")
            liste_corr.append(d)
            
        elif re.search(' jours?',i): 
                
                day = int(re.findall('\d+',i)[0])
                d = datetime.today() + relativedelta(days=-day)
                d =d.strftime("%d/%m/%Y")
                liste_corr.append(d)

                    
        elif 'mois' in i :
            month =  int(re.findall('\d+',i)[0])
            d = datetime.today() + relativedelta(months=-month)
            d =d.strftime("%d/%m/%Y")
            liste_corr.append(d)
        else:
            d= i
            liste_corr.append(d)

      
    return liste_corr


def Calcul_mod(column):
    
    vide = df[column] == 'vide'
    non_vide = df[column] != 'vide'
    null = pd.isnull(df[column])
    nb_vide = df[column][vide].count()
    nb_non_vide = df[column][non_vide].count()
    nb_null = df[column][null].count()
    result = pd.DataFrame({"column":column,
              "vide": [nb_vide],
              "non_vide": [nb_non_vide],
              "null": [nb_null]})
    
    return result 
          
def Recup_exp(string):
    string = str(string)
    debutants = 'junior|débutant| debutant'
    confirmes = 'confirmé|intermédiaire intermediaire'
    senior ='senior'
    
    if re.search(debutants,string.lower()):
        string = 'junior'
    elif re.search(confirmes,string.lower()):
        string = 'confirmé'
        
    elif re.search('senior',string.lower()):
        string = 'senior'
    else:
        string = 'vide'
    return string

def Recup_exp_des(string):
    string = str(string)
    regex = "(?<=avec )(\d+)(?= ans | années d'expériences)| (\d+)(?= ans .Souhaité. ) |(?<=Nombre d’année d.exp...)(.*)(?= ans|années)"

    
    if re.search(regex,string.lower()):
        string = re.findall(regex,string.lower())

    else:
        string = 'vide'
    return string

def Recup_type_contrat(string):
    string = str(string)
    string = string.lower()
    cdi = "cdi"
    cdd = 'cdd'
    stage = 'stage'
    apprentissage = 'apprentissage'
    contrat_pro = 'contrat pro'
    interim = 'intérim'
    freelance_independant = '(?:)(?:freelance|indépendant|independant|independent)'
    
    if re.search(cdi,string):
        string = 'cdi'
    elif re.search(cdd,string):
        string = 'cdd'
    elif re.search(stage,string):
        string = 'stage'
        
    elif re.search(apprentissage,string):
        string = 'apprentissage'
    elif re.search(contrat_pro,string):
        string = 'contrat_pro'
    elif re.search(interim,string):
        string = 'intérim'
    elif re.search(freelance_independant,string):
        string = 'freelance'
    else:
        string = 'vide'
    return string         

Localisations = ["Paris","Lyon", "Toulouse", "Nantes","Bordeaux"]
metiers = {
           'Data_scientist': 'title:("data scientist" or "data science" or "machine learning" or "deep learning")',
           'Data_analyst'  : 'title:("data analyst" or "data analyste" or "data analysis" or "data analytics" or "analyste de données")',
           'Data_architect': 'title:("data architecte" or "data architect" or "administrateur de base de données")',
           'Data_engineer' : 'title:("data engineer" )',
           'Big_data'      : 'title:("big data")',
           'Autres_metiers_data' : 'title:data -scientist -analyst -engineer -"big data" -architecte -bi -"business intelligence"',
           'developer': 'title:("developer web" or "developer web" or "développeur front end" or "developer front end" or "developer front end" or "développeur back end" or "developer back end" or "developer back end" "développeur full" or "developer full" or "developer full" ) -data -business -affaire -toiles -bi -"business intelligence"',
           'software_engineer': 'title:("software engineer" or "ingénieur logiciel" or "ingenieur logiciel") -data -business -affaire -toiles -bi -"business intelligence"',
           'BI'   :  'title:(BI or "business intelligence")',
           'devops':   'title: devops -développeur -developer -developpeur'
           
           }

Titre = []
Nom_Entreprise = []
Adresse = []
Salaire = []
Descriptif_du_poste = []
Date_de_publication = []
Identifiant = []
metier = [list(metiers)[int(x)-1] for x in input("Taper 1 pour 'Data_scientist' \n 2 pour 'Data_analyst' \n 3 pour 'Data_architect' \n 4 pour 'Data_engineer' \n 5 pour 'Big_data' \n 6 pour 'Autres_metiers_data' \n 7 pour 'developer' \n 8 pour 'software_engineer' \n 9 pour 'BI' \n 10 pour 'devops'")][0]
loc = [Localisations[int(x)-1] for x in input("Taper 1 pour 'Paris' \n 2 pour 'Lyon' \n 3 pour 'Toulouse' \n 4 pour 'Nantes' \n 5 pour 'Bordeaux'")][0]

dictionnaire = {"Titre":Titre,"Nom_Entreprise":Nom_Entreprise,"Adresse":Adresse,
                "Salaire":Salaire,"Descriptif_du_poste":Descriptif_du_poste,
                "Date_de_publication":Date_de_publication,"métier_sc": metier,
                "loc_sc":loc,"Date_sc": now.strftime("%d/%m/%Y") ,
                "Identifiant": Identifiant}


browser = webdriver.Chrome()

Scrappeur(metiers[metier],loc)

name_csv = f"{metier}_{loc}_{now.strftime('%d%m%Y')}.csv"
df = pd.DataFrame(dictionnaire) 


masque = df['Descriptif_du_poste'] != 'vide'
df = df.loc[masque]

date = Recup_date(df['Date_de_publication'],df['Descriptif_du_poste'])
date = Correction_date(date)
df['Date_de_publication'] = date

df['Experiences'] = df["Titre"].apply(Recup_exp)

df['contrat'] = df['Descriptif_du_poste'].apply(Recup_type_contrat)

path = f'data/{loc}'
if not os.path.exists(path):
    os.mkdir(path)
df.to_csv(f'data/{loc}/{name_csv}', index=False)


from pymongo import MongoClient
from random import randint


client = MongoClient(host='localhost', port=27017)
db=client.tech_jobs

try:
    data = pd.DataFrame(list(db.indeed_jobs.find({},{"_id":0})))
    cols = df.columns
    data = data[cols]
    df = pd.concat([df,data],axis=0)
    df = df.drop_duplicates(subset=['Identifiant','Titre'], keep="last")
    db.indeed_jobs.drop()
    collection = db.indeed_jobs.insert_many(df.to_dict('records'))
    
except:

    collection = db.indeed_jobs.insert_many(df.to_dict('records'))  

