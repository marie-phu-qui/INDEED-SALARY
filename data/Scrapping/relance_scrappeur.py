var_metier = input("Enter le métier recherché : ")
var_localisation = input("Entrer la localistaion : ")


import pandas as pd 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
browser = webdriver.Chrome()
browser.get('https://www.indeed.fr/')
browser.maximize_window()
from time import sleep
import re

#Faire un script de scraping sur indeed (https://www.indeed.fr/)  qui permette à l’user de spécifier le type d’annonces qu’il souhaite récupérer :
#-	Métier (développeur, data scientist…)
#-	Type de contrat recherché (CDI, CDD, freelance…)
#-	Lieu de recherche (Paris, Toulouse, …)


def Metier(string):
    try:
        browser.find_element_by_xpath('//*[@id="popover-x"]/a').click()
    except:
        pass
        
    recherche = browser.find_element_by_xpath('//*[@id="text-input-what"]')
    recherche.click()
    recherche.clear()
    recherche.send_keys(string)
    sleep(2)
    recherche.send_keys(Keys.ENTER)
    sleep(2)
    try:
        browser.find_element_by_xpath('//*[@id="popover-x"]/a').click()
    except:
        pass

def Localisation(string):
    try:
        browser.find_element_by_xpath('//*[@id="popover-x"]/a').click()
    except:
        pass
    recherche = browser.find_element_by_xpath('//*[@id="where"]')
    recherche.click()
    recherche.clear()
    recherche.send_keys(string)
    sleep(2)
    recherche.send_keys(Keys.ENTER)
    sleep(2)
    try:
        browser.find_element_by_xpath('//*[@id="popover-x"]/a').click()
    except:
        pass
    
def Type_Contrat(string):
    try:
        browser.find_element_by_xpath('//*[@id="popover-x"]/a').click()
    except:
        pass
    
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
    sleep(2)
    try:
        browser.find_element_by_xpath('//*[@id="popover-x"]/a').click()
    except:
        pass
    
def Debug_pop():
    sleep(2)
    try:
        browser.find_element_by_xpath('//*[@id="popover-x"]/a').click()
    except:
        pass
    

#Les infos à scraper :
#-	Titre
#-	Nom de la boite
#-	Adresse
#-	Salaire
#-	Descriptif du poste
#-	Date de publication de l’annonce




#**Liste  à remplir**

Metier(var_metier)
Localisation(var_localisation)

#**Creation des listes vides pour chaque colonne**

import numpy as np 
Titre = []
Nom_Entreprise = []
Adresse = []
Salaire = []
Descriptif_du_poste = []
Date_de_publication = []


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
        browser.find_element_by_xpath("//*[contains(text(), 'Suivant')]").click() 
        sleep(2)
    except:
        break

dictionnaire = {"Titre":Titre,"Nom_Entreprise":Nom_Entreprise,"Adresse":Adresse,
                  "Salaire":Salaire,"Descriptif_du_poste":Descriptif_du_poste,"Date_de_publication":Date_de_publication,
               "métier scrappé": var_metier,"localisation scrapéé":var_localisation}
df = pd.DataFrame(dictionnaire) 

df.to_csv('indeed_jobs.csv',index=False)


from pymongo import MongoClient
from random import randint

client = MongoClient(host='localhost', port=27017)
db=client.tech_jobs
collection = db.indeed_jobs.insert_many(df.to_dict('records'))

data = pd.DataFrame(list(db.indeed_jobs.find()))