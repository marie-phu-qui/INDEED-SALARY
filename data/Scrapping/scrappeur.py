import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
browser = webdriver.Chrome()
browser.get('https://www.indeed.fr/')
browser.maximize_window()
from time import sleep
import re


def Metier(string):
    recherche = browser.find_element_by_xpath('//*[@id="text-input-what"]')
    recherche.click()
    recherche.clear()
    recherche.send_keys(string)
    sleep(3)
    recherche.send_keys(Keys.ENTER)


def Localisation(string):
    recherche = browser.find_element_by_xpath('//*[@id="where"]')
    recherche.click()
    recherche.clear()
    recherche.send_keys(string)
    sleep(3)
    recherche.send_keys(Keys.ENTER)


def Type_Contrat(string):
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




Metier("data scientist")
Localisation("Paris")

import numpy as np

Titre = []
Nom_Entreprise = []
Adresse = []
Salaire = []
Descriptif_du_poste = []
Date_de_publication = []

while True:
    annonces = browser.find_elements_by_class_name("jobsearch-SerpJobCard")
    for i in range(0, len(annonces)):
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
            adresse = annonces[i].find_element_by_class_name('location')
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
            sleep(2)
            descriptif = browser.find_element_by_xpath("//*[@id='vjs-content']")
            Descriptif_du_poste.append(descriptif.text)
        except:
            Descriptif_du_poste.append('vide')

    try:
        browser.find_element_by_xpath("//*[contains(text(), 'Suivant')]").click()
        sleep(2)
    except:
        break

df = pd.DataFrame({"Titre": Titre, "Nom_Entreprise": Nom_Entreprise, "Adresse": Adresse,
                   "Salaire": Salaire, "Descriptif_du_poste": Descriptif_du_poste,
                   "Date_de_publication": Date_de_publication})

