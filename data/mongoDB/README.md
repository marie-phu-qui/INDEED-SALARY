# Partager la base de données MongoDB

## Envoi en fichier csv pour traitement

Sur le terminal une fois la base de données sur MongoDB, l'exporter telle que :

```
mongoexport --db tech_jobs --collection indeed_jobs --type=csv --fields _id,Titre,Nom_Entreprise,Adresse,Salaire,Descriptif_du_poste,Date_de_publication,'métier scrappé', 'localisation scrapéé' --out dump/indeed_jobs/indeed.csv
```
