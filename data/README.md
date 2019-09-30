## LA DATA

On forme un base de données au format requis : 

1. Faire un script de scraping sur **[INDEED](https://www.indeed.fr/)**  qui permette de spécifier le type d’annonces à récupérer :
-	Métier (développeur, data scientist…)
-	Type de contrat recherché (CDI, CDD, freelance…)
-	Lieu de recherche (Paris, Toulouse, …)

2. Les features à voir figurer dans les colonnes de notre dataset (! cette liste n'est pas exhaustive) :
Les infos à scraper :
-	Titre
-	Nom de la boite
-	Adresse
-	Salaire
-	Descriptif du poste
-	Date de publication de l’annonce

3. Il faudra se concentrer sur les annonces ( Les axes de classification selon les annonces) : 
-	Métiers : développeur, data scientist, data analyst, business intelligence.
-	Localisation : Paris, Lyon, Toulouse, Nantes et Bordeaux.
-	Type de contrat : tous

## LOCALISATION

Base de données **MONGODB**

## FORMAT 

**JSON**  
Nous avons également trouvé une **[API](https://opensource.indeedeng.io/api-documentation/)** afin de requester les informations en temps réel.