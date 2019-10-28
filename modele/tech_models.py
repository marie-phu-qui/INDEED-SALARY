from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import re
import nltk
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score,recall_score
from sklearn.model_selection import GridSearchCV


class models_indeed():
     
    def __init__(self):
        
        self.models_param = {'REG_LOG' :{'penalty' : ['l1', 'l2']},
                            'XGBOOST':{'min_child_weight': [1, 5, 10],'gamma': [0.5, 1, 1.5, 2, 5],
                                       'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'max_depth': [3, 4, 5]},
                           
                            'GBOOST': {'n_estimators':  [10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 1],
                                       'max_depth': [1, 2]},
                                       
                            'SVM_RBF': {'kernel': ['rbf'],'C': [0.01,0.1,1, 10, 50, 100, 500, 1000], 
                                    'gamma': ['auto',0.1, 1, 5, 10, 50,'auto_deprecated']},
                             
                            'RANDOM_FOREST': {'n_estimators':  [1,10], 'criterion': ['entropy', 'gini'], 
                                   'max_depth': [1, 2], 'max_features': ['auto']},
                             
                            'ADABOOST': {'n_estimators':  [10, 50, 100, 200], 'learning_rate': [0.1, 1, 10]}
}
     
        
        
        self.models_grid = {}
        self.models_grid['REG_LOG'] = GridSearchCV(LogisticRegression(random_state=0),self.models_param['REG_LOG'] ,
                                             scoring='accuracy', verbose=True, n_jobs=-1)
        self.models_grid['XGBOOST'] = GridSearchCV(XGBClassifier(random_state=0),self.models_param['XGBOOST'], 
                                              scoring='accuracy', verbose=True, n_jobs=-1)
        self.models_grid['GBOOST'] = GridSearchCV(GradientBoostingClassifier(random_state=0),self.models_param['GBOOST'],
                                             scoring='accuracy', verbose=True, n_jobs=-1)
        self.models_grid['SVM_RBF'] = GridSearchCV(SVC(kernel='rbf',random_state=0),self.models_param['SVM_RBF'],
                                          scoring='accuracy', verbose=True, n_jobs=-1)
        self.models_grid['RANDOM_FOREST'] = GridSearchCV(RandomForestClassifier(random_state=0),
                                                    self.models_param['RANDOM_FOREST'],scoring='accuracy', verbose=True, n_jobs=-1)
        
        self.models_grid['ADABOOST'] = GridSearchCV(AdaBoostClassifier(random_state=0),self.models_param['ADABOOST'],
                                               scoring='accuracy', verbose=True, n_jobs=-1)
  
        self.models = {}
        self.models['REG_LOG'] = LogisticRegression(random_state=0)
        self.models['XGBOOST'] = XGBClassifier(random_state=0)
        self.models['GBOOST'] = GradientBoostingClassifier(random_state=0)
        self.models['SVM_RBF'] = SVC(kernel='rbf',random_state=0)
        self.models['RANDOM_FOREST'] = RandomForestClassifier(random_state=0)                                 
        self.models['ADABOOST'] = AdaBoostClassifier(random_state=0)
        
    def fit(self,X,y,grid_search = True):
        if grid_search:
            
            for model in self.models_grid:
        
                self.models_grid[model].fit(X,y).best_estimator_
        else:
            
            for model in self.models:
                self.models[model].fit(X,y)
        self.grid_search = grid_search
             
    
    def predict(self,X):
        predict_dic = {}
        if self.grid_search:
            for model in self.models_grid:
                predict_dic[model] = self.models_grid[model].predict(X)
            return predict_dic
        else:
            for model in self.models:
                predict_dic[model] = self.models[model].predict(X)
            return predict_dic
                
            
            

    def accuracy_ml(self,y,y_pred):
        accuracy_dict = {}
        if self.grid_search:
            for model in self.models_grid:
                accuracy_dict[model] = accuracy_score(y_pred[model], y) 
            return accuracy_dict
        else:
            for model in self.models:
                accuracy_dict[model] = accuracy_score(y_pred[model], y) 
            return accuracy_dict
    
    def f1score_ml(self,y,y_pred):
        f1_score_dict = {}
        if self.grid_search:
            for model in self.models_grid:
                f1_score_dict[model] = f1_score(y_pred[model], y,average='weighted') 
            return f1_score_dict
        else:
            for model in self.models:
                f1_score_dict[model] = f1_score(y_pred[model], y,average='weighted') 
            return f1_score_dict
        


def Recup_salaire(salaire):
   
    match = re.findall(r"(\d+)", re.sub(r"(?<=\d)\s+(?=\d)", "", str(salaire))) 
    if match != 'vide':
        try:
            sal_freq = ['an', 'mois', 'semaine', 'jour', 'heure']
            sal_factor = [1, 12, 52.14, 228/1.5, 35 * 52.14]
            freq_res = re.search(r'|'.join(sal_freq), str(salaire)).group()
            sal_avg = np.array(match).astype(np.float).mean() * sal_factor[sal_freq.index(freq_res)]
            if sal_avg > 1000:
                sal_avg /= 1000
            assert sal_avg > 20 and sal_avg < 200
            return sal_avg
        except:
            return 'vide'
    else:
        return 'vide'
    
def attribution_class_salaire(salaire):
    
    salaire = float(salaire)
    if salaire < 33.0:
        salaire = 'faible'
    elif salaire < 43.5:
        salaire = 'moyen'
    elif salaire < 56.0:
        salaire = 'élevé'
    else:
        salaire ='tres élevé'
    return salaire





df = pd.read_csv("../data/preprocess/data/indeed_jobs_exp_contrat.csv")
df = df.drop(['Adresse','Date_sc','Identifiant','Nom_Entreprise','Data_de_publication','Experiences'],axis = 1)
df['Salaire_corr'] = df['Salaire'].map(Recup_salaire)
df = df.drop(['Salaire'],axis=1)
masque = (df != 'vide')
df = df[masque]
df = df.dropna()
df['target'] = df["Salaire_corr"].apply(attribution_class_salaire)
df = df.reset_index(drop=True)
df.head(3)
masque_1 = (df['métier_sc'] != 'Autres_metiers_data')
masque_2 = (df['métier_sc'] != 'Data_scientist')
masque_3 = (df['métier_sc'] != 'Data_engineer')
masque_4 = (df['métier_sc'] != 'Big_data')
masque_5 = (df['métier_sc'] != 'Data_analyst')
masque_6 = (df['métier_sc'] != 'Data_architect')
masque_7 = (df['métier_sc'] != 'BI')
df = df[masque_1]
df = df[masque_2]
df = df[masque_3]
df = df[masque_4]
df = df[masque_5]
df = df[masque_6]
df = df[masque_7]
df = df.reset_index(drop=True)


# # Choix des classes salaires


X = df.loc[:,'Salaire_corr'].values.reshape(-1,1)
y = np.array([0 for x in range(len(X))]).reshape(-1,1)


X = np.append(X, y, axis=1)



from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))    
plt.grid()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()



kmeansmodel = KMeans(n_clusters= 4, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)
fig = plt.figure(figsize=(14,10))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s = 300,
            c = 'black', label = 'Centroids',marker="X")
plt.title('Clusters salaire')
plt.xlabel('Salaire moyen annuel(k$)')
plt.axis([0, 100, -0.1, 1])
plt.grid()
plt.legend()
plt.show();


# # Titres descriptions Count Vectorizer 

from gensim.utils import tokenize # ana
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=set(stopwords.words('french'))

def Tokenizer(text):
    temp = tokenize(text,lowercase=True)
    temp = " ".join(temp) 
    return temp
df['Titre_propre'] = df['Titre'].apply(Tokenizer)
df['Desciption_propre'] = df['Descriptif_du_poste'].apply(Tokenizer)


from sklearn.feature_extraction.text import CountVectorizer
cv_titre  = CountVectorizer(ngram_range=(1, 2), min_df = .005, max_df=0.8, stop_words=stop_words)
cv_des = CountVectorizer(ngram_range=(1, 2), min_df = .02, max_df=0.8, stop_words=stop_words)

train_titre = cv_titre.fit_transform(df["Titre_propre"])
train_des = cv_des.fit_transform(df['Desciption_propre'])

values_titre = train_titre.toarray()
columns_titre = cv_titre.get_feature_names()

values_des = train_des.toarray()
columns_des = cv_des.get_feature_names()

df_titre = pd.DataFrame(values_titre,columns = columns_titre)
df_titre['Salaire_corr']=df['Salaire_corr'].astype(float)

df_des = pd.DataFrame(values_des,columns = columns_des)
df_des['Salaire_corr']=df['Salaire_corr'].astype(float)


titre_matrice_corr_salaire = df_titre.corr()['Salaire_corr'].sort_values(ascending=False)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df_titre_test = df_titre.copy()
df_titre_test['Salaire_corr'] = df_titre_test['Salaire_corr'].apply(attribution_class_salaire)
df_titre_test.loc[:,'Salaire_corr'] = labelencoder.fit_transform(df_titre_test.loc[:,'Salaire_corr'])
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(criterion='entropy',random_state=0).fit(df_titre_test.iloc[:,:-1].values, df_titre_test.iloc[:,-1])
features_poid_titre =pd.DataFrame({"features":df_titre_test.columns[:-1],"importance":tree_clf.feature_importances_})
features_poid_titre = features_poid_titre.sort_values('importance',ascending=False)

df_des_test = df_des.copy()
df_des_test['Salaire_corr'] = df_des_test['Salaire_corr'].apply(attribution_class_salaire)
df_des_test.loc[:,'Salaire_corr'] = labelencoder.fit_transform(df_des_test.loc[:,'Salaire_corr'])
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(criterion='entropy',random_state=0).fit(df_des_test.iloc[:,:-1].values, df_des_test.iloc[:,-1])
features_poid_des =pd.DataFrame({"features":df_des_test.columns[:-1],"importance":tree_clf.feature_importances_})
features_poid_des = features_poid_des.sort_values('importance',ascending=False)



text = ''
for x in df['Desciption_propre'].str.lower():
    
    text = text + x

LISTE = ["signaler cette","cette offre",'poste','plus','continuer postuler','cette','client',
        'postuler','signaler','continuer','jours','clients','mission'] + stopwords.words('french')
text = ' '.join([word for word in text.split() if word not in (LISTE)]) 

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.figure(figsize=(15,8))
wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()




text = ''
for x in df['Titre_propre'].str.lower():
    
    text = text + x

LISTE = ["signaler cette","cette offre",'poste','plus','continuer postuler','cette','client',
        'postuler','signaler','continuer','jours','clients','mission'] + stopwords.words('french')
text = ' '.join([word for word in text.split() if word not in (LISTE)]) 

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.figure(figsize=(15,8))
wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



df_features = df.copy()
salaires = df_features['Salaire_corr'].copy()
df_features =df_features.drop(['Freq','Sal','Titre', 'Descriptif_du_poste','target', 'Titre_propre', 'Desciption_propre','Salaire_corr'],axis=1)
df_features.head()

labelencoder = LabelEncoder()

df_features = pd.concat([df_features, pd.get_dummies(df_features['métier_sc'])], axis=1)
df_features = pd.concat([df_features, pd.get_dummies(df_features['loc_sc'])], axis=1)
df_features = pd.concat([df_features, pd.get_dummies(df_features['contrat'])], axis=1)
df_features = df_features.drop(['métier_sc', 'contrat' ,'loc_sc'],axis=1)
df_features['Salaire_corr'] = salaires.apply(attribution_class_salaire)
df_features['Salaire_corr'] = labelencoder.fit_transform(df_features['Salaire_corr'])
df_features.head(3)



X=df_features.iloc[:,:-1].values
y=df_features.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =0 )



models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=False)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result0 = pd.concat(frames)
result0=result0.round(2)
result0['Features'] = 'Métier +Contrat +Localisation'
result0



models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=True)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result01 = pd.concat(frames)
result01=result01.round(2)
result01['Features'] = 'Métier +Contrat +Localisation +Gs'
result01




df_titre_models = df_titre.copy()
labelencoder = LabelEncoder()
df_titre_models['Salaire_corr'] = df_titre_models['Salaire_corr'].apply(attribution_class_salaire)
df_titre_models['Salaire_corr'] = labelencoder.fit_transform(df_titre_models['Salaire_corr'])
df_titre_models = df_titre_models.loc[:,list(features_poid_titre['features'][:20])+['Salaire_corr']]
df_titre_models.head(3)

X=df_titre_models.iloc[:,:-1].values
y=df_titre_models.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =0 )




models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=False)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result1 = pd.concat(frames)
result1=result1.round(2)
result1['Features'] = 'Titre'
result1


models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=True)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result2 = pd.concat(frames)
result2=result2.round(2)
result2['Features'] = 'Titre +Gs'
result2



df_titre_models1 = df_titre_models.iloc[:,:-1].copy()
labelencoder = LabelEncoder()
df_titre_models1['métier_sc'] = df['métier_sc'] # correct
df_titre_models1['contrat'] = df['contrat'] # correct
df_titre_models1['loc_sc'] = df['loc_sc'] # correct
df_titre_models1 = pd.concat([df_titre_models1, pd.get_dummies(df_titre_models1['métier_sc'])], axis=1)
df_titre_models1 = pd.concat([df_titre_models1, pd.get_dummies(df_titre_models1['loc_sc'])], axis=1)
df_titre_models1 = pd.concat([df_titre_models1, pd.get_dummies(df_titre_models1['contrat'])], axis=1)
df_titre_models1 = df_titre_models1.drop(['métier_sc', 'contrat' ,'loc_sc'],axis=1)
df_titre_models1['Salaire_corr'] = df_titre_models.iloc[:,-1]
df_titre_models1.head()


df_titre_models1.columns


X=df_titre_models1.iloc[:,:-1].values
y=df_titre_models1.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =0 )


# # Comparaison modèles avec hyperparamètres par défault


models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=False)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result3 = pd.concat(frames)
result3=result3.round(2)
result3['Features'] = 'Titre +Métier +Contrat +Localisation'
result3



models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=True)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result4 = pd.concat(frames)
result4=result4.round(2)
result4['Features'] = 'Titre +Métier +Contrat +Localisation +Gs'
result4




df_des_models = df_des.copy()
labelencoder = LabelEncoder()
df_des_models['Salaire_corr'] = df_des_models['Salaire_corr'].apply(attribution_class_salaire)
df_des_models['Salaire_corr'] = labelencoder.fit_transform(df_des_models['Salaire_corr'])
df_des_models = df_des_models.loc[:,list(features_poid_des['features'][:20])+['Salaire_corr']]

X=df_des_models.iloc[:,:-1].values
y=df_des_models.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =0 )



models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=False)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result5 = pd.concat(frames)
result5=result5.round(2)
result5['Features'] = 'Desc'
result5



models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=True)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result6 = pd.concat(frames)
result6=result6.round(2)
result6['Features'] = 'Desc +Gs'
result6




df_titre_desc = pd.concat([df_titre_models.iloc[:,:-1].copy(),df_des_models.copy() ], axis=1)
X=df_titre_desc.iloc[:,:-1].values
y=df_titre_desc.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =0 )



models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=False)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result7 = pd.concat(frames)
result7=result7.round(2)
result7['Features'] = 'Titre +Desc'
result7


models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=True)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result8 = pd.concat(frames)
result8=result8.round(2)
result8['Features'] = 'Titre +Desc +Gs'
result8


# # <center> Modeles Titre + Description + métier + contrat + localisation  </center>

df_titre_desc_features = pd.concat([df_titre_models1.iloc[:,:-1].copy(),df_des_models.copy() ], axis=1)

X=df_titre_desc_features.iloc[:,:-1].values
y=df_titre_desc_features.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =0 )


# # Comparaison modèles avec hyperparamètres par défault


models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=False)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result9 = pd.concat(frames)
result9=result9.round(2)
result9['Features'] = 'Titre +Desc +Métier +contrat +Localisation'
result9


# # Comparaison modèles en utilisant gridsearch


models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=True)
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
result10 = pd.concat(frames)
result10=result10.round(2)
result10['Features'] = 'Titre +Des +Métier +Contrat +localisation +Gs'
result10


# # Résultat final


RESULT = pd.concat([result0,result1,result3,result7,result9,result5,result01,result2,result4,result8,result10,result6],axis=0)
RESULT



df_titre_models1.head(2)
X=df_titre_models1.iloc[:,:-1].values
y=df_titre_models1.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =0 )


models_ind = models_indeed()
models_ind.fit(X_train,y_train,grid_search=True)
y_pred = models_ind.predict(X_test)
y_pred
accuracy_test = models_ind.accuracy_ml(y_test, models_ind.predict(X_test))
fscore = models_ind.f1score_ml(y_test, models_ind.predict(X_test))
accuracy_train = models_ind.accuracy_ml(y_train, models_ind.predict(X_train))
df_accuracy_test = pd.DataFrame(accuracy_test,index=['Accuracy_test'])
df_accuracy_train = pd.DataFrame(accuracy_train,index=['Accuracy_train'])
df_f1_score = pd.DataFrame(fscore,index=['Fscore'])
frames = [df_accuracy_train, df_accuracy_test,df_f1_score]
resultat = pd.concat(frames)
resultat=resultat.round(2)
resultat['Features'] = 'Titre +Métier +Contrat +Localisation +Gs'
resultat


from sklearn.ensemble import VotingClassifier

estimators=[('REG_LOG',models_ind.models_grid['REG_LOG']), ('XGBOOST',models_ind.models_grid['XGBOOST']),
            ('GBOOST',models_ind.models_grid['GBOOST']), ('SVM_RBF',models_ind.models_grid['SVM_RBF']), 
            ('RANDOM_FOREST',models_ind.models_grid['RANDOM_FOREST']), ('ADABOOST',models_ind.models_grid['ADABOOST'])]

ensemble = VotingClassifier(estimators, voting='hard')
ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
accuracy_test = accuracy_score(y_pred,y_test)
fscore = f1_score(y_pred,y_test,average='weighted')
accuracy_train = accuracy_score(ensemble.predict(X_train),y_train)
                                
df_ensemble = pd.DataFrame({'Méthode_ensembliste':[accuracy_train,accuracy_test,fscore]},index=['Accuracy_train','Accuracy_test','Fscore'])
df_ensemble['Features']  = 'Titre +Métier +Contrat +Localisation +Gs'                                                                                          
df_ensemble=df_ensemble.round(2)

df = pd.read_csv("indeed_jobs1.csv")
# selection des colonnes
df = df.loc[:,['Titre','métier_sc','loc_sc','contrat']]
# drop des nans et valeurs 'vide'
df=df.dropna()
df=df[(df != 'vide')]
# récuperation des metiers de la data uniquement
masque_1 = (df['métier_sc'] != 'Autres_metiers_data')
masque_2 = (df['métier_sc'] != 'Data_scientist')
masque_3 = (df['métier_sc'] != 'Data_engineer')
masque_4 = (df['métier_sc'] != 'Big_data')
masque_5 = (df['métier_sc'] != 'Data_analyst')
masque_6 = (df['métier_sc'] != 'Data_architect')
masque_7 = (df['métier_sc'] != 'BI')
df = df[masque_1]
df = df[masque_2]
df = df[masque_3]
df = df[masque_4]
df = df[masque_5]
df = df[masque_6]
df = df[masque_7]
df = df.reset_index(drop=True)

df.shape


# NLP du titre

df['Titre_propre'] = df['Titre'].apply(Tokenizer)

cv_titre  = CountVectorizer(ngram_range=(1, 2), min_df = .005, max_df=0.8, stop_words=stop_words)
train_titre = cv_titre.fit_transform(df["Titre_propre"])
values_titre = train_titre.toarray()
columns_titre = cv_titre.get_feature_names()
df_titre = pd.DataFrame(values_titre,columns = columns_titre)
df_titre.head(2)

df_titre = df_titre.loc[:,list(features_poid_titre['features'][:20])]
df_final = df_titre.copy()
df_final['métier_sc'] = df['métier_sc']
df_final['loc_sc'] = df['loc_sc']
df_final['contrat'] = df['contrat']
df_final = pd.concat([df_final, pd.get_dummies(df_final['métier_sc'])], axis=1)
df_final = pd.concat([df_final, pd.get_dummies(df_final['loc_sc'])], axis=1)
df_final = pd.concat([df_final, pd.get_dummies(df_final['contrat'])], axis=1)
df_final= df_final.drop(['métier_sc', 'contrat' ,'loc_sc'],axis=1)
df_final = df_final.drop(['contrat_pro'],axis=1)
df_final= df_final.fillna(0)
salaire = ensemble.predict(df_final.values)

csv_tech = df.copy()
csv_tech.shape
csv_tech['Salaire'] = salaire
classe_0 = '20 - 33.0'
classe_1 = '33 - 43.5'
classe_2 = '43.5 - 56.0'
classe_3 = '56.0 - 77.5'

#43.5
#56.0
#77.55825
#33.0

def range_salaire(string):
    if int(string) == 0:
        return classe_0
    
    if int(string) == 1:
        return classe_1
    
    if int(string) == 2:
        return classe_2
    
    if int(string) == 3:
        return classe_3
    


def x_min(string):
    if string == classe_0:
        return 20
    if string == classe_1:
        return 33
    if string == classe_2:
        return 43.5
    if string == classe_3:
        return 56.0



def x_max(string):
    if string == classe_0:
        return 33.0
    if string == classe_1:
        return 43.5
    if string == classe_2:
        return 56.0
    if string == classe_3:
        return 77.5

csv_tech['range_salaire'] = csv_tech['Salaire'].apply(range_salaire)
csv_tech['x_min'] = csv_tech['range_salaire'].apply(x_min)
csv_tech['x_max'] = csv_tech['range_salaire'].apply(x_max)

csv_tech.head()

if not os.path.exists('data'):
    os.mkdir('data')
csv_tech.to_csv('data\csv_tech.csv',index=False)