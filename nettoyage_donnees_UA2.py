 
#Le script commence par l’importation de plusieurs bibliothèques Python essentielles 
#pour le traitement et la modélisation de données. Pandas et NumPy permettent de manipuler efficacement les 
#tableaux de données et d’effectuer des calculs numériques. Les modules de scikit-learn sont utilisés pour 
#le prétraitement, la sélection de caractéristiques, la création de modèles d’apprentissage automatique et 
#l’évaluation des performances. Plus précisément, OneHotEncoder, MinMaxScaler et FunctionTransformer servent 
#à transformer et normaliser les données, tandis que Pipeline et ColumnTransformer permettent d’automatiser tout 
#le processus d’analyse. Enfin, matplotlib est importé pour visualiser les résultats, notamment les caractéristiques les plus importantes.

#Le code lit ensuite deux fichiers CSV contenant les données : bank-full.csv et bank.csv. Le premier représente le jeu d’entraînement,
#c’est-à-dire celui sur lequel le modèle apprendra à reconnaître les comportements des clients. Le second constitue le jeu de test,
#qui servira plus tard à évaluer la performance du modèle. L’instruction pd.read_csv() charge ces fichiers dans deux DataFrames
#Pandas appelés donnee_train et donnee_test, en précisant que les valeurs sont séparées par des points-virgules (sep=';').
#Un aperçu des premières lignes du jeu d’entraînement est ensuite affiché avec head() afin de vérifier la structure des données 
#et le contenu des colonnes.

#Le script définit ensuite une petite fonction appelée donnee_propre(df). Celle-ci a pour but de nettoyer automatiquement 
#les données avant leur utilisation. Dans ce cas, elle remplace toutes les valeurs "unknown" (inconnues) par la chaîne "missing", 
#plus explicite et plus facile à traiter lors de l’encodage. La fonction fait une copie du DataFrame avant toute modification,
#pour ne pas altérer les données d’origine. Cette fonction est ensuite transformée en un objet scikit-learn grâce à FunctionTransformer, 
#ce qui permet de l’intégrer facilement à un pipeline d’apprentissage automatique. Ainsi, le nettoyage pourra s’exécuter automatiquement 
#sur tout nouveau jeu de données, sans intervention manuelle.

#Les étapes suivantes préparent les données pour l’entraînement. La colonne y, qui indique si le client a souscrit ou non à un dépôt à terme,
#est considérée comme la variable cible à prédire. On l’encode avec LabelEncoder afin de la convertir en valeurs numériques (par exemple,
#0 pour "no" et 1 pour "yes"). Ensuite, la colonne y est supprimée du reste du tableau pour former X_train et X_test, 
#les jeux de données contenant uniquement les variables explicatives (âge, métier, situation matrimoniale, solde, etc.).

#Le code détecte ensuite le type de chaque variable pour appliquer des transformations adaptées. Les colonnes contenant du texte (object)
#sont considérées comme catégorielles, tandis que les autres sont numériques. Cette distinction est importante : les données catégorielles doivent être converties
#en nombres avant d’être utilisées dans un modèle, tandis que les données numériques doivent être normalisées afin que leurs valeurs soient comparables.

#Le bloc suivant crée un préprocesseur automatique à l’aide de ColumnTransformer. Cet objet applique deux traitements différents selon le type de colonne:
#un MinMaxScaler sur les colonnes numériques pour ramener leurs valeurs entre 0 et 1, et un OneHotEncoder sur les colonnes catégorielles pour les transformer 
#en variables binaires (par exemple, "job = admin" devient une colonne prenant 1 ou 0). Grâce à l’option handle_unknown='ignore', 
#l’encodeur ignore les catégories inconnues dans le jeu de test, ce qui évite les erreurs.

#Une fois le nettoyage et le prétraitement définis, le script construit un pipeline complet regroupant toutes les étapes du traitement de bout en bout.
#Ce pipeline comprend quatre parties principales :

#Le nettoyage des données (cleaning),

#Le prétraitement (preprocessing),

#La sélection des caractéristiques (SelectKBest avec la méthode du chi-deux pour choisir les 15 variables les plus pertinentes),

#Et la modélisation avec un RandomForestClassifier, un algorithme d’ensemble basé sur plusieurs arbres de décision.

#Le pipeline est alors prêt à être entraîné. Avec la commande pipeline.fit(X_train, y_train), toutes les étapes sont exécutées dans le bon ordre :
#nettoyage → transformation → sélection → apprentissage. Cela garantit que le modèle est entraîné de manière cohérente, 
#sans fuite de données ni traitement manuel.

#Une fois entraîné, le pipeline est utilisé pour faire des prédictions automatiques sur le jeu de test (y_pred = pipeline.predict(X_test)).
#Cela permet d’évaluer la capacité du modèle à généraliser ses connaissances sur des données qu’il n’a jamais vues.

#Ensuite, le script analyse les caractéristiques les plus importantes utilisées par le modèle. Pour cela, 
#il récupère d’abord tous les noms de variables créés après le prétraitement, puis ne garde que ceux sélectionnés par SelectKBest. 
#Le modèle Random Forest attribue une importance à chaque variable selon son influence sur la prédiction. 
#Ces importances sont regroupées dans une série Pandas (pd.Series) et affichées dans un graphique à barres horizontales, 
#montrant les 15 variables ayant le plus d’impact sur la décision du modèle.

#Enfin, le code évalue la qualité du modèle à l’aide de plusieurs indicateurs. 
#La précision globale (accuracy) mesure la proportion de bonnes prédictions. 
#La matrice de confusion montre combien de cas “yes” et “no” ont été correctement ou incorrectement classés.
#Le rapport de classification donne des informations plus détaillées, notamment la précision, le rappel et le score F1 pour chaque classe, 
#ce qui permet de juger la performance du modèle de manière complète.

#En résumé, ce code met en place un pipeline d’apprentissage supervisé complet et automatique pour le dataset Bank Marketing.
#Il nettoie les données, les prépare, sélectionne les variables les plus utiles, entraîne un modèle robuste et 
#évalue ses performances sur un jeu de test indépendant. L’avantage principal de cette approche est qu’elle rend tout le processus reproductible,
#cohérent et automatisé, sans nécessiter d’interventions manuelles à chaque nouvelle exécution.
