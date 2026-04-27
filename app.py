from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import json
import plotly
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

DB_PATH = "data.csv"

# 20 patients pré-enregistrés
DATA = [
    {'age':25,'poids':70,'taille':175,'fumeur':0,'tension':12,'cholesterol':1.8,'glycemie':0.9,'activite':3,'region':'Centre'},
    {'age':30,'poids':60,'taille':165,'fumeur':0,'tension':11.5,'cholesterol':1.6,'glycemie':0.85,'activite':4,'region':'Centre'},
    {'age':45,'poids':85,'taille':180,'fumeur':1,'tension':14.5,'cholesterol':2.4,'glycemie':1.1,'activite':1,'region':'Littoral'},
    {'age':52,'poids':72,'taille':160,'fumeur':0,'tension':13.5,'cholesterol':2.2,'glycemie':1.0,'activite':2,'region':'Ouest'},
    {'age':35,'poids':90,'taille':185,'fumeur':1,'tension':13,'cholesterol':2.0,'glycemie':0.95,'activite':2,'region':'Nord'},
    {'age':28,'poids':55,'taille':162,'fumeur':0,'tension':11,'cholesterol':1.5,'glycemie':0.8,'activite':5,'region':'Sud'},
    {'age':60,'poids':95,'taille':170,'fumeur':1,'tension':16,'cholesterol':2.8,'glycemie':1.3,'activite':0,'region':'Extreme-Nord'},
    {'age':42,'poids':68,'taille':168,'fumeur':0,'tension':12.5,'cholesterol':1.9,'glycemie':0.9,'activite':3,'region':'Centre'},
    {'age':33,'poids':78,'taille':178,'fumeur':0,'tension':12,'cholesterol':1.7,'glycemie':0.85,'activite':4,'region':'Littoral'},
    {'age':48,'poids':82,'taille':163,'fumeur':1,'tension':14,'cholesterol':2.5,'glycemie':1.15,'activite':1,'region':'Nord-Ouest'},
    {'age':22,'poids':65,'taille':172,'fumeur':0,'tension':11.5,'cholesterol':1.4,'glycemie':0.8,'activite':5,'region':'Sud-Ouest'},
    {'age':55,'poids':75,'taille':158,'fumeur':0,'tension':13,'cholesterol':2.3,'glycemie':1.05,'activite':2,'region':'Est'},
    {'age':38,'poids':88,'taille':182,'fumeur':1,'tension':13.5,'cholesterol':2.1,'glycemie':1.0,'activite':2,'region':'Adamaoua'},
    {'age':27,'poids':58,'taille':170,'fumeur':0,'tension':11,'cholesterol':1.6,'glycemie':0.85,'activite':4,'region':'Centre'},
    {'age':50,'poids':92,'taille':176,'fumeur':1,'tension':15.5,'cholesterol':2.7,'glycemie':1.25,'activite':0,'region':'Littoral'},
    {'age':31,'poids':62,'taille':167,'fumeur':0,'tension':11.5,'cholesterol':1.7,'glycemie':0.9,'activite':3,'region':'Ouest'},
    {'age':58,'poids':98,'taille':175,'fumeur':1,'tension':16.5,'cholesterol':2.9,'glycemie':1.35,'activite':0,'region':'Nord'},
    {'age':40,'poids':70,'taille':165,'fumeur':0,'tension':12.5,'cholesterol':2.0,'glycemie':0.95,'activite':2,'region':'Sud'},
    {'age':36,'poids':80,'taille':180,'fumeur':0,'tension':12,'cholesterol':1.9,'glycemie':0.9,'activite':3,'region':'Extreme-Nord'},
    {'age':65,'poids':78,'taille':155,'fumeur':0,'tension':15,'cholesterol':2.6,'glycemie':1.2,'activite':1,'region':'Centre'}
]

def init_db():
    if not os.path.exists(DB_PATH):
        df = pd.DataFrame(DATA)
        df['imc'] = df.apply(lambda x: round(x['poids']/((x['taille']/100)**2),2), axis=1)
        df['categorie'] = df['imc'].apply(lambda i: 'Normal' if i<25 else ('Surpoids' if i<30 else 'Obèse'))
        df.to_csv(DB_PATH, index=False)
    return pd.read_csv(DB_PATH)

@app.route('/')
def index():
    df = pd.read_csv(DB_PATH) if os.path.exists(DB_PATH) else init_db()
    return render_template('index.html', nb=len(df))

@app.route('/collecte')
def collecte():
    return render_template('collecte.html')

@app.route('/save', methods=['POST'])
def save():
    d = {
        'age': int(request.form['age']),
        'poids': float(request.form['poids']),
        'taille': float(request.form['taille']),
        'fumeur': 1 if request.form.get('fumeur')=='on' else 0,
        'tension': float(request.form['tension']),
        'cholesterol': float(request.form['cholesterol']),
        'glycemie': float(request.form['glycemie']),
        'activite': int(request.form['activite']),
        'region': request.form['region']
    }
    d['imc'] = round(d['poids']/((d['taille']/100)**2), 2)
    d['categorie'] = 'Normal' if d['imc']<25 else ('Surpoids' if d['imc']<30 else 'Obèse')
    
    df = pd.read_csv(DB_PATH)
    df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
    df.to_csv(DB_PATH, index=False)
    return jsonify({"status":"success","imc":d['imc'],"categorie":d['categorie']})

@app.route('/analyses')
def analyses():
    df = pd.read_csv(DB_PATH) if os.path.exists(DB_PATH) else init_db()
    graphs = {}
    
    # 1. Régression simple
    fig1 = px.scatter(df, x='age', y='imc', trendline='ols', color='categorie', title='1. Régression Simple : Âge vs IMC')
    graphs['g1'] = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 2. K-Means
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(df[['age','imc']])
    df['cluster'] = df['cluster'].apply(lambda x: f'Groupe {x+1}')
    fig2 = px.scatter(df, x='age', y='imc', color='cluster', title='2. K-Means Clustering')
    graphs['g2'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 3. PCA
    X = df[['age','imc','tension','cholesterol']].dropna()
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)
    pdf = pd.DataFrame(Xp, columns=['PC1','PC2'])
    pdf['cat'] = df.loc[X.index, 'categorie'].values
    fig3 = px.scatter(pdf, x='PC1', y='PC2', color='cat', title='3. ACP - Réduction de dimension')
    graphs['g3'] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 4. Boxplot
    fig4 = px.box(df, x='region', y='imc', title='4. IMC par Région')
    graphs['g4'] = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 5. Corrélation
    corr = df[['age','poids','taille','imc','tension','cholesterol','glycemie']].corr()
    fig5 = px.imshow(corr, text_auto='.2f', title='5. Matrice de Corrélation', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    graphs['g5'] = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 6. Histogramme
    fig6 = px.histogram(df, x='imc', title='6. Distribution de l\'IMC')
    graphs['g6'] = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)
    
    st = {'nb':len(df), 'imc_moy':round(df['imc'].mean(),2), 'age_moy':round(df['age'].mean(),1)}
    
    return render_template('analyses.html', graphs=graphs, stats=st)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
