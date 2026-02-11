import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Configuration de la page
st.set_page_config(
    page_title="Cancer Prediction App",
    page_icon="🏥",
    layout="wide"
)

# CSS personnalisé pour un design attrayant
st.markdown("""
    <style>
        /* Arrière-plan principal */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        }
        
        /* Texte principal - plus lisible */
        body, p, span, label {
            color: #1a202c !important;
        }
        
        /* Titre principal */
        h1 {
            color: #2d3748;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        
        /* Sous-titres */
        h2, h3 {
            color: #1a202c;
            font-weight: 600;
        }
        
        /* Texte général */
        .stMarkdown {
            color: #2d3748 !important;
        }
        
        /* Metric cards */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.2);
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        [data-testid="metric-container"] label {
            color: rgba(255, 255, 255, 0.9) !important;
        }
        
        [data-testid="metric-container"] div {
            color: white !important;
        }
        
        /* Bouton */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
        }
        
        /* Input fields */
        .stNumberInput > div > input {
            border-radius: 8px;
            border: 2px solid #667eea;
            padding: 10px;
            color: #1a202c !important;
        }
        
        .stNumberInput label {
            color: #2d3748 !important;
            font-weight: 500;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        [data-testid="stSidebar"] label {
            color: white !important;
        }
        
        /* Messages */
        .stSuccess {
            background-color: rgba(72, 187, 120, 0.1);
            border: 2px solid #48bb78;
            border-radius: 8px;
            padding: 12px;
            color: #22543d !important;
        }
        
        .stError {
            background-color: rgba(245, 101, 101, 0.1);
            border: 2px solid #f56565;
            border-radius: 8px;
            padding: 12px;
            color: #742a2a !important;
        }
        
        .stInfo {
            background-color: rgba(102, 126, 234, 0.1);
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 12px;
            color: #2d3748 !important;
        }
        
        /* Divider */
        hr {
            border: 2px solid #667eea;
            margin: 20px 0;
        }
        
        /* Write text */
        .stWrite {
            color: #2d3748 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.title("🏥 Application de Prédiction du Cancer")
st.write("Prédiction du cancer du sein basée sur l'apprentissage automatique")

# Charger les données
df= pd.read_csv("cancer_cleanned.csv")

X = df.drop(columns=['diagnosis(1=m, 0=b)'])
y = df['diagnosis(1=m, 0=b)']

# Entraîner le modèle
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, scaler, accuracy, precision, recall, f1, X_test_scaled, y_test

model, scaler, accuracy, precision, recall, f1, X_test, y_test = train_model()

# Barre latérale
st.sidebar.image("Cancer1.jpg", use_container_width= True, width=100)
st.sidebar.header("🌐 Navigation")
page = st.sidebar.radio("Choisir une page", ["🏠 Accueil", "🔬 Prédiction", "📈 Statistiques", "📋 À propos"])

if page == "🏠 Accueil":
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.header("🎯 Objectif de l'application")
        st.write("""
        Cette application vise à **aider à la détection précoce du cancer du sein**
        en utilisant un **algorithme d'apprentissage automatique (Random Forest)** basé sur des
        **variables cliniques médicalement pertinentes**.
        """)
        st.header("🩺 Public cible")
        st.write("""
        - Médecins
        - Chercheurs
        - Étudiants en data science & santé
        """)

    with col2:
        st.header("📊 Données utilisées")
        st.write("""
        Les données proviennent d’un jeu de données médical 
        décrivant les caractéristiques morphologiques des tumeurs.
        """)
        st.header("🧠 Méthodologie")
        st.write("""
        - Sélection de variables cliniques
        - Standardisation
        - Random Forest (100 arbres, profondeur max 10)
        - Prédiction avec probabilité
        """)

    st.markdown("---")
    st.info("⚠️ Cette application est un outil d’aide à la décision et ne remplace pas un diagnostic médical.")

elif page == "🔬 Prédiction":
    st.header("🔬 Faire une Prédiction")
    
    st.write("Entrez les valeurs des caractéristiques pour prédire si le cancer est bénin ou malin.")
    
    # Créer les inputs
    feature_names = X.columns.tolist()
    input_dict = {}
    
    cols = st.columns(3)
    
    for i, feature in enumerate(feature_names):
        col = cols[i % 3]
        
        min_val = X[feature].min()
        max_val = X[feature].max()
        mean_val = X[feature].mean()
        
        with col:
            value = st.number_input(
                label=feature,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=(max_val - min_val) / 100,
                key=feature
            )
            input_dict[feature] = value
    
    # Bouton de prédiction
    if st.button("🎯 Faire la Prédiction", use_container_width=True):
        # Créer l'array dans le bon ordre
        input_values = [input_dict[feature] for feature in feature_names]
        input_array = np.array([input_values])
        input_scaled = scaler.transform(input_array)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.success("**BÉNIN (0)** - Tumeur bénigne", icon="✅")
            else:
                st.error("**MALIN (1)** - Tumeur maligne", icon="⚠️")
        
        with col2:
            st.info(f"Confiance: {max(probability)*100:.2f}%")
        
        st.write("**Probabilités:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bénin (0)", f"{probability[0]:.2%}")
        with col2:
            st.metric("Malin (1)", f"{probability[1]:.2%}")

elif page == "📈 Statistiques":
    st.header("📈 Statistiques du Modèle")
    st.subheader("Distribution des Classes")
    class_counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(['Bénin (0)', 'Malin (1)'], class_counts.values, color=['green', 'red'])
    ax.set_ylabel("Nombre de cas")
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Importance des Features")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

elif page == "📋 À propos":
    st.header("📋 À propos de l'Application")
    
    st.write("""

    ### 📊 Données
    - **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
    - **Nombre d'échantillons:** 569
    - **Nombre de features:** 30
    - **Classes:** 0 (Bénin) et 1 (Malin)
    
    ### 🤖 Modèle
    - **Algorithme:** Random Forest Classifier
    - **Nombre d'arbres:** 100
    - **Profondeur maximale:** 10
    - **Train/Test Split:** 80/20
    
    
    ### 🛠️ Stack technologique

    - Python  
    - Streamlit  
    - Scikit-learn  
    - Pandas / NumPy  
    - Matplotlib  
    - Seaborn
             
    ### 👨‍💼 Chef de projet
    - **Nom :**  Ahmat Mahamat Abdel-Aziz HABIB  
    - **Rôle :** Data Scientist / Analyste  
             
    ### 📬 Contact
    - 📧 Email : habib.ahmat@email.com  
    - 📞 Téléphone : +221 78 752 75 78 
    - 🔗 LinkedIn : www.linkedin.com/in/habib-ahmat
            """)

    st.markdown("---")
    st.success("Merci d'utiliser cette application 🙏")






