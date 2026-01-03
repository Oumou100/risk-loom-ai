import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="RiskLoom | Prédiction de Risque de Crédit",
    page_icon="favicon.png" if Path("favicon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    
    .main-header {
        background: linear-gradient(135deg, #844226 0%, #A65D3F 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.5rem;
        box-shadow: 0 4px 15px rgba(132, 66, 38, 0.3);
    }
    
    .main-header img {
        width: 80px;
        height: 80px;
        border-radius: 10px;
    }
    
    .header-text h1 { color: white; font-size: 2.5rem; margin: 0; }
    .header-text p { color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0; }
    
    .metric-card {
        background: #F5F0ED;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #844226;
        margin-bottom: 1rem;
    }
    
    .result-good {
        background: linear-gradient(145deg, #2E7D32 0%, #43A047 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
    }
    
    .result-bad {
        background: linear-gradient(145deg, #C62828 0%, #E53935 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(198, 40, 40, 0.3);
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #844226 0%, #A65D3F 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(132, 66, 38, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(132, 66, 38, 0.4);
    }
    
    .info-box {
        background: #F5F0ED;
        border: 1px solid #844226;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #5D4037;
        font-size: 0.9rem;
        border-top: 2px solid #844226;
        margin-top: 2rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #844226 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = Path("best_model_xgboost.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    st.error("Fichier modèle introuvable!")
    return None


@st.cache_resource
def load_encoders():
    encoder_files = {
        "Sex": "Sex_encoder.pkl",
        "Housing": "Housing_encoder.pkl",
        "Saving accounts": "Saving accounts_encoder.pkl",
        "Checking account": "Checking account_encoder.pkl",
        "Purpose": "Purpose_encoder.pkl"
    }
    
    encoders = {}
    for col, filename in encoder_files.items():
        path = Path(filename)
        if path.exists():
            encoders[col] = joblib.load(path)
        else:
            alt_path = Path(f"{col}_encoder (1).pkl")
            if alt_path.exists():
                encoders[col] = joblib.load(alt_path)
    
    return encoders


import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_html = ""
if Path("OumouIT.png").exists():
    logo_b64 = get_base64_image("OumouIT.png")
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="Logo">'

st.markdown(f"""
<div class="main-header">
    {logo_html}
    <div class="header-text">
        <h1>RiskLoom</h1>
        <p>Modélisation Intelligente du Risque de Crédit</p>
    </div>
</div>
""", unsafe_allow_html=True)

model = load_model()
encoders = load_encoders()

with st.sidebar:
    st.markdown("## Informations du Demandeur")
    st.markdown("---")
    
    st.markdown("### Données Personnelles")
    age = st.slider("Âge", min_value=18, max_value=75, value=35)
    sex = st.selectbox("Genre", ["male", "female"], format_func=lambda x: "Homme" if x == "male" else "Femme")
    
    st.markdown("### Logement & Emploi")
    housing = st.selectbox("Logement", ["own", "rent", "free"],
                          format_func=lambda x: {"own": "Propriétaire", "rent": "Locataire", "free": "Gratuit"}[x])
    job = st.selectbox("Niveau d'Emploi", [0, 1, 2, 3], 
                       format_func=lambda x: {0: "Non qualifié (non résident)", 
                                              1: "Non qualifié (résident)", 
                                              2: "Qualifié", 
                                              3: "Hautement qualifié"}[x])
    
    st.markdown("### Situation Financière")
    saving_accounts = st.selectbox("Compte Épargne", ["little", "moderate", "quite rich", "rich"],
                                   format_func=lambda x: {"little": "Faible", "moderate": "Modéré", 
                                                          "quite rich": "Assez élevé", "rich": "Élevé"}[x])
    checking_account = st.selectbox("Compte Courant", ["little", "moderate", "rich"],
                                    format_func=lambda x: {"little": "Faible", "moderate": "Modéré", "rich": "Élevé"}[x])
    
    st.markdown("### Détails du Crédit")
    credit_amount = st.number_input("Montant du Crédit (EUR)", min_value=250, max_value=20000, value=3000, step=100)
    duration = st.slider("Durée (mois)", min_value=4, max_value=72, value=24)
    purpose = st.selectbox("Objet du Crédit", 
                          ["car", "radio/TV", "furniture/equipment", "business", 
                           "education", "repairs", "vacation/others", "domestic appliances"],
                          format_func=lambda x: {"car": "Voiture", "radio/TV": "Radio/TV", 
                                                 "furniture/equipment": "Mobilier/Équipement",
                                                 "business": "Entreprise", "education": "Éducation",
                                                 "repairs": "Réparations", "vacation/others": "Vacances/Autres",
                                                 "domestic appliances": "Électroménager"}[x])
    
    st.markdown("---")
    predict_button = st.button("Prédire le Risque", use_container_width=True)


col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Résumé de la Demande")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    housing_labels = {"own": "Propriétaire", "rent": "Locataire", "free": "Gratuit"}
    purpose_labels = {"car": "Voiture", "radio/TV": "Radio/TV", "furniture/equipment": "Mobilier",
                     "business": "Entreprise", "education": "Éducation", "repairs": "Réparations",
                     "vacation/others": "Vacances", "domestic appliances": "Électroménager"}
    
    with summary_col1:
        st.metric("Montant du Crédit", f"{credit_amount:,} EUR")
        st.metric("Âge", f"{age} ans")
    
    with summary_col2:
        st.metric("Durée", f"{duration} mois")
        st.metric("Mensualité", f"{credit_amount/duration:,.0f} EUR")
    
    with summary_col3:
        st.metric("Objet", purpose_labels.get(purpose, purpose.title()))
        st.metric("Logement", housing_labels.get(housing, housing.title()))

with col2:
    st.markdown("### Facteurs de Risque")
    st.info("""
    **Facteurs clés analysés :**
    - Montant vs Durée du crédit
    - Soldes des comptes
    - Stabilité de l'emploi
    - Âge & Situation de logement
    """)


if predict_button and model is not None:
    st.markdown("---")
    st.markdown("## Résultat de la Prédiction")
    
    try:
        credit_per_month = credit_amount / duration
        is_high_amount = 1 if credit_amount > 2500 else 0
        
        sex_encoded = encoders["Sex"].transform([sex])[0] if "Sex" in encoders else 0
        housing_encoded = encoders["Housing"].transform([housing])[0] if "Housing" in encoders else 0
        saving_encoded = encoders["Saving accounts"].transform([saving_accounts])[0] if "Saving accounts" in encoders else 0
        checking_encoded = encoders["Checking account"].transform([checking_account])[0] if "Checking account" in encoders else 0
        purpose_encoded = encoders["Purpose"].transform([purpose])[0] if "Purpose" in encoders else 0
        
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex_encoded],
            'Job': [job],
            'Housing': [housing_encoded],
            'Saving accounts': [saving_encoded],
            'Checking account': [checking_encoded],
            'Credit amount': [credit_amount],
            'Duration': [duration],
            'Purpose': [purpose_encoded],
            'Credit_per_Month': [credit_per_month],
            'Is_High_Amount': [is_high_amount]
        })
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            if prediction == 1:
                st.markdown("""
                <div class="result-good">
                    <h2 style="color: white; margin-bottom: 0.5rem;">RISQUE FAIBLE</h2>
                    <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem;">Crédit Approuvé</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown("""
                <div class="result-bad">
                    <h2 style="color: white; margin-bottom: 0.5rem;">RISQUE ÉLEVÉ</h2>
                    <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem;">Crédit Refusé</p>
                </div>
                """, unsafe_allow_html=True)
        
        with result_col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilité d'Approbation", 'font': {'size': 16, 'color': '#1A1A1A'}},
                number={'suffix': '%', 'font': {'size': 36, 'color': '#844226'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#1A1A1A'},
                    'bar': {'color': '#844226'},
                    'bgcolor': '#F5F0ED',
                    'borderwidth': 2,
                    'bordercolor': '#844226',
                    'steps': [
                        {'range': [0, 40], 'color': '#FFCDD2'},
                        {'range': [40, 60], 'color': '#FFE0B2'},
                        {'range': [60, 100], 'color': '#C8E6C9'}
                    ],
                    'threshold': {
                        'line': {'color': '#844226', 'width': 4},
                        'thickness': 0.75,
                        'value': probability[1] * 100
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=250,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Analyse des Risques")
        
        risk_factors = {
            'Risque Durée': min(duration / 72 * 100, 100),
            'Risque Montant': min(credit_amount / 15000 * 100, 100),
            'Facteur Âge': 100 - min(abs(age - 35) / 40 * 100, 100),
            'État des Comptes': (saving_encoded + checking_encoded) / 6 * 100
        }
        
        fig_bar = px.bar(
            x=list(risk_factors.values()),
            y=list(risk_factors.keys()),
            orientation='h',
            color=list(risk_factors.values()),
            color_continuous_scale=['#C8E6C9', '#FFE0B2', '#FFCDD2']
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Niveau de Risque (%)",
            yaxis_title="",
            showlegend=False,
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(color='#1A1A1A')
        )
        fig_bar.update_xaxes(gridcolor='rgba(132, 66, 38, 0.2)')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur de prédiction: {str(e)}")
        st.info("Veuillez vérifier que tous les fichiers d'encodeurs sont présents.")


st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>RiskLoom</strong> - Modélisation Intelligente du Risque de Crédit</p>
    <p>Développé par <strong>Koné Oumou</strong></p>
</div>
""", unsafe_allow_html=True)
