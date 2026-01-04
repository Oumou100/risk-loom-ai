import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

PRIMARY_COLOR = "#844226"
SECONDARY_COLOR = "#A65D3F"

CURRENCY_CONFIG = {
    "EUR": {"rate": 1.0, "symbol": "€", "label": "Euro (€)", "min": 250, "max": 20000, "default": 3000, "step": 100},
    "MAD": {"rate": 10.85, "symbol": "DH", "label": "Dirham Marocain (DH)", "min": 2500, "max": 220000, "default": 33000, "step": 1000},
    "XOF": {"rate": 655.96, "symbol": "CFA", "label": "Franc CFA", "min": 165000, "max": 13000000, "default": 2000000, "step": 50000}
}

LABELS = {
    "housing": {"own": "Propriétaire", "rent": "Locataire", "free": "Gratuit"},
    "purpose": {"car": "Voiture", "radio/TV": "Radio/TV", "furniture/equipment": "Mobilier/Équipement",
                "business": "Entreprise", "education": "Éducation", "repairs": "Réparations",
                "vacation/others": "Vacances/Autres", "domestic appliances": "Électroménager"},
    "job": {0: "Non qualifié (non résident)", 1: "Non qualifié (résident)", 2: "Qualifié", 3: "Hautement qualifié"},
    "saving": {"little": "Faible", "moderate": "Modéré", "quite rich": "Assez élevé", "rich": "Élevé"},
    "checking": {"little": "Faible", "moderate": "Modéré", "rich": "Élevé"}
}

# =============================================================================
# PAGE CONFIG & STATE
# =============================================================================

st.set_page_config(
    page_title="RiskLoom | Prédiction de Risque de Crédit",
    page_icon="favicon.png" if Path("favicon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# =============================================================================
# STYLES CSS
# =============================================================================

st.markdown(f"""
<style>
    .main {{ padding: 0rem 1rem; }}
    
    .main-header {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(132, 66, 38, 0.3);
    }}
    .header-content {{ display: flex; align-items: center; justify-content: center; gap: 1.5rem; }}
    .header-text h1 {{ color: white; font-size: 2.5rem; margin: 0; font-weight: 700; }}
    .header-text p {{ color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0.3rem 0 0 0; }}
    
    .result-good {{
        background: linear-gradient(145deg, #2E7D32 0%, #43A047 100%);
        padding: 2rem; border-radius: 15px; text-align: center;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
    }}
    .result-bad {{
        background: linear-gradient(145deg, #C62828 0%, #E53935 100%);
        padding: 2rem; border-radius: 15px; text-align: center;
        box-shadow: 0 4px 15px rgba(198, 40, 40, 0.3);
    }}
    
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%) !important;
        color: white !important; border: none; padding: 0.75rem 2rem;
        font-size: 1.1rem; font-weight: 600; border-radius: 10px;
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(132, 66, 38, 0.3);
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(132, 66, 38, 0.4);
        color: white !important;
    }}
    
    .stDownloadButton > button {{
        width: 100%;
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%) !important;
        color: white !important; border: none; padding: 0.75rem 2rem;
        font-size: 1rem; font-weight: 600; border-radius: 10px;
        transition: all 0.3s ease; box-shadow: 0 2px 6px rgba(132, 66, 38, 0.2);
    }}
    .stDownloadButton > button:hover {{
        box-shadow: 0 3px 10px rgba(132, 66, 38, 0.3);
        color: white !important;
    }}
    
    .stButton > button:active, .stButton > button:focus,
    .stDownloadButton > button:active, .stDownloadButton > button:focus {{
        color: white !important;
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%) !important;
    }}
    
    .conversion-display {{
        background: #F5F0ED; border: 2px solid {PRIMARY_COLOR};
        border-radius: 10px; padding: 0.8rem; text-align: center; margin: 0.5rem 0;
    }}
    
    .footer {{
        text-align: center; padding: 2rem; color: #5D4037;
        font-size: 0.9rem; border-top: 2px solid {PRIMARY_COLOR}; margin-top: 2rem;
    }}
    .footer a {{ color: {PRIMARY_COLOR}; text-decoration: none; margin: 0 0.5rem; font-weight: 500; }}
    .footer a:hover {{ text-decoration: underline; }}
    .social-links {{ margin-top: 1rem; }}
    .social-links a {{ display: inline-block; margin: 0 0.8rem; font-size: 1.5rem; }}
    
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    
    .stSelectbox label, .stSlider label, .stNumberInput label {{
        color: {PRIMARY_COLOR} !important; font-weight: 500;
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

@st.cache_resource
def load_model():
    model_path = Path("best_model_xgboost.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    st.error("Fichier modèle introuvable!")
    return None


@st.cache_resource
def load_encoders():
    encoder_names = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    encoders = {}
    for col in encoder_names:
        for pattern in [f"{col}_encoder.pkl", f"{col}_encoder (1).pkl"]:
            path = Path(pattern)
            if path.exists():
                encoders[col] = joblib.load(path)
                break
    return encoders


def generate_csv_report(data, prediction, probability, currency_symbol):
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    result_text = "RISQUE FAIBLE - Credit Approuve" if prediction == 1 else "RISQUE ELEVE - Credit Refuse"
    risk_duration = "Eleve" if data['duration'] > 48 else "Modere" if data['duration'] > 24 else "Faible"
    risk_amount = "Eleve" if data['credit_amount_eur'] > 10000 else "Modere" if data['credit_amount_eur'] > 5000 else "Faible"
    
    # BOM UTF-8 pour Excel + contenu sans accents pour compatibilité maximale
    return f"""\ufeffChamp;Valeur
Date evaluation;{date_str}
Age;{data['age']} ans
Genre;{data['sex_label']}
Logement;{data['housing_label']}
Niveau emploi;{data['job_label']}
Compte epargne;{data['saving_label']}
Compte courant;{data['checking_label']}
Montant demande;{data['credit_amount_local']:.0f} {currency_symbol}
Equivalent EUR;{data['credit_amount_eur']:.0f} EUR
Duree;{data['duration']} mois
Mensualite estimee;{data['credit_amount_local']/data['duration']:.0f} {currency_symbol}
Objet du credit;{data['purpose_label']}
Decision;{result_text}
Probabilite approbation;{probability[1]*100:.1f}%
Probabilite refus;{probability[0]*100:.1f}%
Risque duree;{risk_duration}
Risque montant;{risk_amount}
"""


def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability[1] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité d'Approbation", 'font': {'size': 16, 'color': '#1A1A1A'}},
        number={'suffix': '%', 'font': {'size': 36, 'color': PRIMARY_COLOR}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#1A1A1A'},
            'bar': {'color': PRIMARY_COLOR},
            'bgcolor': '#F5F0ED',
            'borderwidth': 2,
            'bordercolor': PRIMARY_COLOR,
            'steps': [
                {'range': [0, 40], 'color': '#FFCDD2'},
                {'range': [40, 60], 'color': '#FFE0B2'},
                {'range': [60, 100], 'color': '#C8E6C9'}
            ],
            'threshold': {'line': {'color': PRIMARY_COLOR, 'width': 4}, 'thickness': 0.75, 'value': probability[1] * 100}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_risk_chart(risk_factors):
    fig = px.bar(x=list(risk_factors.values()), y=list(risk_factors.keys()), orientation='h',
                 color=list(risk_factors.values()), color_continuous_scale=['#C8E6C9', '#FFE0B2', '#FFCDD2'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis_title="Niveau de Risque (%)", yaxis_title="", showlegend=False,
                      height=200, margin=dict(l=0, r=0, t=10, b=0), font=dict(color='#1A1A1A'))
    fig.update_xaxes(gridcolor='rgba(132, 66, 38, 0.2)')
    return fig

# =============================================================================
# CHARGEMENT DES RESSOURCES
# =============================================================================

model = load_model()
encoders = load_encoders()

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <div class="header-content">
        <div class="header-text">
            <h1>RiskLoom</h1>
            <p>Modélisation Intelligente du Risque de Crédit</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Comment utiliser cette application ?", expanded=False):
    st.markdown("""
    ### Guide d'utilisation
    
    **RiskLoom** est un outil d'aide à la décision pour l'évaluation du risque de crédit.
    
    #### Étapes :
    1. **Remplissez les informations** du demandeur dans la barre latérale gauche
    2. **Sélectionnez votre devise** (EUR, MAD ou CFA) et ajustez le taux si nécessaire
    3. **Cliquez sur "Prédire le Risque"** pour obtenir l'évaluation
    
    #### Résultats :
    - **RISQUE FAIBLE** (vert) : Le profil est favorable, crédit recommandé
    - **RISQUE ÉLEVÉ** (rouge) : Le profil présente des risques, crédit déconseillé
    
    #### Téléchargement :
    Après la prédiction, vous pouvez **télécharger un rapport CSV** contenant toutes les informations.
    
    #### Facteurs analysés :
    | Facteur | Impact |
    |---------|--------|
    | Montant du crédit | Plus le montant est élevé, plus le risque augmente |
    | Durée | Les crédits longs sont plus risqués |
    | Comptes épargne/courant | Des soldes élevés réduisent le risque |
    | Âge | L'expérience financière compte |
    | Type d'emploi | La stabilité professionnelle est importante |
    """)

# =============================================================================
# SIDEBAR - FORMULAIRE
# =============================================================================

with st.sidebar:
    if Path("OumouIT.png").exists():
        st.image("OumouIT.png", width=120)
    
    st.markdown("## Informations du Demandeur")
    st.markdown("---")
    
    st.markdown("### Données Personnelles")
    age = st.slider("Âge", min_value=18, max_value=75, value=35)
    sex = st.selectbox("Genre", ["male", "female"], format_func=lambda x: "Homme" if x == "male" else "Femme")
    
    st.markdown("### Logement & Emploi")
    housing = st.selectbox("Logement", list(LABELS["housing"].keys()), format_func=lambda x: LABELS["housing"][x])
    job = st.selectbox("Niveau d'Emploi", list(LABELS["job"].keys()), format_func=lambda x: LABELS["job"][x])
    
    st.markdown("### Situation Financière")
    saving_accounts = st.selectbox("Compte Épargne", list(LABELS["saving"].keys()), format_func=lambda x: LABELS["saving"][x])
    checking_account = st.selectbox("Compte Courant", list(LABELS["checking"].keys()), format_func=lambda x: LABELS["checking"][x])
    
    st.markdown("### Détails du Crédit")
    currency = st.selectbox("Devise", list(CURRENCY_CONFIG.keys()), format_func=lambda x: CURRENCY_CONFIG[x]["label"])
    
    curr = CURRENCY_CONFIG[currency]
    if currency == "MAD":
        exchange_rate = st.number_input("Taux EUR → MAD", min_value=1.0, max_value=20.0, value=curr["rate"], step=0.01)
    elif currency == "XOF":
        exchange_rate = st.number_input("Taux EUR → CFA", min_value=100.0, max_value=1000.0, value=curr["rate"], step=0.01)
    else:
        exchange_rate = curr["rate"]
    
    symbol = curr["symbol"]
    credit_amount_local = st.number_input(f"Montant du Crédit ({symbol})", min_value=curr["min"], max_value=curr["max"], value=curr["default"], step=curr["step"])
    credit_amount_eur = credit_amount_local / exchange_rate
    
    if currency != "EUR":
        st.markdown(f'<div class="conversion-display"><strong>{credit_amount_local:,.0f} {symbol}</strong><br><span style="color: {PRIMARY_COLOR};">≈ {credit_amount_eur:,.0f} EUR</span></div>', unsafe_allow_html=True)
    
    duration = st.slider("Durée (mois)", min_value=4, max_value=72, value=24)
    purpose = st.selectbox("Objet du Crédit", list(LABELS["purpose"].keys()), format_func=lambda x: LABELS["purpose"][x])
    
    st.markdown("---")
    predict_button = st.button("Prédire le Risque", use_container_width=True)

# =============================================================================
# CONTENU PRINCIPAL - RÉSUMÉ
# =============================================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Résumé de la Demande")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Montant du Crédit", f"{credit_amount_local:,.0f} {symbol}")
        if currency != "EUR":
            st.caption(f"≈ {credit_amount_eur:,.0f} EUR")
        st.metric("Âge", f"{age} ans")
    
    with summary_col2:
        st.metric("Durée", f"{duration} mois")
        st.metric("Mensualité", f"{credit_amount_local / duration:,.0f} {symbol}")
    
    with summary_col3:
        st.metric("Objet", LABELS["purpose"].get(purpose, purpose))
        st.metric("Logement", LABELS["housing"].get(housing, housing))

with col2:
    st.markdown("### Facteurs de Risque")
    st.info("**Facteurs clés analysés :**\n- Montant vs Durée du crédit\n- Soldes des comptes\n- Stabilité de l'emploi\n- Âge & Situation de logement")

# =============================================================================
# PRÉDICTION
# =============================================================================

if predict_button and model is not None:
    try:
        sex_encoded = encoders["Sex"].transform([sex])[0] if "Sex" in encoders else 0
        housing_encoded = encoders["Housing"].transform([housing])[0] if "Housing" in encoders else 0
        saving_encoded = encoders["Saving accounts"].transform([saving_accounts])[0] if "Saving accounts" in encoders else 0
        checking_encoded = encoders["Checking account"].transform([checking_account])[0] if "Checking account" in encoders else 0
        purpose_encoded = encoders["Purpose"].transform([purpose])[0] if "Purpose" in encoders else 0
        
        input_data = pd.DataFrame({
            'Age': [age], 'Sex': [sex_encoded], 'Job': [job], 'Housing': [housing_encoded],
            'Saving accounts': [saving_encoded], 'Checking account': [checking_encoded],
            'Credit amount': [credit_amount_eur], 'Duration': [duration], 'Purpose': [purpose_encoded],
            'Credit_per_Month': [credit_amount_eur / duration],
            'Is_High_Amount': [1 if credit_amount_eur > 2500 else 0]
        })
        
        st.session_state.prediction_result = {
            'prediction': model.predict(input_data)[0],
            'probability': model.predict_proba(input_data)[0],
            'report_data': {
                'age': age, 'sex_label': "Homme" if sex == "male" else "Femme",
                'housing_label': LABELS["housing"].get(housing), 'job_label': LABELS["job"].get(job),
                'saving_label': LABELS["saving"].get(saving_accounts), 'checking_label': LABELS["checking"].get(checking_account),
                'credit_amount_local': credit_amount_local, 'credit_amount_eur': credit_amount_eur,
                'duration': duration, 'purpose_label': LABELS["purpose"].get(purpose)
            },
            'symbol': symbol,
            'saving_encoded': saving_encoded,
            'checking_encoded': checking_encoded
        }
    except Exception as e:
        st.error(f"Erreur de prédiction: {str(e)}")

# =============================================================================
# AFFICHAGE DES RÉSULTATS
# =============================================================================

if st.session_state.prediction_result is not None:
    r = st.session_state.prediction_result
    
    st.markdown("---")
    st.markdown("## Résultat de la Prédiction")
    
    result_col1, result_col2 = st.columns([1, 1])
    
    with result_col1:
        if r['prediction'] == 1:
            st.markdown('<div class="result-good"><h2 style="color: white; margin-bottom: 0.5rem;">RISQUE FAIBLE</h2><p style="color: rgba(255,255,255,0.9); font-size: 1.2rem;">Crédit Approuvé</p></div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown('<div class="result-bad"><h2 style="color: white; margin-bottom: 0.5rem;">RISQUE ÉLEVÉ</h2><p style="color: rgba(255,255,255,0.9); font-size: 1.2rem;">Crédit Refusé</p></div>', unsafe_allow_html=True)
    
    with result_col2:
        st.plotly_chart(create_gauge_chart(r['probability']), use_container_width=True)
    
    st.download_button(
        label="Télécharger le rapport (CSV)",
        data=generate_csv_report(r['report_data'], r['prediction'], r['probability'], r['symbol']),
        file_name=f"riskloom_rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("### Analyse des Risques")
    risk_factors = {
        'Risque Durée': min(r['report_data']['duration'] / 72 * 100, 100),
        'Risque Montant': min(r['report_data']['credit_amount_eur'] / 15000 * 100, 100),
        'Facteur Âge': 100 - min(abs(r['report_data']['age'] - 35) / 40 * 100, 100),
        'État des Comptes': (r['saving_encoded'] + r['checking_encoded']) / 6 * 100
    }
    st.plotly_chart(create_risk_chart(risk_factors), use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<div class="footer">
    <p><strong>RiskLoom</strong> - Modélisation Intelligente du Risque de Crédit</p>
    <p>Développé par <strong>Koné Oumou</strong></p>
    <div class="social-links">
        <a href="https://oumou100.github.io/" target="_blank" title="Portfolio"><i class="fas fa-globe" style="color: #844226;"></i></a>
        <a href="https://www.linkedin.com/in/kon%C3%A9-oumou-98bb6229a/" target="_blank" title="LinkedIn"><i class="fab fa-linkedin" style="color: #0A66C2;"></i></a>
        <a href="https://github.com/Oumou100" target="_blank" title="GitHub"><i class="fab fa-github" style="color: #333;"></i></a>
    </div>
</div>
""", unsafe_allow_html=True)
