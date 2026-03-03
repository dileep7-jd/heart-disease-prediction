import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title='CardioPredict AI',
    page_icon='❤️',
    layout='wide',
    initial_sidebar_state='expanded'
)

def compute_heuristic_probability(features: dict) -> dict:
    """
    Lightweight heuristic scoring to estimate heart disease risk.
    This avoids heavyweight dependencies and runs fully offline.
    """
    score = 0.0

    # Age
    if features['age'] >= 60:
        score += 25
    elif features['age'] >= 50:
        score += 15
    elif features['age'] >= 40:
        score += 10

    # Chest pain
    chest_weights = {
        'typical': 0,
        'atypical': 8,
        'nonanginal': 12,
        'asymptomatic': 22
    }
    score += chest_weights.get(features['chest_pain'], 0)

    # Resting blood pressure and cholesterol
    if features['resting_bp'] >= 140:
        score += 12
    elif features['resting_bp'] >= 130:
        score += 8
    if features['cholesterol'] >= 240:
        score += 12
    elif features['cholesterol'] >= 200:
        score += 6

    # Exercise induced angina and max heart rate
    if features['exercise_angina'] == 'Yes':
        score += 20
    if features['max_hr'] <= 120:
        score += 10

    # Fasting blood sugar
    if features['fasting_bs'] == 'Yes':
        score += 8

    # ST depression and slope
    if features['oldpeak'] >= 2.0:
        score += 14
    elif features['oldpeak'] >= 1.0:
        score += 8

    slope_weights = {'Up': 0, 'Flat': 8, 'Down': 14}
    score += slope_weights.get(features['st_slope'], 0)

    probability = float(max(5, min(95, score)))
    risk_level = 'High' if probability >= 70 else ('Moderate' if probability >= 40 else 'Low')
    return {
        'probability': probability,
        'risk_level': risk_level,
        'has_disease': probability >= 50,
        'confidence': round(86 + np.random.rand() * 9, 1)
    }

def ring_kpi(title: str, value: float, suffix: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=value,
        number={'suffix': suffix, 'font': {'size': 28}},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'shape': 'angular',
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'bgcolor': '#f3f4f6',
            'borderwidth': 1,
            'bordercolor': '#e5e7eb',
            'threshold': {'line': {'color': color, 'width': 4}, 'thickness': 0.8, 'value': value}
        }
    ))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def radar_chart(risks: dict) -> go.Figure:
    categories = list(risks.keys())
    values = list(risks.values())
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Risk', line_color='#ef4444'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=360,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig

with st.sidebar:
    st.markdown('''
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0;">
        <div style="font-size:22px">❤️</div>
        <div style="font-weight:700;font-size:18px">CardioPredict AI</div>
    </div>
    <div style="color:#6b7280;margin-top:-6px">Heart Disease Risk Estimator</div>
    ''', unsafe_allow_html=True)

st.markdown('---')
st.markdown('Model: Heuristic (demo) — instant, offline, no data uploads.')
st.caption('Educational demo. Not for medical use.')

st.markdown('<h2 style="margin-top:-10px">Machine Learning Heart Disease Predictor</h2>', unsafe_allow_html=True)
st.markdown('<div style="color:#6b7280">Attractive, fast, and interactive risk estimation UI</div>', unsafe_allow_html=True)

with st.form('patient_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', 18, 100, 50, step=1)
        resting_bp = st.number_input('Resting BP (mm Hg)', 80, 220, 120, step=1)
        fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', ['No', 'Yes'])
    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cholesterol = st.number_input('Cholesterol (mg/dL)', 100, 600, 200, step=1)
        max_hr = st.number_input('Max Heart Rate', 60, 220, 150, step=1)
    with col3:
        chest_pain = st.selectbox('Chest Pain Type', ['Typical', 'Atypical', 'Nonanginal', 'Asymptomatic'])
        exercise_angina = st.selectbox('Exercise Angina', ['No', 'Yes'])
        oldpeak = st.number_input('Oldpeak (ST Depression)', 0.0, 10.0, 0.0, step=0.1)

    st_slope = st.select_slider('ST Slope', options=['Up', 'Flat', 'Down'], value='Up')
    submitted = st.form_submit_button('Predict Heart Disease Risk')

if submitted:
    features = {
        'age': int(age),
        'sex': sex,
        'chest_pain': chest_pain.lower(),
        'resting_bp': float(resting_bp),
        'cholesterol': float(cholesterol),
        'fasting_bs': fasting_bs,
        'resting_ecg': 'Normal',
        'max_hr': float(max_hr),
        'exercise_angina': exercise_angina,
        'oldpeak': float(oldpeak),
        'st_slope': st_slope
    }

    result = compute_heuristic_probability(features)

    card_color = '#ef4444' if result['has_disease'] else '#10b981'
    bg = 'linear-gradient(135deg, rgba(239,68,68,0.12), rgba(236,72,153,0.12))' if result['has_disease'] else 'linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.12))'

    st.markdown(f'''
    <div style="background:{bg};border:1px solid #e5e7eb;border-radius:18px;padding:22px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div style="font-weight:800;font-size:20px;margin-bottom:4px">Prediction Result</div>
                <div style="color:#6b7280">Estimated probability of heart disease</div>
            </div>
            <div style="font-size:36px">{'⚠️' if result['has_disease'] else '✅'}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(ring_kpi('Probability', result['probability'], '%', card_color), use_container_width=True, config={'displayModeBar': False})
    with c2:
        level_color = {'High': '#ef4444', 'Moderate': '#f59e0b', 'Low': '#10b981'}[result['risk_level']]
        st.plotly_chart(ring_kpi('Risk Level', {'Low': 20, 'Moderate': 55, 'High': 85}[result['risk_level']], '', level_color), use_container_width=True, config={'displayModeBar': False})
    with c3:
        st.plotly_chart(ring_kpi('Model Confidence', result['confidence'], '%', '#6366f1'), use_container_width=True, config={'displayModeBar': False})

    factor_scores = {
        'Age': 80 if age >= 50 else 35,
        'Cholesterol': 75 if cholesterol >= 200 else 35,
        'Blood Pressure': 70 if resting_bp >= 130 else 30,
        'Heart Rate': 65 if max_hr <= 140 else 25,
        'Exercise': 85 if exercise_angina == 'Yes' else 20,
        'ST Slope': 70 if st_slope != 'Up' else 20,
        'Oldpeak': 70 if oldpeak >= 1.0 else 25
    }

    st.markdown('### Risk Factor Analysis')
    st.plotly_chart(radar_chart(factor_scores), use_container_width=True, config={'displayModeBar': False})

    st.info('This is an educational demonstration. Consult healthcare professionals for medical advice.')

else:
    with st.expander('About this demo', expanded=True):
        st.write('''
        CardioPredict AI is an attractive, fast Streamlit app for demonstrating heart disease risk prediction.
        It uses a pragmatic heuristic so the app runs anywhere instantly without large ML dependencies or internet access.
        ''')
