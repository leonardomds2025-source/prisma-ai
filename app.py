import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import time
import shutil
import sqlite3
from datetime import datetime
from faster_whisper import WhisperModel
import google.generativeai as genai
from fpdf import FPDF

# CONFIGURAÇÃO
COST_PER_MIN = 0.50
# No Streamlit Cloud, não persistimos arquivos (eles somem no reboot). 
# Usamos memória temporária.
DB_FILE = "prisma_core.db" 
MODEL_NAME = 'gemini-2.5-flash' 

st.set_page_config(layout="wide", page_title="PRISMA AI | Cloud", page_icon="💎")

# CSS (Mantido igual)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif; background-color: #0b0f19; color: #e2e8f0;}
    .metric-card {background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 15px; text-align: center;}
    .glass-card {background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; margin-bottom: 15px;}
    .feedback-box {border-left: 4px solid #8b5cf6; background: rgba(139, 92, 246, 0.1); padding: 15px; border-radius: 0 8px 8px 0; margin-top: 10px;}
    .timestamp-badge {background: #334155; color: #cbd5e1; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-family: monospace; margin-right: 8px;}
    .stButton>button {width: 100%; border-radius: 8px; font-weight: 600; height: 3rem; background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%); border: none;}
</style>
""", unsafe_allow_html=True)

# BANCO DE DADOS (SQLite Local)
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS auditorias (
        id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT, arquivo TEXT, campanha TEXT, modo TEXT,
        nota REAL, resumo TEXT, tags TEXT, duracao REAL, checklist_json TEXT)''')
    conn.close()

def db_save_audit(filename, nota, campanha, mode, resumo, tags, duration_sec=0, checklist_data=None):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    checklist_str = json.dumps(checklist_data) if checklist_data else "[]"
    c.execute('''INSERT INTO auditorias (data, arquivo, campanha, modo, nota, resumo, tags, duracao, checklist_json) VALUES (?,?,?,?,?,?,?,?,?)''', 
             (datetime.now().strftime("%Y-%m-%d %H:%M"), str(filename), str(campanha), mode, float(nota), str(resumo)[:500], ",".join(tags) if tags else "", round(duration_sec, 2), checklist_str))
    conn.commit(); conn.close()

def db_load_data():
    conn = sqlite3.connect(DB_FILE)
    try: return pd.read_sql_query("SELECT * FROM auditorias", conn)
    except: return pd.DataFrame()
    finally: conn.close()

init_db()

# INTELIGÊNCIA (Adaptada para st.secrets)
def ai_init():
    # Pega a chave dos Segredos do Streamlit Cloud
    key = st.secrets.get("GOOGLE_API_KEY")
    if not key: return False
    genai.configure(api_key=key)
    return True

@st.cache_resource
def ai_load_whisper():
    # ADAPTAÇÃO PARA CPU: device="cpu", compute_type="int8"
    return WhisperModel("large-v3", device="cpu", compute_type="int8")

def ai_analyze_text(text, mode, keywords):
    role = "AUDITOR DE QUALIDADE SÊNIOR."
    prompt = f"""{role} Busque: {keywords}
    Analise a transcrição. Retorne JSON válido.
    TXT: {text[:60000]}
    Output JSON format: {{ "nota": 0.0, "resumo": "...", "checklist": [{{"item": "...", "status": "OK/NOK", "obs": "...", "tempo": "00:00"}}], "swot": {{...}}, "pdca": {{...}}, "feedback_gestor": "..." }}"""
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        res = model.generate_content(prompt)
        clean = res.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean[clean.find("{"):clean.rfind("}")+1])
    except Exception as e: return {"nota":0, "resumo": f"Erro IA: {e}"}

# UI - MAIN
st.sidebar.title("💎 PRISMA Cloud")
nav = st.sidebar.radio("Menu", ["Auditoria", "Dashboard"])

if nav == "Auditoria":
    st.title("🎧 Auditoria Cloud")
    f_up = st.file_uploader("Áudio", type=['mp3','wav','m4a'])
    
    if st.button("PROCESSAR (Pode demorar s/ GPU)") and f_up:
        if not ai_init(): st.error("Configure a GOOGLE_API_KEY nos Secrets!"); st.stop()
        
        with st.status("Processando na CPU...", expanded=True):
            st.write("Transcrevendo (Whisper CPU)...")
            model = ai_load_whisper()
            # Salva temp
            with open("temp_audio", "wb") as f: f.write(f_up.getbuffer())
            seg, info = model.transcribe("temp_audio", language="pt")
            txt = " ".join([s.text for s in seg])
            
            st.write("Analisando (Gemini)...")
            d = ai_analyze_text(txt, "Padrão", "")
            
            # Salva
            db_save_audit(f_up.name, d.get('nota',0), "Cloud", "Padrão", d.get('resumo',''), [], info.duration, d.get('checklist',[]))
            st.success("Pronto!")
            st.write(d)
            os.remove("temp_audio")

elif nav == "Dashboard":
    page_dashboard() # Mesma lógica do dashboard anterior