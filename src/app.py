import streamlit as st
import pandas as pd
import joblib
import numpy as np
from typing import Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# --- IMPORTAÇÕES DAS SUAS ABSTRAÇÕES ---
from config import MODEL_FILE, LABEL_ENCODER_FILE
from transformers import MtransGrouper, CalcGrouper, RoundingTransformer

# -------------------------------------------------------------------
# Configuração da Página
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Calculadora de Obesidade - FIAP", page_icon="🏋️‍♂️", layout="wide"
)

# --- INICIALIZAÇÃO DO SESSION STATE ---
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "calculation_error" not in st.session_state:
    st.session_state.calculation_error = None


# -------------------------------------------------------------------
# CSS Customizado (Cores FIAP + Texto Preto nos Inputs)
# -------------------------------------------------------------------
def local_css():
    st.markdown(
        """
    <style>
    .stApp { background-color: #000000; }
    header[data-testid="stHeader"] { display: none; visibility: hidden; }
    footer { display: none; visibility: hidden; }
    
    label { color: #FFFFFF !important; }
    h1, h2, h3 { color: #E6007E !important; }

    .stButton > button {
        background-color: #E6007E;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        width: 100%;
    }

    /* CONFIGURAÇÃO DOS INPUTS (Fundo Branco + Texto Preto) */
    div[data-baseweb="select"] > div:first-child,
    div[data-baseweb="input"] > div,
    div[data-baseweb="text-input"] > div {
        background-color: #FFFFFF !important;
        border-radius: 5px;
    }

    div[data-baseweb="select"] > div:first-child div, input { 
        color: #000000 !important; 
        -webkit-text-fill-color: #000000 !important;
    }

    input::placeholder { color: #000000 !important; opacity: 0.7 !important; }
    ::-webkit-input-placeholder { color: #000000 !important; }
    ul[data-testid="stSelectboxVirtualList"] li { color: #000000 !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Carregamento de Artefatos
# -------------------------------------------------------------------
@st.cache_resource
def carregar_artefatos() -> Tuple[Optional[Pipeline], Optional[LabelEncoder]]:
    """
    Carrega o pipeline e o label encoder.
    Retorna (None, None) em caso de falha.
    """
    try:
        model = joblib.load(MODEL_FILE)
        encoder = joblib.load(LABEL_ENCODER_FILE)
        return model, encoder
    except Exception as e:
        st.error(f"Erro crítico ao carregar artefatos: {e}")
        return None, None


pipeline_raw, le_raw = carregar_artefatos()
if pipeline_raw is None or le_raw is None:
    st.error("O sistema não pôde iniciar porque os modelos não foram encontrados.")
    st.stop()

pipeline: Pipeline = pipeline_raw
le: LabelEncoder = le_raw

local_css()

# -------------------------------------------------------------------
# Mapeamentos e Opções
# -------------------------------------------------------------------
MAPA_TRADUCOES_DISPLAY = {
    "genero": {"feminino": "Feminino", "masculino": "Masculino"},
    "sim_nao": {"sim": "Sim", "nao": "Não"},
    "mtrans": {
        "carro": "Automóvel",
        "transporte_publico": "Transporte Público",
        "caminhando": "Caminhando",
        "moto": "Moto",
        "bicicleta": "Bicicleta",
    },
    "frequencia": {
        "nunca": "Nunca",
        "as_vezes": "Às vezes",
        "frequentemente": "Frequentemente",
        "sempre": "Sempre",
    },
    "fcvc": {1: "Nunca", 2: "Às vezes", 3: "Sempre"},
    "ncp": {1: "1 refeição", 2: "2 refeições", 3: "3 refeições", 4: "4 ou mais"},
    "ch20": {1: "Menos que 1L", 2: "1 a 2L", 3: "Mais que 2L"},
    "faf": {
        0: "Nunca",
        1: "1-2 dias/semana",
        2: "2-4 dias/semana",
        3: "4+ dias/semana",
    },
    "tue": {0: "0-2h", 1: "3-5h", 2: "5h+"},
}
# -------------------------------------------------------------------
# Layout e Coleta de Inputs
# -------------------------------------------------------------------
col_logo, col_titulo = st.columns([1, 5])
with col_logo:
    st.image("imagens/logo_fiap.png", width=100)
with col_titulo:
    st.markdown("<h1>Calculadora de Obesidade</h1>", unsafe_allow_html=True)

inputs_usuario = {}
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Dados Pessoais")
    nome_usuario = st.text_input("Nome", placeholder="Insira seu nome")
    inputs_usuario["idade"] = st.number_input(
        "Qual é a sua idade?", 1, 100, value=None, placeholder="Ex: 40"
    )
    inputs_usuario["genero"] = st.selectbox(
        "Qual é o seu genero?",
        ["feminino", "masculino"],
        format_func=lambda x: MAPA_TRADUCOES_DISPLAY["genero"].get(x, x),
        index=None,
        placeholder="Selecione uma opção",
    )
    inputs_usuario["historico_familiar"] = st.selectbox(
        "Possui histórico familiar de obesidade?",
        ["sim", "nao"],
        format_func=lambda x: MAPA_TRADUCOES_DISPLAY["sim_nao"].get(x, x),
        index=None,
        placeholder="Selecione uma opção",
    )

with c2:
    st.subheader("Rotina")
    inputs_usuario["faf"] = st.selectbox(
        "Quantas vezes pratica atividade física?",
        [0, 1, 2, 3],
        format_func=lambda x: str(MAPA_TRADUCOES_DISPLAY["faf"].get(x, x)),
        index=None,
        placeholder="Selecione uma opção",
    )
    inputs_usuario["mtrans"] = st.selectbox(
        "Qual é o seu meio de transporte principal?",
        ["carro", "transporte_publico", "caminhando", "moto", "bicicleta"],
        format_func=lambda x: MAPA_TRADUCOES_DISPLAY["mtrans"].get(x, x),
        index=None,
        placeholder="Selecione uma opção",
    )
    inputs_usuario["tue"] = st.selectbox(
        "Quantas horas por dia faz o uso de telas?",
        [0, 1, 2],
        format_func=lambda x: str(MAPA_TRADUCOES_DISPLAY["tue"].get(x, x)),
        index=None,
        placeholder="Selecione uma opção",
    )
    scc = st.selectbox(
        "Você monitora calorias?",
        ["sim", "nao"],
        format_func=lambda x: MAPA_TRADUCOES_DISPLAY["sim_nao"].get(x, x),
        index=None,
        placeholder="Selecione uma opção",
    )

with c3:
    st.subheader("Alimentação")
    inputs_usuario["favc"] = st.selectbox(
        "Consome comida calórica?",
        ["sim", "nao"],
        format_func=lambda x: MAPA_TRADUCOES_DISPLAY["sim_nao"].get(x, x),
        index=None,
        placeholder="Selecione uma opção",
    )
    inputs_usuario["fcvc"] = st.selectbox(
        "Com qual frequêcia consome vegetais?",
        [1, 2, 3],
        format_func=lambda x: str(MAPA_TRADUCOES_DISPLAY["fcvc"].get(x, x)),
        index=None,
        placeholder="Selecione uma opção",
    )
    inputs_usuario["caec"] = st.selectbox(
        "Com qual frequência come entre refeições?",
        ["nunca", "as_vezes", "frequentemente", "sempre"],
        format_func=lambda x: MAPA_TRADUCOES_DISPLAY["frequencia"].get(x, x),
        index=None,
        placeholder="Selecione uma opção",
    )
    inputs_usuario["ch20"] = st.selectbox(
        "Quantos litros de água você consome por dia?",
        [1, 2, 3],
        format_func=lambda x: str(MAPA_TRADUCOES_DISPLAY["ch20"].get(x, x)),
        index=None,
        placeholder="Selecione uma opção",
    )
    inputs_usuario["ncp"] = st.selectbox(
        "Quantas refeições você faz por dia?",
        [1, 2, 3, 4],
        format_func=lambda x: str(MAPA_TRADUCOES_DISPLAY["ncp"].get(x, x)),
        index=None,
        placeholder="Selecione uma opção",
    )
    inputs_usuario["calc"] = st.selectbox(
        "Com qual frequência você faz o uso de álcool?",
        ["nunca", "as_vezes", "frequentemente", "sempre"],
        format_func=lambda x: MAPA_TRADUCOES_DISPLAY["frequencia"].get(x, x),
        index=None,
        placeholder="Selecione uma opção",
    )

# -------------------------------------------------------------------
# Lógica de Cálculo e Auditoria
# -------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])

with col_btn_2:
    if st.button("Calcular Classificação", width="stretch"):
        if None in inputs_usuario.values() or not nome_usuario:
            st.warning("Preencha todos os campos.")
        else:
            try:
                # 1. ALINHAMENTO AUTOMÁTICO (Recupera a ordem do Modelo)
                colunas_modelo = list(pipeline.feature_names_in_)
                df_completo = pd.DataFrame([inputs_usuario])
                df_input = df_completo.reindex(columns=colunas_modelo)

                # 2. Execução da Predição
                previsao = pipeline.predict(df_input)
                st.session_state.probabilidade = pipeline.predict_proba(df_input)
                st.session_state.resultado_classe = le.inverse_transform(previsao)[0]
                st.session_state.inputs_validados = df_input
                st.session_state.show_results = True

            except Exception as e:
                st.session_state.calculation_error = f"Erro na predição: {e}"

# -------------------------------------------------------------------
# Exibição dos Resultados
# -------------------------------------------------------------------
if st.session_state.show_results:
    st.markdown("---")
    res_classe = st.session_state.resultado_classe.replace("_", " ").title()

    st.markdown(
        f"""
        <div style="background-color: #333; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #E6007E;">Olá {nome_usuario}, seu resultado é:</h2>
            <h1 style="color: #FFFFFF;">{res_classe}</h1>
        </div>
    """,
        unsafe_allow_html=True,
    )
    c_res, c_data = st.columns(2)
    with c_res:
        st.subheader("Distribuição de risco")
        df_prob = pd.DataFrame(st.session_state.probabilidade, columns=le.classes_).T
        df_prob.columns = ["Probabilidade"]
        st.dataframe(df_prob.style.format("{:.2%}"), width="stretch")

    with c_data:
        st.subheader("Dados Processados")
        df_display = st.session_state.inputs_validados.T.reset_index()
        df_display.columns = ["Variável", "Valor"]
        st.dataframe(df_display.astype(str), width="stretch")
