from pathlib import Path
from datetime import datetime

data_hoje = datetime.now().strftime("%Y_%m_%d")

# Raiz do Projeto
ROOT_DIR = Path(__file__).resolve().parent.parent

# Caminhos das pastas principais
DATA_DIR = ROOT_DIR / "dados"
MODELS_DIR = ROOT_DIR / "models"
SRC_DIR = ROOT_DIR / "src"

# Caminhos pasta data
OBESITY_CSV = DATA_DIR / "Obesity.csv"
MAPA_COLUNAS = DATA_DIR / "mapa_colunas.json"
MAPA_VALORES_COLUNA = DATA_DIR / "mapa_valores_colunas.json"
RELATORIO_MODELO = DATA_DIR / f"relatorio_classificacao_{data_hoje}.txt"

# caminhos pasta models
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder_rf.joblib"
MODEL_FILE = MODELS_DIR / "pipeline_completa_rf.joblib"

# Garante que as pastas essenciais existam
MODELS_DIR.mkdir(parents=True, exist_ok=True)
