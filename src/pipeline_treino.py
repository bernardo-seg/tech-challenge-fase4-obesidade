import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


from config import (
    OBESITY_CSV,
    MODEL_FILE,
    DATA_DIR,
    MAPA_COLUNAS,
    MAPA_VALORES_COLUNA,
    LABEL_ENCODER_FILE,
)
from utils import ler_json, validar_processamento, criar_csv_de_dataframe
from transformers import RoundingTransformer, MtransGrouper, CalcGrouper
from preprocessamento import preparar_dados_obesidade
from config import RELATORIO_MODELO


# ETAPA 1 - importação de dados

df = pd.read_csv(OBESITY_CSV)
mapa_colunas = ler_json(MAPA_COLUNAS)
mapa_valores_colunas = ler_json(MAPA_VALORES_COLUNA)

print("DataFrame e mapas JSON carregados com sucesso.")

# ETAPA 2: PRÉ-PROCESSAMENTO
df_processado = preparar_dados_obesidade(df, mapa_colunas, mapa_valores_colunas)

# ETAPA 3: VALIDAÇÃO (Chamada simples)
validar_processamento(df, df_processado, mapa_colunas, mapa_valores_colunas)

# ETAPA 4: Criação arquivo df_processado.csv

caminho_csv_proc = DATA_DIR / "obesidade_processado_pipeline.csv"
criar_csv_de_dataframe(df_processado, caminho_csv_proc)

print("Pré-processamentos iniciais aplicados e df_processado criado.")

# ETAPA 5: PREPARAÇÃO PARA O MODELO

print("Iniciando prepação do modelo")

coluna_alvo = "classificacao_peso_corporal"

# Remover variaveis com vazamento ou risco extremo e colunas que não serão usadas no pipeline
variaveis_a_remover_de_X_antes_do_pipeline = ["peso", "altura", "fumante", "scc"]

X = df_processado.drop(
    columns=[coluna_alvo] + variaveis_a_remover_de_X_antes_do_pipeline
)
y = df_processado[coluna_alvo]

# Codificar o Alvo (y) de texto para números
le = LabelEncoder()
y_codificada = le.fit_transform(y)

# Separar dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y_codificada, test_size=0.2, random_state=42, stratify=y_codificada
)

print("Dados preparados: X, y definidos, y codificado e dividido em treino/teste.")

variaveis_continuas = ["idade"]
variaveis_bin_nominal = ["genero", "historico_familiar", "favc"]
variaveis_multi_nominal = ["mtrans"]
variaveis_clean_ordenadas = ["caec"]
variaveis_para_arredondar_e_codificar = ["fcvc", "ncp", "ch20", "faf", "tue"]

pipeline_continua = Pipeline(steps=[("scaler", StandardScaler())])

# Pipeline que primeiro arredonda e depois codifica ordinalmente
pipeline_arrredonamento_ordenacao = Pipeline(
    steps=[
        ("rounder", RoundingTransformer()),
        (
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ]
)

pipeline_ordenada_limpa = Pipeline(
    steps=[
        (
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        )
    ]
)

pipeline_calc = Pipeline(
    steps=[
        ("grouper", CalcGrouper()),
        (
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ]
)

pipeline_nominal_bin = Pipeline(
    steps=[
        (
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        )
    ]
)

pipeline_nominal_multi = Pipeline(
    steps=[
        ("grouper", MtransGrouper()),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ]
)

preprocessador = ColumnTransformer(
    transformers=[
        ("cont", pipeline_continua, variaveis_continuas),
        (
            "arredondada_ord",
            pipeline_arrredonamento_ordenacao,
            variaveis_para_arredondar_e_codificar,
        ),
        ("ord_limpa", pipeline_ordenada_limpa, variaveis_clean_ordenadas),
        ("calc_pipe", pipeline_calc, ["calc"]),
        ("nom_bin", pipeline_nominal_bin, variaveis_bin_nominal),
        ("nom_multi", pipeline_nominal_multi, variaveis_multi_nominal),
    ],
    remainder="drop",
)

modelo_rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    max_depth=20,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
)

pipeline_completa_rf = Pipeline(
    steps=[("preprocessor", preprocessador), ("model", modelo_rf)]
)

print(
    "Pipelines de pré-processamento e o modelo Random Forest definidos e combinados no pipeline completo."
)

# ETAPA 6 - treinameto
print("Iniciando o treinamento do pipeline Random Forest")

pipeline_completa_rf.fit(X_treino, y_treino)

y_prev_rf = pipeline_completa_rf.predict(X_teste)
rf_acuracia = accuracy_score(y_teste, y_prev_rf)
print(f"Acurácia do Pipeline Random Forest: {rf_acuracia * 100:.2f}%")

nomes_classes = le.classes_

# ETAPA 8 - Relatório

report_str = classification_report(y_teste, y_prev_rf, target_names=nomes_classes)


print("\nRelatório de Classificação:")
print(report_str)


print(f"📄 Salvando relatório em: {RELATORIO_MODELO}")

try:
    with open(RELATORIO_MODELO, "w", encoding="utf-8") as f:
        f.write("=== RELATÓRIO DE PERFORMANCE DO MODELO ===\n")
        f.write(f"Acurácia Geral: {rf_acuracia * 100:.2f}%\n")
        f.write("-" * 43 + "\n")
        f.write(report_str)
    print("✅ Relatório salvo com sucesso!")
except Exception as e:
    print(f"❌ Erro ao salvar o relatório: {e}")

# Etapa 8 - salvando os artefatos
print("\n💾 Salvando artefatos na pasta models...")

try:
    # Salva o Pipeline completo (Preprocessamento + Modelo)
    joblib.dump(pipeline_completa_rf, MODEL_FILE)

    # Salva o LabelEncoder para decodificar as previsões no Streamlit
    joblib.dump(le, LABEL_ENCODER_FILE)

    print(f"✅ Artefatos salvos com sucesso em: {MODEL_FILE.parent}")
    print(f"📦 Classes mapeadas no LabelEncoder: {list(le.classes_)}")

except Exception as e:
    print(f"❌ Erro ao salvar os artefatos: {e}")
