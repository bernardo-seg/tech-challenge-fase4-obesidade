import pandas as pd
import requests
from pathlib import Path
import json


def criar_csv_de_dataframe(df, nome_arquivo) -> None:
    # Converte para string apenas para a verificação do .endswith()
    nome_str = str(nome_arquivo)

    if not nome_str.lower().endswith(".csv"):
        print("Erro: O nome do arquivo deve terminar com '.csv'.")
        return

    try:
        # O Pandas aceita o objeto Path ou String sem problemas
        df.to_csv(nome_arquivo, index=False)
        print(f"✅ DataFrame salvo com sucesso em '{nome_str}'")
    except Exception as e:
        print(f"❌ Erro ao salvar: {e}")


def renomear_colunas(df: pd.DataFrame, mapa_renomeacao: dict = None) -> pd.DataFrame:
    """
    Renomeia as colunas de um DataFrame utilizando um dicionário fornecido
    ou aplicando um padrão de formatação (minúsculas, espaços por '_').
    Retorna o DataFrame com as colunas renomeadas.
    """
    colunas_originais = df.columns.tolist()
    df_modificado = df.copy()

    if mapa_renomeacao:

        if any(col in mapa_renomeacao for col in colunas_originais):
            df_modificado.rename(columns=mapa_renomeacao, inplace=True)
        else:
            print(
                "Não foi possível realizar a alteração dos nomes das colunas. Rever arquivo de entrada."
            )
    else:

        print(
            "Não foi possível realizar a alteração dos nomes das colunas. Rever arquivo de entrada."
        )

    return df_modificado


def transformar_valores_string(df: pd.DataFrame, coluna: str, mapa_transformacao: dict):
    """
    Transforma os valores de uma coluna do tipo string para string utilizando um mapeamento.
    Modifica o DataFrame inplace.
    """
    if coluna not in df.columns:
        print(f"Erro: A coluna '{coluna}' não existe no DataFrame.")
        return

    if df[coluna].dtype != "object":
        print(
            f"Erro: A coluna '{coluna}' não é do tipo 'object'. Nenhuma transformação será aplicada."
        )
        return

    valores_unicos_na_coluna = df[coluna].dropna().unique()
    valores_nao_mapeados = [
        valor for valor in valores_unicos_na_coluna if valor not in mapa_transformacao
    ]

    if valores_nao_mapeados:
        print(
            f"Erro: Os seguintes valores únicos na coluna '{coluna}' não foram encontrados no mapeamento:"
        )
        print(valores_nao_mapeados)
        print(
            "A transformação não será realizada pois nem todos os valores possuem um mapeamento."
        )
        return

    df[coluna] = df[coluna].map(mapa_transformacao)
    print(f"Coluna '{coluna}' transformada com sucesso.")


def ler_json(caminho_ou_url: str) -> dict:
    """
    Lê um JSON de uma URL ou de um caminho local.
    """
    # Se começar com http, usa o requests (para o GitHub)
    if str(caminho_ou_url).startswith("http"):
        try:
            resposta = requests.get(caminho_ou_url)
            resposta.raise_for_status()
            return resposta.json()
        except Exception as e:
            print(f"Erro ao ler JSON da URL {caminho_ou_url}: {e}")
            return None

    # Se não for URL, tenta ler como arquivo local (para a sua pasta /dados)
    else:
        try:
            with open(caminho_ou_url, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Erro ao ler JSON local {caminho_ou_url}: {e}")
            return None


def validar_processamento(df_orig, df_proc, mapa_cols, mapa_vals):
    """Realiza um sanity check para garantir a qualidade dos dados processados."""
    print("\n🔍 Iniciando validação dos dados...")

    # Teste de Colunas
    colunas_esperadas = list(mapa_cols.values())
    faltam_colunas = [col for col in colunas_esperadas if col not in df_proc.columns]

    if not faltam_colunas:
        print("✅ Sucesso: Todas as colunas foram renomeadas.")
    else:
        print(f"⚠️ Atenção: Colunas não encontradas: {faltam_colunas}")

    # Teste de Integridade de Linhas
    if len(df_orig) == len(df_proc):
        print(f"✅ Sucesso: Integridade de linhas mantida ({len(df_proc)} linhas).")
    else:
        print(f"❌ Erro: O número de linhas mudou!")
