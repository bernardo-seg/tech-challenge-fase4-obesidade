from utils import renomear_colunas, transformar_valores_string


def preparar_dados_obesidade(df, mapa_colunas, mapa_valores_colunas):
    """
    Orquestra todo o tratamento de dados brutos para o projeto de obesidade.

    df: arquivo com os dados brutos a serem processados

    mapa_colunas: Contém os novos nomes das colunas dos dados brutos

    map_valores: contém os novos valores a serem transformados

    """

    df_processado = df.copy()

    # 1. Validação e Renomeação de Colunas
    if not mapa_colunas:
        print(
            "❌ Arquivo mapa_colunas não encontrado. Retornado dataset df sem transformações"
        )
        return df_processado

    else:
        df_processado = renomear_colunas(df_processado, mapa_colunas)
        print("✅ colunas renomeadas")

    # 2. Validação e Transformação de Valores
    if not mapa_valores_colunas:
        print(
            "❌ Arquivo mapa_valores não encontrado. Retornado dataset df sem transformações"
        )
        return df_processado

    # Início das transformações categóricas
    print("Iniciando transformações de valores...")

    # Classificação do Peso (Alvo)
    if "mapeamento_classificacao_peso_corporal" in mapa_valores_colunas:
        transformar_valores_string(
            df_processado,
            "classificacao_peso_corporal",
            mapa_valores_colunas["mapeamento_classificacao_peso_corporal"][
                "valores_novos_classificacao_peso_corporal"
            ],
        )
        print("  ✅ Valores classificacao_peso_corporal ok.")

    # Meio de Transporte
    if "mapeamento_mtrans" in mapa_valores_colunas:
        transformar_valores_string(
            df_processado,
            "mtrans",
            mapa_valores_colunas["mapeamento_mtrans"]["valores_novos_mtrans"],
        )
        print("  ✅ Valores mtrans ok.")

    # Frequência (CAEC e CALC)
    if "mapeamento_frequencia" in mapa_valores_colunas:
        transformar_valores_string(
            df_processado,
            "caec",
            mapa_valores_colunas["mapeamento_frequencia"]["valores_novos_frequencia"],
        )
        transformar_valores_string(
            df_processado,
            "calc",
            mapa_valores_colunas["mapeamento_frequencia"]["valores_novos_frequencia"],
        )
        print("  ✅ Valores caec e calc ok.")

    # Gênero
    if "mapeamento_genero" in mapa_valores_colunas:
        transformar_valores_string(
            df_processado,
            "genero",
            mapa_valores_colunas["mapeamento_genero"]["transformacao_genero"],
        )
        print("  ✅ Valores genero ok.")

    # Variáveis Binárias (Sim/Não)
    if "mapeamento_sim_nao" in mapa_valores_colunas:
        colunas_sim_nao = ["historico_familiar", "favc", "fumante", "scc"]
        for coluna in colunas_sim_nao:
            transformar_valores_string(
                df_processado,
                coluna,
                mapa_valores_colunas["mapeamento_sim_nao"]["transformacao_sim_nao"],
            )
            print(f"  ✅ Colunas binárias {colunas_sim_nao} ok.")

    return df_processado
