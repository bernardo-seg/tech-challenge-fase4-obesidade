Este projeto foi desenvolvido como parte do **Tech Challenge (Fase 4)** na FIAP. 

.

## 🎯 Objetivo
Trata-se de uma solução que utiliza **Machine Learning** para prever a probilidade de níveis de obesidade, baseado em um questionário de informação pessoais e hábitos


## 📊 Performance do Modelo

* **Acurácia Geral**: 83.69%
* **F1-Score (Média)**: 0.83

O relatório de classificação pode foi gerado em 07/01/2026

local: 
nome do arquivo: dados/relatorio_classificacao_2026_01_07.txt
script gerador: src/pipeline_treino.py


## 🛠️ Tecnologias Utilizadas
* **Linguagem**: Python 3.12
* **Interface**: Streamlit para a criação do Web App
* **Data Science**: Scikit-Learn (Pipelines, Transformers, Random Forest)

## 🏗️ Estrutura do Projeto
```text
CALCULADORA-OBESIDADE/
├── dados/             # Arquivos utilizados no projeto
├── models/            # Pipeline e Encoder salvos em .joblib
├── notebooks/         # Arquivos da extapa de exploração de dados, treinamento do modelo e criação da pipeline
├── src/               # Códigos fonte da aplicação produtiva (pipeline de treinamento do modelo de machine learning e aplicação streamlit)
└── requirements.txt   # Dependências do projeto

## 🔎 Análise e engenharia de feature

### **Ambiente de Desenvolvimento:** google colab 
    
    Caso execute o notebook localmente, pode ocorrer erros de versões de utilitários. O requirements.txt está com as versões utilizadas para os scripts python da pasta src

    **local:** notebooks/

#### **arquivos:**
    
    ##### analise_engenharia_feature.ipynb
                  
        **conteúdo:**
            - Análise exploratória dos dados
            - Engenharia de feature

        **Arquivos utilizados:**

            local: dados/
                - Obesity.csv -> Arquivo base para o projeto

                - descricao_dados_obesidade.csv: descrição do significado de cada coluna do arquivo Obesity.csv

                - mapa_colunas.json: Contém os valores utilizados para alterar os nomes das colunas do arquivo Obesity.csv

                - mapa_valores_colunas.csv: Contém os valores utilizados para alterar os valores das colunas do arquivo Obesity.csv

        **Arquivos gerados:**

            local: dados/

            - dicionario_dados_tech_challenge_4_notebook.json: Contém as alterações realizadas na colunas e valores originais do arquivo csv Obesity.csv

            - obesidade_processado_notebook.csv: Arquivo csv gerado com as alterações realizadas na colunas e valores originais do arquivo csv Obesity.csv no notebook analise_engenharia_feature.ipynb

            - analise_engenharia_feature.pdf: Versão pdf do notebook caso prefira acessá-lo nesse formato.

    ##### treinamento_modelo.ipynb
                  
        **conteúdo:**
            - Treinamento dos dados em modelos de machine learning
            - Escolha do modelo a ser utilizado no projeto

        **Arquivos utilizados:**

            local: dados/
                - obesidade_processado_notebook.csv -> Arquivo base para o treinamento
             
        **Arquivos gerados:**

            local: dados/

                - treinamento_modelo.pdf: Versão pdf do notebook caso prefira acessá-lo nesse formato.

            Obs: O notebook gera o arquivo modelo_obesidade_final.joblib. Não salvamos esse arquivo na pasta do projeto porque o arquivo usado no projeto final foi gerado pelo processamento da pipeline produtiva. 
    
    ##### pipeline_modelo_rf.ipynb
                  
        **conteúdo:**
            - Criação de uma pipeline completa de treinamento de machine learning com o algoritimo de treinamento escolhido após análise realizada no arquivo treinamento_modelo
            - Algoritmo utilziado para o treinamento do modelo de machine learning foi o Random Forest

        **Arquivos utilizados:**

            - Obesity.csv -> Arquivo base para o projeto

            - mapa_colunas.json: Contém os valores utilizados para alterar os nomes das colunas do arquivo Obesity.csv

            - mapa_valores_colunas.csv: Contém os valores utilizados para alterar os valores das colunas do arquivo Obesity.csv
             
        **Arquivos gerados:**

            local: dados/

                - pipeline_modelo_rf.pdf: Versão pdf do notebook caso prefira acessá-lo nesse formato.

            Obs: O notebook gera os arquivos pipeline_obesidade_completo_rf.joblib, label_encoder_rf.joblib e obessidade_processado.csv. Não salvamos esse arquivo na pasta do projeto porque os arquivos usados no projeto final foi gerado pelo processamento da pipeline produtiva. 


## ✳️ Sobre a aplicação

**Arquivos:**

- app.py: Contém o código da aplicação do streamlit.

- config.py: Arquivos com as configurações dos arquivos, diretórios e caminhos utilizados nos códigos do projeto.

- pipeline_treino.py: código responsável pelo treinamento do modelo de machine learning utilizado no projeto. 
    
    Gera os arquivos label_encoder_rf.joblib e pipeline_completa_rf.joblib no diretorio models/

- preprocessamento.py: responsável pelas estapas de preprocessamento da pipeline_treino.py

- transformers.py: contém as funções de transformação utilizadas no projeto

- utils.py: contém funções auxiliares utilizadas no projeto.


### 🚀 Como Executar

1. Clone o repositório.

2. Crie um ambiente virtual: python -m venv .venv.

3. Instale as dependências: pip install -r requirements.txt.

4. Execute o app: streamlit run src/app.py.