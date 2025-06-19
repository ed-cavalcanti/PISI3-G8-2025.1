# Streamlit EDA Dashboard

Este projeto é um dashboard interativo de Análise Exploratória de Dados (EDA) desenvolvido com Streamlit. O objetivo é permitir que os usuários explorem um conjunto de dados sobre diabetes através de visualizações interativas.

## Estrutura do Projeto

O projeto é organizado da seguinte forma:

```

├── src
│   ├── app.py                                           # Ponto de entrada da aplicação Streamlit
│   ├── components                                       # Componentes do dashboard
│   │   ├── visualizations.py                            # Funções para gerar gráficos
├── data
│   ├── diabetes_health_indicators_BRFSS2015.csv         # Dataset utilizado para análise
├── requirements.txt                                     # Dependências do projeto
├── .streamlit
│   └── config.toml                                      # Configurações específicas do Streamlit
├── .gitignore                                           # Arquivos a serem ignorados pelo Git
└── README.md                                            # Documentação do projeto
```

## Instalação

Para instalar as dependências do projeto, execute o seguinte comando:

```
pip install -r requirements.txt
```

## Uso

Para iniciar o dashboard, execute o seguinte comando:

```
streamlit run src/app.py
```

## Dataset

O dataset utilizado para a análise está localizado em `data/diabetes_health_indicators_BRFSS2015.csv`.
O datasset foi baixado através da plataforma [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
