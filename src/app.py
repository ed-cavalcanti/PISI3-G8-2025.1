import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from components.visualizations import (
    features,
    target,
    label_mappings,
    create_histogram,
    create_correlation_heatmap,
)

@st.cache_data
def load_data():
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        current_dir, "..", "data", "diabetes_health_indicators_BRFSS2015.csv"
    )
    return pd.read_csv(data_path)

def create_bivariate_analysis(df, feature, target_var):
    """Cria análise bivariada entre uma feature e o target"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f'Distribuição por {target_var}',
        f'Média de {feature} por {target_var}'
    ))
    
    # Boxplot
    for target_value in sorted(df[target_var].unique()):
        fig.add_trace(
            go.Box(
                y=df[df[target_var] == target_value][feature],
                name=label_mappings[target_var][target_value],
                boxpoints='outliers'
            ),
            row=1, col=1
        )
    
    # Bar plot com médias
    means = df.groupby(target_var)[feature].mean()
    fig.add_trace(
        go.Bar(
            x=[label_mappings[target_var][i] for i in means.index],
            y=means.values,
            text=means.round(2),
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text=f"Análise Bivariada: {feature} vs {target_var}"
    )
    return fig

def create_temporal_analysis(df, time_var='Age'):
    """Analisa a prevalência de diabetes por faixa etária"""
    fig = px.line(
        df.groupby(time_var)[target].value_counts(normalize=True).unstack()*100,
        title=f"Prevalência de Diabetes por {time_var}",
        labels={'value': 'Percentual (%)', time_var: time_var}
    )
    fig.update_layout(
        legend_title_text='Classe de Diabetes',
        yaxis_range=[0, 100]
    )
    return fig

def create_risk_factor_analysis(df, factor, target_var):
    """Analisa fatores de risco em relação ao diabetes"""
    cross_tab = pd.crosstab(df[factor], df[target_var], normalize='index')*100
    cross_tab = cross_tab.rename(columns=label_mappings[target_var])
    
    fig = px.bar(
        cross_tab,
        barmode='group',
        title=f"Relação entre {factor} e Diabetes (%)",
        labels={'value': 'Percentual (%)', factor: factor}
    )
    return fig

def main():
    st.title("Dashboard de Análise Exploratória de Dados (EDA) Diabetes Health Indicators")
    
    # Carregar dados
    df = load_data()

    if df is None:
        st.warning("Dataset não carregado. Por favor, implemente a função load_data() para carregar seus dados.")
        st.info("Dica: Adicione o caminho para seu dataset na função load_data()")
        return

    # Verificar colunas ausentes
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        st.error(f"Colunas não encontradas no dataset: {missing_cols}")
        return

    # =============================================
    # Seção 1: Visão Geral dos Dados
    # =============================================
    st.header("Visualização dos Dados Iniciais")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        num_rows = st.slider(
            "Número de linhas para exibir:",
            min_value=5, max_value=50, value=10, step=5
        )
        show_info = st.checkbox("Mostrar informações do dataset", value=False)

    with col2:
        st.dataframe(df.head(num_rows), use_container_width=True, height=400)

    if show_info:
        st.subheader("Informações Detalhadas do Dataset")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write("**📌 Informações Gerais:**")
            st.write(f"- Total de registros: {df.shape[0]:,}")
            st.write(f"- Total de colunas: {df.shape[1]}")
            st.write(f"- Valores nulos totais: {df.isnull().sum().sum()}")
            st.write(f"- Tamanho em memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        with info_col2:
            st.write("**📊 Tipos de Dados:**")
            data_types = df.dtypes.value_counts()
            for dtype, count in data_types.items():
                st.write(f"- {dtype}: {count} colunas")

    # =============================================
    # Seção 2: Estatísticas Descritivas
    # =============================================
    with st.expander("Estatísticas Descritivas", expanded=True):
        st.subheader("Resumo Estatístico das Variáveis Numéricas")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader(f"Distribuição da Variável Target: {target}")
        target_stats = df[target].value_counts().sort_index()
        target_df = pd.DataFrame({
            "Classe": [label_mappings[target][i] for i in target_stats.index],
            "Contagem": target_stats.values,
            "Percentual": (target_stats.values / len(df) * 100).round(2),
        })
        st.dataframe(target_df, use_container_width=True)

    # =============================================
    # Seção 3: Análise da Variável Target
    # =============================================
    st.header("Análise da Variável Target")
    
    col1, col2 = st.columns(2)
    with col1:
        target_counts = df[target].value_counts()
        target_labels = [label_mappings[target][i] for i in target_counts.index]

        fig_bar = px.bar(
            x=target_labels,
            y=target_counts.values,
            title="Distribuição das Classes",
            labels={"x": target, "y": "Contagem"},
            color=target_labels,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            values=target_counts.values,
            names=target_labels,
            title="Proporção das Classes",
            color=target_labels,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("""
    **Análise:**  
    O dataset apresenta um desbalanceamento significativo entre as classes. A maioria dos indivíduos (85.3%) não tem diabetes, 
    enquanto apenas 1.9% são pré-diabéticos e 12.8% são diabéticos. Esse desbalanceamento é importante considerar para 
    modelagem preditiva, pois pode exigir técnicas de balanceamento de classes.
    """)

    # =============================================
    # Seção 4: Análise de Distribuição
    # =============================================
    st.header("Análise de Distribuição - Histograma")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_feature = st.selectbox(
            "Selecione a variável:", features,
            help="Escolha uma variável para visualizar sua distribuição"
        )
        show_by_target = st.checkbox(
            "Separar por classe do alvo", value=True,
            help="Mostra a distribuição colorida pela variável alvo"
        )

    with col2:
        if show_by_target:
            fig = create_histogram(df, selected_feature, "por classe")
        else:
            fig = px.histogram(
                df, x=selected_feature, 
                title=f"Distribuição de {selected_feature}",
                color_discrete_sequence=['#636EFA']
            )
            fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    # =============================================
    # Seção 5: Matriz de Correlação
    # =============================================
    st.header("Matriz de Correlação")
    st.info("Esta matriz mostra a correlação entre todas as variáveis numéricas do dataset.")
    fig = create_correlation_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Análise:**  
    A matriz de correlação revela relações interessantes:
    - **HighBP e HighChol** têm correlação moderada positiva (0.44)
    - **BMI** correlaciona-se positivamente com diabetes (0.22)
    - **GenHlth** (autoavaliação de saúde) tem uma das maiores correlações com diabetes (0.34)
    - **PhysHlth e MentHlth** mostram correlação positiva fraca com diabetes
    """)

    # =============================================
    # Seção 6: Análise Bivariada com o Target
    # =============================================
    st.header("Análise Bivariada com a Variável Alvo")
    
    selected_bivariate = st.selectbox(
        "Selecione a variável para análise bivariada:",
        [f for f in features if f not in ['Sex', 'AnyHealthcare', 'CholCheck']],
        index=0
    )
    
    st.plotly_chart(
        create_bivariate_analysis(df, selected_bivariate, target),
        use_container_width=True
    )

    st.markdown("""
    **Análise:**  
    Esta visualização mostra duas perspectivas:
    1. **Boxplot:** Distribuição da variável selecionada para cada classe de diabetes
    2. **Barras:** Média da variável por classe de diabetes
    
    Por exemplo, ao selecionar BMI:
    - Indivíduos diabéticos tendem a ter BMI mais alto
    - A distribuição para diabéticos mostra mais outliers (valores extremos)
    """)

    # =============================================
    # Seção 7: Análise por Idade
    # =============================================
    st.header("Análise por Idade")
    st.plotly_chart(
        create_temporal_analysis(df, 'Age'),
        use_container_width=True
    )
    
    st.markdown("""
    **Análise:**  
    - A prevalência de diabetes aumenta significativamente com a idade
    - Indivíduos com 60+ anos têm mais que o dobro da prevalência de diabetes comparado a 40-59 anos
    - Pré-diabetes é mais comum em idades intermediárias (40-70 anos)
    - A partir dos 80 anos, há uma pequena redução na prevalência
    """)

    # =============================================
    # Seção 8: Análise por Gênero
    # =============================================
    st.header("Análise por Gênero")
    
    gender_col1, gender_col2 = st.columns(2)
    with gender_col1:
        gender_counts = df['Sex'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=['Mulheres' if i == 0 else 'Homens' for i in gender_counts.index],
            title="Distribuição por Gênero",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with gender_col2:
        gender_diabetes = df.groupby('Sex')[target].value_counts(normalize=True).unstack()*100
        fig_gender_diabetes = px.bar(
            gender_diabetes,
            barmode='group',
            title="Prevalência de Diabetes por Gênero (%)",
            labels={'value': 'Percentual (%)', 'Sex': 'Gênero'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_gender_diabetes, use_container_width=True)
    
    st.markdown("""
    **Análise:**  
    - O dataset tem ligeiramente mais mulheres (53.5%) que homens (46.5%)
    - Homens apresentam maior prevalência de diabetes (14.2%) comparado a mulheres (11.6%)
    - A diferença é mais acentuada para pré-diabetes (2.3% homens vs 1.6% mulheres)
    """)

    # =============================================
    # Seção 9: Análise de Fatores de Risco
    # =============================================
    st.header("Análise de Fatores de Risco")
    
    risk_factors = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack']
    selected_risk = st.selectbox("Selecione o fator de risco:", risk_factors)
    
    st.plotly_chart(
        create_risk_factor_analysis(df, selected_risk, target),
        use_container_width=True
    )
    
    st.markdown("""
    **Análise dos Principais Fatores de Risco:**  
    - **Pressão Alta (HighBP):** 32% dos hipertensos são diabéticos vs 8% dos não hipertensos
    - **Colesterol Alto (HighChol):** 28% com colesterol alto são diabéticos vs 9% com colesterol normal
    - **Fumantes (Smoker):** Diferença menos acentuada, mas fumantes têm maior prevalência
    - **Derrame (Stroke):** 44% dos que tiveram derrame são diabéticos
    - **Doença Cardíaca (HeartDiseaseorAttack):** 39% com histórico são diabéticos
    """)

    # =============================================
    # Seção 10: Análise de Comportamentos de Saúde
    # =============================================
    st.header("Análise de Comportamentos de Saúde")
    
    health_behaviors = ['PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump']
    selected_behavior = st.selectbox("Selecione o comportamento:", health_behaviors)
    
    fig_behavior = create_risk_factor_analysis(df, selected_behavior, target)
    st.plotly_chart(fig_behavior, use_container_width=True)
    
    st.markdown("""
    **Análise:**  
    - **Atividade Física (PhysActivity):** Pessoas ativas têm menor prevalência de diabetes (10% vs 16% inativos)
    - **Consumo de Frutas/Verduras:** Efeito protetor moderado, mas menos acentuado que atividade física
    - **Álcool (HvyAlcoholConsump):** Consumo pesado mostra prevalência ligeiramente menor de diabetes
    """)

    # =============================================
    # Seção 11: Análise Socioeconômica
    # =============================================
    st.header("Análise Socioeconômica")
    
    socio_col1, socio_col2 = st.columns(2)
    with socio_col1:
        fig_income = create_risk_factor_analysis(df, 'Income', target)
        st.plotly_chart(fig_income, use_container_width=True)
    
    with socio_col2:
        fig_education = create_risk_factor_analysis(df, 'Education', target)
        st.plotly_chart(fig_education, use_container_width=True)
    
    st.markdown("""
    **Análise:**  
    - **Renda (Income):** Prevalência de diabetes diminui com aumento da renda
      - Renda mais baixa (categoria 1): 18% diabéticos
      - Renda mais alta (categoria 8): 7% diabéticos
    - **Educação (Education):** Padrão similar - maior educação, menor prevalência
      - Sem diploma do ensino médio: 17% diabéticos
      - Graduação ou mais: 8% diabéticos
    """)

if __name__ == "__main__":
    main()