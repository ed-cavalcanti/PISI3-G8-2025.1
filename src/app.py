import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import pickle
import joblib
import os
from components.visualizations import (
    features,
    target,
    label_mappings,
    create_histogram,
    create_correlation_heatmap,
)


@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        current_dir, "..", "data", "diabetes_health_indicators_BRFSS2015.csv"
    )
    return pd.read_csv(data_path)


@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(current_dir, "..", "models")

        # Carregar modelo - tentar primeiro com joblib, depois pickle
        model_joblib_path = os.path.join(models_path, "xgb_diabetes_model.joblib")
        model_pkl_path = os.path.join(models_path, "xgb_diabetes_model.pkl")

        if os.path.exists(model_joblib_path):
            model = joblib.load(model_joblib_path)
        elif os.path.exists(model_pkl_path):
            with open(model_pkl_path, "rb") as f:
                model = pickle.load(f)
        else:
            raise FileNotFoundError("Nenhum arquivo de modelo encontrado")

        scaler_joblib_path = os.path.join(models_path, "scaler.joblib")
        scaler_pkl_path = os.path.join(models_path, "scaler.pkl")

        if os.path.exists(scaler_joblib_path):
            scaler = joblib.load(scaler_joblib_path)
        elif os.path.exists(scaler_pkl_path):
            with open(scaler_pkl_path, "rb") as f:
                scaler = pickle.load(f)
        else:
            raise FileNotFoundError("Nenhum arquivo de scaler encontrado")

        model_info_path = os.path.join(models_path, "model_info.pkl")
        with open(model_info_path, "rb") as f:
            model_info = pickle.load(f)

        return model, scaler, model_info
    except FileNotFoundError as e:
        st.error(f"Modelo não encontrado! {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None, None


def preprocess_input(input_data, scaler, model_info):
    try:
        model_features = model_info.get("features", [])

        if len(input_data) != len(model_features):
            st.error(
                f"Erro: Esperado {len(model_features)} features, recebido {len(input_data)}"
            )
            return None

        df = pd.DataFrame([input_data], columns=model_features)

        if "cols_to_scale" in model_info and scaler is not None:
            cols_to_scale = model_info["cols_to_scale"]
            cols_to_scale = [col for col in cols_to_scale if col in df.columns]
            if cols_to_scale:
                df[cols_to_scale] = scaler.transform(df[cols_to_scale])

        return df.values
    except Exception as e:
        st.error(f"Erro no pré-processamento: {e}")
        return None


def create_bivariate_analysis(df, feature, target_var):
    """Cria análise bivariada entre uma feature e o target"""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Distribuição por {target_var}",
            f"Média de {feature} por {target_var}",
        ),
    )

    # Boxplot
    for target_value in sorted(df[target_var].unique()):
        fig.add_trace(
            go.Box(
                y=df[df[target_var] == target_value][feature],
                name=label_mappings[target_var][target_value],
                boxpoints="outliers",
            ),
            row=1,
            col=1,
        )

    # Bar plot com médias
    means = df.groupby(target_var)[feature].mean()
    fig.add_trace(
        go.Bar(
            x=[label_mappings[target_var][i] for i in means.index],
            y=means.values,
            text=means.round(2),
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        title_text=f"Análise Bivariada: {feature} vs {target_var}",
    )
    return fig


def create_temporal_analysis(df, time_var="Age"):
    """Analisa a prevalência de diabetes por faixa etária"""
    fig = px.line(
        df.groupby(time_var)[target].value_counts(normalize=True).unstack() * 100,
        title=f"Prevalência de Diabetes por {time_var}",
        labels={"value": "Percentual (%)", time_var: time_var},
    )
    fig.update_layout(legend_title_text="Classe de Diabetes", yaxis_range=[0, 100])
    return fig


def create_risk_factor_analysis(df, factor, target_var):
    """Analisa fatores de risco em relação ao diabetes"""
    cross_tab = pd.crosstab(df[factor], df[target_var], normalize="index") * 100
    cross_tab = cross_tab.rename(columns=label_mappings[target_var])

    fig = px.bar(
        cross_tab,
        barmode="group",
        title=f"Relação entre {factor} e Diabetes (%)",
        labels={"value": "Percentual (%)", factor: factor},
    )
    return fig


def prediction_page():
    st.title("Predição de Diabetes")
    st.write(
        "Este dashboard utiliza um modelo XGBoost para prever o risco de diabetes."
    )

    model, scaler, model_info = load_model()

    if model is None:
        st.error(
            "Não foi possível carregar o modelo. Verifique se os arquivos estão na pasta 'models'."
        )
        return

    st.sidebar.header("Informações do Modelo")
    if model_info:
        if "best_score" in model_info:
            st.sidebar.metric("Melhor Score CV", f"{model_info['best_score']:.4f}")

        if "best_params" in model_info:
            st.sidebar.write("**Melhores Parâmetros:**")
            for param, value in model_info["best_params"].items():
                st.sidebar.write(f"- {param}: {value}")

        if "features" in model_info:
            st.sidebar.write("**📊 Features Utilizadas pelo Modelo:**")
            for i, feature in enumerate(model_info["features"], 1):
                st.sidebar.write(f"{i:2d}. {feature}")

    st.info(
        """
    **Importante:** Este modelo foi treinado com 16 variáveis específicas. 
    Algumas informações coletadas (como Sexo, Frutas, e dados de acesso à saúde) 
    não são utilizadas na predição, mas estão disponíveis para referência.
    """
    )

    st.header("Insira os dados para predição:")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Condições de Saúde")
        high_bp = st.selectbox(
            "Pressão Alta", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim"
        )
        high_chol = st.selectbox(
            "Colesterol Alto", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim"
        )
        bmi = st.number_input(
            "IMC (Índice de Massa Corporal)",
            min_value=10.0,
            max_value=50.0,
            value=25.0,
            step=0.1,
        )
        stroke = st.selectbox(
            "Histórico de AVC", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim"
        )
        heart_disease = st.selectbox(
            "Doença Cardíaca ou Ataque do Coração",
            [0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
        )

        st.subheader("Hábitos de Vida")
        smoker = st.selectbox(
            "Fumante", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim"
        )
        phys_activity = st.selectbox(
            "Atividade Física Regular",
            [0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
        )

        fruits = st.selectbox(
            "Consome Frutas Regularmente (não usado pelo modelo)",
            [0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Esta informação é coletada mas não é usada pelo modelo atual",
        )
        veggies = st.selectbox(
            "Consome Vegetais Regularmente",
            [0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
        )
        heavy_alcohol = st.selectbox(
            "Consumo Pesado de Álcool",
            [0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
        )

    with col2:
        st.subheader("Estado de Saúde Atual")
        gen_health = st.slider(
            "Saúde Geral",
            1,
            5,
            3,
            help="1=Excelente, 2=Muito Boa, 3=Boa, 4=Razoável, 5=Ruim",
        )
        mental_health = st.slider("Dias de Saúde Mental Ruim (último mês)", 0, 30, 0)
        phys_health = st.slider("Dias de Saúde Física Ruim (último mês)", 0, 30, 0)
        diff_walk = st.selectbox(
            "Dificuldade para Caminhar ou Subir Escadas",
            [0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
        )

        st.subheader("Características Pessoais")
        sex = st.selectbox(
            "Sexo (não usado pelo modelo)",
            [0, 1],
            format_func=lambda x: "Feminino" if x == 0 else "Masculino",
            help="Esta informação é coletada mas não é usada pelo modelo atual",
        )
        age = st.slider(
            "Faixa Etária",
            1,
            13,
            7,
            help="1 = 18-24, 2 = 25-29, 3 = 30-34, 4 = 35-39, 5 = 40-44, 6 = 45-49, 7 = 50-54, 8 = 55-59, 9 = 60-64, 10 = 65-69, 11 = 70-74, 12 = 75-79, 13=80+",
        )
        education = st.slider(
            "Nível de Educação",
            1,
            6,
            4,
            help="1=Nunca frequentou escola, 2=Fundamental incompleto, 3=Fundamental completo, 4=Ensino médio completo, 5=Faculdade incompleta, 6=Faculdade completa",
        )
        income = st.slider(
            "Faixa de Renda",
            1,
            8,
            5,
            help="1=<$10k, 2=$10k-$15k, 3=$15k-$20k, 4=$20k-$25k, 5=$25k-$35k, 6=$35k-$50k, 7=$50k-$75k, 8=>$75k",
        )

        st.subheader("Acesso aos Cuidados de Saúde")
        chol_check = st.selectbox(
            "Verificou Colesterol (últimos 5 anos) - não usado pelo modelo",
            [0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Esta informação é coletada mas não é usada pelo modelo atual",
        )
        any_healthcare = st.selectbox(
            "Tem Plano de Saúde (não usado pelo modelo)",
            [0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Esta informação é coletada mas não é usada pelo modelo atual",
        )
        no_doc_bc_cost = st.selectbox(
            "Deixou de ir ao médico por custo (não usado pelo modelo)",
            [0, 1],
            format_func=lambda x: "Não" if x == 0 else "Sim",
            help="Esta informação é coletada mas não é usada pelo modelo atual",
        )

    input_data = [
        high_bp,
        high_chol,
        bmi,
        smoker,
        stroke,
        heart_disease,
        phys_activity,
        veggies,
        heavy_alcohol,
        gen_health,
        mental_health,
        phys_health,
        diff_walk,
        age,
        education,
        income,
    ]

    if st.button("Fazer Predição", type="primary", use_container_width=True):
        with st.spinner("Processando predição..."):
            processed_data = preprocess_input(input_data, scaler, model_info)

            if processed_data is not None:
                try:
                    prediction = model.predict(processed_data)[0]
                    probabilities = model.predict_proba(processed_data)[0]

                    st.header("Resultado da Predição:")

                    class_names = ["Sem Diabetes", "Diabetes"]
                    predicted_class = class_names[prediction]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if prediction == 0:
                            st.success(f"✅ **{predicted_class.upper()}**")
                        else:
                            st.error(f"🚨 **{predicted_class.upper()}**")

                    with col2:
                        max_prob = max(probabilities)
                        st.metric("Confiança da Predição", f"{max_prob:.1%}")

                    with col3:
                        diabetes_prob = probabilities[1]
                        st.metric("Probabilidade de Diabetes", f"{diabetes_prob:.1%}")

                    st.subheader("📈 Probabilidades por Classe:")
                    prob_df = pd.DataFrame(
                        {
                            "Classe": class_names,
                            "Probabilidade": probabilities,
                            "Percentual": [f"{p:.1%}" for p in probabilities],
                        }
                    )

                    fig = px.bar(
                        prob_df,
                        x="Classe",
                        y="Probabilidade",
                        text="Percentual",
                        color="Classe",
                        color_discrete_map={
                            "Sem Diabetes": "#28a745",
                            "Diabetes": "#dc3545",
                        },
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(
                        showlegend=False,
                        yaxis_title="Probabilidade",
                        title="Distribuição de Probabilidades",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Recomendações:")

                    if prediction == 0 and max_prob > 0.8:
                        st.success(
                            """
                        🎉 **Excelente!** Você apresenta baixo risco de diabetes.
                        - Continue mantendo seus hábitos saudáveis
                        - Mantenha atividade física regular
                        - Faça check-ups médicos preventivos
                        """
                        )
                    elif prediction == 0:
                        st.info(
                            """
                        ✅ **Bom!** Baixo risco, mas há margem para melhorar.
                        - Considere otimizar sua dieta e exercícios
                        - Monitore regularmente sua saúde
                        """
                        )
                    else:
                        st.error(
                            """
                        🚨 **Alerta!** Alto risco de diabetes detectado.
                        - **Procure atendimento médico imediatamente**
                        - Siga rigorosamente as orientações médicas
                        - Monitore constantemente a glicemia
                        - Adote mudanças significativas no estilo de vida
                        """
                        )

                    st.info(
                        """
                    **⚠️ Importante:** Esta predição é baseada em um modelo estatístico e não substitui 
                    a avaliação médica profissional. Sempre consulte um médico para diagnóstico e 
                    tratamento adequados.
                    """
                    )

                except Exception as e:
                    st.error(f"Erro ao fazer predição: {e}")


def eda_page():
    """Página de análise exploratória (código original)"""
    st.title(
        "Dashboard de Análise Exploratória de Dados (EDA) Diabetes Health Indicators"
    )

    # Carregar dados
    df = load_data()

    if df is None:
        st.warning(
            "Dataset não carregado. Por favor, implemente a função load_data() para carregar seus dados."
        )
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
            "Número de linhas para exibir:", min_value=5, max_value=50, value=10, step=5
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
            st.write(
                f"- Tamanho em memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )

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
        target_df = pd.DataFrame(
            {
                "Classe": [label_mappings[target][i] for i in target_stats.index],
                "Contagem": target_stats.values,
                "Percentual": (target_stats.values / len(df) * 100).round(2),
            }
        )
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
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            values=target_counts.values,
            names=target_labels,
            title="Proporção das Classes",
            color=target_labels,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(
        """
    **Análise:**  
    O dataset apresenta um desbalanceamento significativo entre as classes. A maioria dos indivíduos (85.3%) não tem diabetes, 
    enquanto apenas 1.9% são pré-diabéticos e 12.8% são diabéticos. Esse desbalanceamento é importante considerar para 
    modelagem preditiva, pois pode exigir técnicas de balanceamento de classes.
    """
    )

    # =============================================
    # Seção 4: Análise de Distribuição
    # =============================================
    st.header("Análise de Distribuição - Histograma")

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_feature = st.selectbox(
            "Selecione a variável:",
            features,
            help="Escolha uma variável para visualizar sua distribuição",
        )
        show_by_target = st.checkbox(
            "Separar por classe do alvo",
            value=True,
            help="Mostra a distribuição colorida pela variável alvo",
        )

    with col2:
        if show_by_target:
            fig = create_histogram(df, selected_feature, "por classe")
        else:
            fig = px.histogram(
                df,
                x=selected_feature,
                title=f"Distribuição de {selected_feature}",
                color_discrete_sequence=["#636EFA"],
            )
            fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    # =============================================
    # Seção 5: Matriz de Correlação
    # =============================================
    st.header("Matriz de Correlação")
    st.info(
        "Esta matriz mostra a correlação entre todas as variáveis numéricas do dataset."
    )
    fig = create_correlation_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
    **Análise:**  
    A matriz de correlação revela relações interessantes:
    - **HighBP e HighChol** têm correlação moderada positiva (0.44)
    - **BMI** correlaciona-se positivamente com diabetes (0.22)
    - **GenHlth** (autoavaliação de saúde) tem uma das maiores correlações com diabetes (0.34)
    - **PhysHlth e MentHlth** mostram correlação positiva fraca com diabetes
    """
    )

    # =============================================
    # Seção 6: Análise Bivariada com o Target
    # =============================================
    st.header("Análise Bivariada com a Variável Alvo")

    selected_bivariate = st.selectbox(
        "Selecione a variável para análise bivariada:",
        [f for f in features if f not in ["Sex", "AnyHealthcare", "CholCheck"]],
        index=0,
    )

    st.plotly_chart(
        create_bivariate_analysis(df, selected_bivariate, target),
        use_container_width=True,
    )

    st.markdown(
        """
    **Análise:**  
    Esta visualização mostra duas perspectivas:
    1. **Boxplot:** Distribuição da variável selecionada para cada classe de diabetes
    2. **Barras:** Média da variável por classe de diabetes
    
    Por exemplo, ao selecionar BMI:
    - Indivíduos diabéticos tendem a ter BMI mais alto
    - A distribuição para diabéticos mostra mais outliers (valores extremos)
    """
    )

    # =============================================
    # Seção 7: Análise por Idade
    # =============================================
    st.header("Análise por Idade")
    st.plotly_chart(create_temporal_analysis(df, "Age"), use_container_width=True)

    st.markdown(
        """
    **Análise:**  
    - A prevalência de diabetes aumenta significativamente com a idade
    - Indivíduos com 60+ anos têm mais que o dobro da prevalência de diabetes comparado a 40-59 anos
    - Pré-diabetes é mais comum em idades intermediárias (40-70 anos)
    - A partir dos 80 anos, há uma pequena redução na prevalência
    """
    )

    # =============================================
    # Seção 8: Análise por Gênero
    # =============================================
    st.header("Análise por Gênero")

    gender_col1, gender_col2 = st.columns(2)
    with gender_col1:
        gender_counts = df["Sex"].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=["Mulheres" if i == 0 else "Homens" for i in gender_counts.index],
            title="Distribuição por Gênero",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    with gender_col2:
        gender_diabetes = (
            df.groupby("Sex")[target].value_counts(normalize=True).unstack() * 100
        )
        fig_gender_diabetes = px.bar(
            gender_diabetes,
            barmode="group",
            title="Prevalência de Diabetes por Gênero (%)",
            labels={"value": "Percentual (%)", "Sex": "Gênero"},
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_gender_diabetes, use_container_width=True)

    st.markdown(
        """
    **Análise:**  
    - O dataset tem ligeiramente mais mulheres (53.5%) que homens (46.5%)
    - Homens apresentam maior prevalência de diabetes (14.2%) comparado a mulheres (11.6%)
    - A diferença é mais acentuada para pré-diabetes (2.3% homens vs 1.6% mulheres)
    """
    )

    # =============================================
    # Seção 9: Análise de Fatores de Risco
    # =============================================
    st.header("Análise de Fatores de Risco")

    risk_factors = [
        "HighBP",
        "HighChol",
        "BMI",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
    ]
    selected_risk = st.selectbox("Selecione o fator de risco:", risk_factors)

    st.plotly_chart(
        create_risk_factor_analysis(df, selected_risk, target), use_container_width=True
    )

    st.markdown(
        """
    **Análise dos Principais Fatores de Risco:**  
    - **Pressão Alta (HighBP):** 32% dos hipertensos são diabéticos vs 8% dos não hipertensos
    - **Colesterol Alto (HighChol):** 28% com colesterol alto são diabéticos vs 9% com colesterol normal
    - **Fumantes (Smoker):** Diferença menos acentuada, mas fumantes têm maior prevalência
    - **Derrame (Stroke):** 44% dos que tiveram derrame são diabéticos
    - **Doença Cardíaca (HeartDiseaseorAttack):** 39% com histórico são diabéticos
    """
    )

    # =============================================
    # Seção 10: Análise de Comportamentos de Saúde
    # =============================================
    st.header("Análise de Comportamentos de Saúde")

    health_behaviors = ["PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"]
    selected_behavior = st.selectbox("Selecione o comportamento:", health_behaviors)

    fig_behavior = create_risk_factor_analysis(df, selected_behavior, target)
    st.plotly_chart(fig_behavior, use_container_width=True)

    st.markdown(
        """
    **Análise:**  
    - **Atividade Física (PhysActivity):** Pessoas ativas têm menor prevalência de diabetes (10% vs 16% inativos)
    - **Consumo de Frutas/Verduras:** Efeito protetor moderado, mas menos acentuado que atividade física
    - **Álcool (HvyAlcoholConsump):** Consumo pesado mostra prevalência ligeiramente menor de diabetes
    """
    )

    # =============================================
    # Seção 11: Análise Socioeconômica
    # =============================================
    st.header("Análise Socioeconômica")

    socio_col1, socio_col2 = st.columns(2)
    with socio_col1:
        fig_income = create_risk_factor_analysis(df, "Income", target)
        st.plotly_chart(fig_income, use_container_width=True)

    with socio_col2:
        fig_education = create_risk_factor_analysis(df, "Education", target)
        st.plotly_chart(fig_education, use_container_width=True)

    st.markdown(
        """
    **Análise:**  
    - **Renda (Income):** Prevalência de diabetes diminui com aumento da renda
      - Renda mais baixa (categoria 1): 18% diabéticos
      - Renda mais alta (categoria 8): 7% diabéticos
    - **Educação (Education):** Padrão similar - maior educação, menor prevalência
      - Sem diploma do ensino médio: 17% diabéticos
      - Graduação ou mais: 8% diabéticos
    """
    )


def main():
    """Função principal com navegação entre páginas"""
    # Configurar a página
    st.set_page_config(
        page_title="Dashboard Diabetes Health Indicators",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar para navegação
    st.sidebar.title("🏥 Dashboard Diabetes")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Escolha uma página:",
        ["📊 Análise Exploratória", "🔮 Predição de Diabetes"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    ### ℹ️ Sobre o Projeto
    Este dashboard analisa indicadores de saúde relacionados ao diabetes 
    usando dados do BRFSS 2015.
    
    **Funcionalidades:**
    - Análise exploratória completa dos dados
    - Predição de diabetes usando Machine Learning
    - Visualizações interativas
    """
    )

    # Renderizar página selecionada
    if page == "📊 Análise Exploratória":
        eda_page()
    elif page == "🔮 Predição de Diabetes":
        prediction_page()


if __name__ == "__main__":
    main()
