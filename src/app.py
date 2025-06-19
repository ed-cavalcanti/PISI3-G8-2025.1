import streamlit as st
import pandas as pd
import plotly.express as px
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


def main():
    st.title(
        "Dashboard de Análise Exploratória de Dados (EDA) Diabetes Health Indicators"
    )

    df = load_data()

    if df is None:
        st.warning(
            "Dataset não carregado. Por favor, implemente a função load_data() para carregar seus dados."
        )
        st.info("Dica: Adicione o caminho para seu dataset na função load_data()")
        return

    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        st.error(f"Colunas não encontradas no dataset: {missing_cols}")
        return

    # Container principal

    st.header("Visualização dos Dados Iniciais")

    col1, col2 = st.columns([1, 3])

    with col1:
        num_rows = st.slider(
            "Número de linhas para exibir:",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Escolha quantas linhas deseja visualizar",
        )

        show_info = st.checkbox(
            "Mostrar informações do dataset",
            value=False,
            help="Exibe informações detalhadas sobre tipos de dados e valores nulos",
        )

    with col2:
        st.dataframe(
            df.head(num_rows),
            use_container_width=True,
            height=400,
        )

    if show_info:
        st.subheader("Informações Detalhadas do Dataset")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.write("**Informações Gerais:**")
            st.write(f"- **Total de registros:** {df.shape[0]:,}")
            st.write(f"- **Total de colunas:** {df.shape[1]}")
            st.write(f"- **Valores nulos totais:** {df.isnull().sum().sum()}")
            st.write(
                f"- **Tamanho em memória:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )

        with info_col2:
            st.write("**Tipos de Dados:**")
            data_types = df.dtypes.value_counts()
            for dtype, count in data_types.items():
                st.write(f"- **{dtype}:** {count} colunas")

    with st.expander("Estatísticas Descritivas"):
        st.subheader("Resumo Estatístico das Variáveis Numéricas")
        st.dataframe(
            df.describe(),
            use_container_width=True,
        )

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

    st.header("Análise da Variável alvo")
    col1, col2 = st.columns(2)
    with col1:
        target_counts = df[target].value_counts()
        target_labels = [label_mappings[target][i] for i in target_counts.index]

        fig_bar = px.bar(
            x=target_labels,
            y=target_counts.values,
            title="Distribuição das Classes",
            labels={"x": target, "y": "Contagem"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            values=target_counts.values,
            names=target_labels,
            title="Proporção das Classes",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

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
                df, x=selected_feature, title=f"Distribuição de {selected_feature}"
            )
            fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    st.header("Matriz de Correlação")
    st.info(
        "Esta matriz mostra a correlação entre todas as variáveis numéricas do dataset."
    )
    fig = create_correlation_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
