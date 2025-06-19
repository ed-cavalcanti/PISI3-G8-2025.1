import plotly.express as px

features = [
    "HighBP",
    "HighChol",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
    "CholCheck",
    "AnyHealthcare",
    "NoDocbcCost",
]

target = "Diabetes_012"

label_mappings = {
    "Diabetes_012": {0: "Sem Diabetes", 1: "Pré-diabetes", 2: "Diabetes"},
    "Sex": {0: "Feminino", 1: "Masculino"},
    "HighBP": {0: "Pressão Normal", 1: "Pressão Alta"},
    "HighChol": {0: "Colesterol Normal", 1: "Colesterol Alto"},
    "Smoker": {0: "Não Fumante", 1: "Fumante"},
    "Stroke": {0: "Sem AVC", 1: "Com AVC"},
    "HeartDiseaseorAttack": {0: "Sem Doença Cardíaca", 1: "Com Doença Cardíaca"},
    "PhysActivity": {0: "Sedentário", 1: "Ativo"},
    "Fruits": {0: "Não Consome Frutas", 1: "Consome Frutas"},
    "Veggies": {0: "Não Consome Vegetais", 1: "Consome Vegetais"},
    "HvyAlcoholConsump": {0: "Baixo Consumo", 1: "Alto Consumo"},
    "DiffWalk": {0: "Sem Dificuldade", 1: "Com Dificuldade"},
}


def apply_labels(df, column):
    """Aplica labels descritivos para melhor visualização"""
    if column in label_mappings:
        df_labeled = df.copy()
        df_labeled[column] = df_labeled[column].map(label_mappings[column])
        return df_labeled, column
    return df, column


def create_correlation_heatmap(df):
    """Cria um heatmap de correlação"""
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
    )
    fig.update_layout(height=700)
    return fig


def create_histogram(df, column, title_suffix=""):
    """Cria um histograma para uma coluna específica"""
    df_plot, col_name = apply_labels(df, column)

    if target in label_mappings:
        df_plot[target] = df_plot[target].map(label_mappings[target])

    fig = px.histogram(
        df_plot,
        x=col_name,
        color=target,
        title=f"Distribuição de {column} {title_suffix}",
        nbins=30,
        barmode="overlay",
    )
    fig.update_layout(xaxis_title=column, yaxis_title="Frequência", height=400)
    return fig
