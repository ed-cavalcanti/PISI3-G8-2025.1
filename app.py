from sklearn.calibration import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd

st.title("Diabetes Health Indicators Dashboard")

DATA_PATH = "./data/diabetes_health_indicators_BRFSS2015.csv"


def get_bmi_category(BMI):
    if BMI < 18.5:
        return "underweight"
    elif 18.5 <= BMI < 24.9:
        return "normal weight"
    elif 25 <= BMI < 29.9:
        return "overweight"
    else:
        return "obese"


try:
    df = pd.read_csv(DATA_PATH)

    diabetes_map = {0: "Sem Diabetes", 1: "Pré-Diabético", 2: "Diabético"}
    df["Diabetes_Status"] = df["Diabetes_012"].map(diabetes_map)

    st.subheader("Visualização do Dataset")
    st.dataframe(df.head())

    st.subheader("Proporção de Diabetes na Amostra")

    diabetes_counts = df["Diabetes_Status"].value_counts()
    diabetes_percent = (diabetes_counts / diabetes_counts.sum()) * 100

    fig, ax = plt.subplots()
    sns.barplot(
        x=diabetes_counts.index, y=diabetes_counts.values, palette="Set2", ax=ax
    )

    for i, (count, percent) in enumerate(
        zip(diabetes_counts.values, diabetes_percent.values)
    ):
        ax.text(
            i,
            count + max(diabetes_counts.values) * 0.01,
            f"{percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Número de Pessoas")
    ax.set_title("Distribuição de Status de Diabetes na Base")
    st.pyplot(fig)

    age_map = {
        1: "18-24",
        2: "25-29",
        3: "30-34",
        4: "35-39",
        5: "40-44",
        6: "45-49",
        7: "50-54",
        8: "55-59",
        9: "60-64",
        10: "65-69",
        11: "70-74",
        12: "75-79",
        13: "80+",
    }

    df["AgeGroup"] = df["Age"].map(age_map)

    # Calculo de proporção percentual por faixa etária
    count_data = (
        df.groupby(["AgeGroup", "Diabetes_Status"]).size().reset_index(name="Count")
    )
    total_by_age = count_data.groupby("AgeGroup")["Count"].transform("sum")
    count_data["Percent"] = count_data["Count"] / total_by_age * 100
    age_order = list(age_map.values())

    # Gráficos de porcentagem de diaéticos por faixa etária
    st.subheader("Percentual de Diabetes por Faixa Etária")

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_df = count_data.pivot(
        index="AgeGroup", columns="Diabetes_Status", values="Percent"
    ).reindex(age_order)

    pivot_df.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("Percentual (%)")
    ax.set_xlabel("Faixa Etária")
    ax.set_title("Percentual de Status de Diabetes por Faixa Etária")
    plt.xticks(rotation=45)
    ax.legend(title="Status de Diabetes")
    st.pyplot(fig)

    # Gráficos de fatores de risco
    risk_factors = ["Smoker", "PhysActivity", "Veggies", "HvyAlcoholConsump"]

    risk_factor_map = {
        "Smoker": "fumantes",
        "PhysActivity": "atividades físicas",
        "Veggies": "consumo de vegetais",
        "HvyAlcoholConsump": "consumo de álcool",
    }

    for factor in risk_factors:
        st.subheader(f"Percentual de {risk_factor_map[factor]} por Status de Diabetes")

        count_data = (
            df.groupby(["Diabetes_Status", factor]).size().reset_index(name="Count")
        )
        total_by_group = count_data.groupby("Diabetes_Status")["Count"].transform("sum")
        count_data["Percent"] = count_data["Count"] / total_by_group * 100

        fig, ax = plt.subplots(figsize=(8, 5))
        pivot_df = count_data.pivot(
            index="Diabetes_Status", columns=factor, values="Percent"
        )

        pivot_df.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_ylabel("Percentual (%)")
        ax.set_xlabel("Status de Diabetes")
        ax.set_title(f"Percentual de {risk_factor_map[factor]} por Status de Diabetes")
        ax.legend(title=factor, labels=["Não", "Sim"])
        st.pyplot(fig)

    # Feature importance
    
    df2 = pd.read_csv("./data/diabetes_health_indicators_BRFSS2015.csv")

    # Separar X e y
    X = df2.drop(columns=["Diabetes_012"])
    X = X.drop(columns=["Income"])
    X = X.drop(columns=["Education"])
    y = df2["Diabetes_012"]

    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = le.fit_transform(y)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_

    importance_df2 = pd.DataFrame({"feature": X.columns, "importance": importances})
    top5 = importance_df2.sort_values(by="importance", ascending=False).head(5)

    st.header("Top 5 fatores de risco para diabetes (multiclasse) árvore de decisão")
    fig, ax = plt.subplots()
    sns.barplot(
        x="importance", y="feature", data=top5, palette="Set2", ax=ax
    )
    plt.xlabel("Importância")
    plt.ylabel("Fator de Risco")
    plt.title("Importância dos 5 Principais Fatores de Risco para Diabetes")
    plt.xlim(0, top5["importance"].max() + 0.02)
    st.pyplot(fig)

except FileNotFoundError:
    st.error(
        f"Arquivo não encontrado: {DATA_PATH}. Verifique se ele está no mesmo diretório do script."
    )
