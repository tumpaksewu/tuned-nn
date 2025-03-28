import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Model Training Dashboard", layout="wide")

# --- Блок 1: Процесс обучения модели ---
st.header("📈 Процесс обучения модели")
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Кривые обучения")
    st.image("images/train-plot.jpeg", use_container_width=True)


with col2:
    subcol1, subcol2 = st.columns(2)

    with subcol1:
        st.subheader("Метрики")
        st.markdown(
            '<h2 style="text-align: center;">Accuracy SWIN: 0.95!</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h2 style="text-align: center;">F1 Score EffNet: 0.9</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h2 style="text-align: center;">Общее время обучения </br> на 4070S Ti: </br> 34:00 </h2>',
            unsafe_allow_html=True,
        )

    with subcol2:
        st.subheader("Confusion Matrix")
        conf_matrix = np.array([[8477, 523], [660, 5340]])
        plt.style.use("dark_background")
        plt.figure(figsize=(5, 5), dpi=100)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="magma", ax=ax)
        st.pyplot(fig)


# --- Блок 3: Состав датасета ---
st.header("📊 Состав датасетов")
dataset_size1 = 24350
dataset_size2 = 3297
num_classes = 5
class_distribution1 = {
    "buildings": 2191,
    "forest": 2271,
    "glacier": 2404,
    "mountain": 2512,
    "sea": 2274,
    "street": 2382,
}
class_distribution2 = {"benign": 1630, "malignant": 1680}

col3, col4 = st.columns(2)

with col3:
    st.metric(label="Объектов в датасете 1:", value=f"{dataset_size1:,}")
    st.metric(label="Объектов в датасете 2:", value=f"{dataset_size2:,}")

with col4:
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        st.write("### Распределение по классам (датасет 1)")
        st.bar_chart(class_distribution1)
    with subcol2:
        st.write("### Распределение по классам (датасет 2)")
        st.bar_chart(class_distribution2)
