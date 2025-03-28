import streamlit as st
from PIL import Image, ImageFilter
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
import time

st.set_page_config(page_title="EfficientNet Классификация", layout="wide")
class_names = ["benign", "malignant"]


@st.cache_resource
def load_model():
    model = torch.load("models/cancer.pt", map_location="cpu", weights_only=False)
    model.eval()
    return model


model = load_model()

# Преобразования для EfficientNet
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


st.title("Классификация рака кожи с помощью EfficientNet")

source = st.radio("Источник изображения:", ["Файл", "Ссылка"])
images = []

if source == "Файл":
    uploaded = st.sidebar.file_uploader(
        "Загрузите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if uploaded:
        for f in uploaded:
            img = Image.open(f).convert("RGB")
            images.append(img)

elif source == "Ссылка":
    url = st.text_input("Вставьте ссылку на изображение")
    if url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(img)
        except:
            st.error("Ошибка загрузки изображения.")


if images:
    for img in images:
        blurred_img = img.filter(
            ImageFilter.GaussianBlur(radius=5)
        )  # Apply blur with a radius of 5 (Adjust as needed)

        st.image(blurred_img, caption="Загруженное изображение (размытое)", width=300)

        input_tensor = transform(img).unsqueeze(
            0
        )  # Using original image for classification

        start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        elapsed = time.time() - start

        prob_class = torch.sigmoid(output).detach().cpu().numpy().round(1)
        st.success(f"Класс: {class_names[int(prob_class)]}")
        st.info(f"Время обработки: {elapsed:.3f} сек")
