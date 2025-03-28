# Fine-Tuned Модели для Классификации Изображений

**Группа 2: Креативные Кадры + Свин** <br/>
[Александр Годелашвили](https://github.com/tumpaksewu) ——
[Зиярат Гаджиева](https://github.com/Ziiarat) ——
[Карина Ходжигорова](https://github.com/KarinaKhod)

<br/>


---

## 📖 Описание
В данном репозитории содержатся:
- 🎯 **SWIN Transformer Model** — обучена на классификацию 6 классов из датасета [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data).
- 🔬 **EfficientNet Model** — обучена для бинарной классификации рака кожи на основе датасета [Skin Cancer: Malignant vs. Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign?select=train).

В репозитории также представлены:
- 📒 Jupyter ноутбуки с процессом обучения и дообучения моделей.
- 💾 Сохраненные модели с весами.
- 🌐 Streamlit демо-страница для проверки работы моделей.

---

## 💻 Установка
```bash
# Клонируйте репозиторий
$ git clone https://github.com/yourusername/yourrepository.git

# Перейдите в директорию проекта
$ cd yourrepository

# Установите зависимости
$ pip install -r requirements.txt
```

---

## 🚀 Использование
### Запуск Streamlit демо-страницы
```bash
streamlit run main.py
```

---

## 🔍 Модели
1. **SWIN Transformer**
   - Классифицирует изображения на 6 категорий: ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'].
   - Точность на тестовом наборе: **XX.XX%** (указать точность).

2. **EfficientNet**
   - Бинарная классификация изображений на **злокачественные** и **доброкачественные**.
   - Точность на тестовом наборе: **XX.XX%** (указать точность).

---

## 🌐 Демо
Streamlit приложение предоставляет удобный интерфейс для проверки обеих моделей.

```bash
streamlit run demo_app.py
```

Перейдите по локальному адресу, указанному в консоли, чтобы воспользоваться приложением.


