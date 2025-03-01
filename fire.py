import streamlit as st  # type: ignore
import cv2
from ultralytics import YOLO
import requests  # type: ignore
from PIL import Image
import os
import io
import tempfile
import numpy as np
from typing import Tuple
import asyncio
from dataclasses import dataclass
from collections import defaultdict

# --- Configuration ---
WEBCAM_SOURCE = 0  # Default webcam source, change if needed

# --- Styles ---
SIDEBAR_BG_COLOR = "linear-gradient(to bottom, #2e2e2e, #0e0e0e)"  # Gradient black
TEXT_COLOR = "white"

# --- Application Config ---
st.set_page_config(
    page_title="Детектор светящихся объектов", initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@dataclass
class DetectionStats:
    """Класс для хранения статистики детекции"""

    class_counts: dict[str, int]
    total_objects: int
    processing_time: float


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """Загрузка модели YOLO с кэшированием"""
    model = YOLO(model_path)
    return model


async def process_video_async(
    model: YOLO, video_file: str, conf_threshold: float, iou_threshold: float, st_frame
) -> None:
    """Асинхронная обработка видео"""
    try:
        vid = cv2.VideoCapture(video_file)
        if not vid.isOpened():
            st.error("Ошибка открытия видео")
            return

        alert_placeholder = st.empty()  # Placeholder for alert message

        while True:
            ret, frame = vid.read()
            if not ret:
                st.write("Конец видео.")
                break

            # Изменить размер кадра для повышения производительности
            frame = cv2.resize(frame, (640, 480))

            # Прогнозирование на кадре
            res = model.predict(
                frame, conf=conf_threshold, iou=iou_threshold, device="cpu"
            )

            res_image = res[0].plot()
            res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

            # Обновление кадра Streamlit
            st_frame.image(res_image, channels="RGB", use_container_width=True)

            # Обновление статистики
            class_counts = defaultdict(int)
            fire_detected = False
            for c in res[0].boxes.cls:
                class_name = model.model.names[int(c)]
                class_counts[class_name] += 1
                if class_name in ["fire", "smoke"]:
                    fire_detected = True

            # stats = DetectionStats(
            #     class_counts=class_counts,
            #     total_objects=sum(class_counts.values()),
            #     processing_time=sum(res[0].speed.values()) / 1000,
            # )
            # update_stats_display(stats)

            # Вывод сообщения об обнаружении возгорания или нормальном состоянии
            if fire_detected:
                alert_placeholder.markdown(
                    "<p style='color:red; font-size:20px;'><b>Внимание! Тревога. Обнаружен признак возгорания!</b></p>",
                    unsafe_allow_html=True,
                )
            else:
                alert_placeholder.markdown(
                    "<p style='color:blue; font-size:20px;'><b>Нормальное состояние</b></p>",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"Ошибка обработки видео: {e}")
    finally:
        if isinstance(video_file, int):
            vid.release()
        else:
            vid.release()


def predict_image(
    model: YOLO, image: Image.Image, conf_threshold: float, iou_threshold: float
) -> Tuple[np.ndarray, str, DetectionStats]:
    """Детекция объектов на изображении"""
    # Детектировать класс объекта, применив модель
    res = model.predict(image, conf=conf_threshold, iou=iou_threshold, device="cpu")

    class_counts = defaultdict(int)
    for c in res[0].boxes.cls:
        class_counts[model.model.names[int(c)]] += 1

    stats = DetectionStats(
        class_counts=class_counts,
        total_objects=sum(class_counts.values()),
        processing_time=sum(res[0].speed.values()) / 1000,
    )

    # Создать текст с результатом детекции класса с помощью модели
    prediction_text = "Результат детекции: "
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f"{v} {k}"
        if v > 1:
            prediction_text += "s"
        prediction_text += ", "

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "Заданных классов объектов не обнаружено"

    prediction_text += f", время детекции: {stats.processing_time:.2f} секунд."

    # Преобразовать изображение в RGB
    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
    return res_image, prediction_text, stats


# --- Main App ---
def main():
    # --- Sidebar Styling ---
    st.markdown(
        f"""
    <style>
        [data-testid="stSidebar"] {{
            background: {SIDEBAR_BG_COLOR};
            color: {TEXT_COLOR};
        }}
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {{
            color: {TEXT_COLOR};
        }}
        [data-testid="stSidebar"] .stRadio > label {{
            color: {TEXT_COLOR};
        }}
        [data-testid="stSidebar"] .stSelectbox > label {{
            color: {TEXT_COLOR};
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # --- Sidebar Content ---
    with st.sidebar:
        # --- Application Info ---
        st.markdown(
            f"<h1 style='color: {TEXT_COLOR};'>Инфо</h1>", unsafe_allow_html=True
        )
        st.markdown("<div style='text-align: center; border-top: 1px solid white; width: 1000%; margin: auto;'></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<p style='color: {TEXT_COLOR};'>Автор: Семенов Артём</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {TEXT_COLOR};'>Заказчик: ЦИТМ Экспонента</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {TEXT_COLOR};'>Школа: 2107</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {TEXT_COLOR};'>   </p>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; border-top: 1px solid white; width: 1000%; margin: auto;'></div>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {TEXT_COLOR};'>   </p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {TEXT_COLOR};'>   </p>", unsafe_allow_html=True)


        # Model Selection
        st.markdown(
            f"<h1 style='color: {TEXT_COLOR};'>Модель</h1>", unsafe_allow_html=True
        )

        model_type = st.radio(
            "Тип", ("Пожар", "Базовая"), index=0,
        )

        models_dir = "general-models" if model_type == "Базовая" else "fire-models"
        model_files = [
            f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")
        ]

        # Ensure there are model files to select from
        if model_files:
            selected_model = st.selectbox(
                "Размер", sorted(model_files), index=0
            )
        else:
            st.warning("No model files found in the specified directory.")
            selected_model = None
        st.markdown("<div style='text-align: center; border-top: 1px solid white; width: 1000%; margin: auto;'></div>",
                    unsafe_allow_html=True)

        st.markdown(f"<p style='color: {TEXT_COLOR};'>   </p>", unsafe_allow_html=True)
        # Load the selected model
        if selected_model:
            model_path = os.path.join(
                models_dir, selected_model + ".pt"
            )  # type: ignore
            model = load_model(model_path)

            # Confidence and IOU thresholds
            st.markdown(f"<p style='color: {TEXT_COLOR};'>   </p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {TEXT_COLOR};'>   </p>", unsafe_allow_html=True)
            st.markdown(
                f"<h1 style='color: {TEXT_COLOR};'>Параметры обнаружения</h1>", unsafe_allow_html=True
            )
            conf_threshold = st.slider("Точность детекции", 0.0, 1.0, 0.20, 0.05)
            iou_threshold = st.slider(
                "Перекрытие между рамками",
                0.0,
                1.0,
                0.5,
                0.05,
            )

            st.markdown(f"<p style='color: {TEXT_COLOR};'>   </p>", unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align: center; border-top: 1px solid white; width: 1000%; margin: auto;'></div>",
                unsafe_allow_html=True)
            # Source selection
            st.markdown(f"<p style='color: {TEXT_COLOR};'>   </p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {TEXT_COLOR};'>   </p>", unsafe_allow_html=True)
            st.markdown(
                f"<h1 style='color: {TEXT_COLOR};'>Входные данные</h1>", unsafe_allow_html=True
            )
            source_type = st.radio(
                "Выберите источник данных:", ("Изображение", "Видео", "Веб-камера")
            )
            st.markdown(
                "<div style='text-align: center; border-top: 1px solid white; width: 1000%; margin: auto;'></div>",
                unsafe_allow_html=True)
        else:
            model = None
            source_type = None
            conf_threshold = None
            iou_threshold = None

    # --- Main content area ---
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(r"g:\Pyprojects\IntellectDetector\pngegg.jpg", width=100)

    with col2:

        st.markdown(f"<h1 style='white-space: nowrap;'>Детектор светящихся объектов</h1>", unsafe_allow_html=True)
        st.markdown("---")

    if model:
        if source_type == "Изображение":
            # Image selection
            image = None
            image_source = st.radio(
                "Выберите источник изображения:", ("Введите URL", "Загрузить с ПК")
            )
            if image_source == "Загрузить с ПК":
                uploaded_file = st.file_uploader(
                    "Выбрать изображение", type=["png", "jpg", "jpeg"]
                )
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                else:
                    image = None
            else:
                url = st.text_input("Введите URL изображения:")
                if url:
                    try:
                        response = requests.get(url, stream=True)
                        if response.status_code == 200:
                            image = Image.open(response.raw)
                        else:
                            st.error("Ошибка загрузки URL.")
                            image = None
                    except requests.exceptions.RequestException as e:
                        st.error(f"Ошибка загрузки URL: {e}")
                        image = None

            if image:
                with st.spinner("Выполняется обнаружение"):
                    res_image, text, stats = predict_image(
                        model, image, conf_threshold, iou_threshold
                    )
                    st.image(res_image, caption="Prediction", use_container_width=True)
                    st.success(text)
                    # update_stats_display(stats)

                # Конвертация numpy.ndarray в PIL Image
                pil_image = Image.fromarray(res_image)
                prediction_buffer = io.BytesIO()
                pil_image.save(prediction_buffer, format="PNG")

                st.download_button(
                    label="Сохранить результат детекции",
                    data=prediction_buffer.getvalue(),
                    file_name="prediction.png",
                    mime="image/png",
                )

        elif source_type in ["Видео", "Веб-камера"]:
            if source_type == "Видео":
                video_file_buffer = st.file_uploader(
                    "Загрузить видео", type=["mp4", "avi", "mov"]
                )

                if video_file_buffer is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(video_file_buffer.read())
                    video_file_path = tfile.name
                else:
                    video_file_path = None  # No video uploaded
            else:  # Webcam
                video_file_path = WEBCAM_SOURCE  # Use webcam source

            if video_file_path is not None:
                st_frame = st.empty()  # Placeholder for live video
                asyncio.run(
                    process_video_async(
                        model,
                        video_file_path,
                        conf_threshold,
                        iou_threshold,
                        st_frame,
                    )
                )

                # Clean up temp file if it exists and we're not using the webcam
                if source_type == "Видео":
                    tfile.close()
                    os.unlink(tfile.name)  # Удаление временного файла

    else:
        st.info("Пожалуйста, выберите модель в боковой панели.")


if __name__ == "__main__":
    main()
