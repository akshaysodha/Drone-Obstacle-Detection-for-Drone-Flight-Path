import streamlit as st
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
import openai

openai.api_key = "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"

st.set_page_config(page_title="Final Project Team6 DATA 298B", layout="centered")
st.title(" 🎮🚁 Drone Inflight Path Intrusions Detection")
st.markdown("Upload an image and let our model predict what's in it.")

MODEL_PATH = r"C:\Users\vabss\Downloads\bestmodel\Yolov11\best.pt"  # <-- update this path

uploaded_image = st.file_uploader("Upload an image for inference", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.subheader("Model and Image Preview")

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Image Preview
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
        tmp_img.write(uploaded_image.read())
        tmp_img_path = tmp_img.name

    image = Image.open(tmp_img_path).convert("RGB")
    results = model.predict(source=image, save=False, imgsz=640)

    st.subheader("📊 YOLO Prediction Results")
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Predicted Image", use_column_width=True)

    st.subheader("🤖 GPT Analysis")

    detections = []
    for box in results[0].boxes:
        cls_name = results[0].names[int(box.cls[0])]
        conf = float(box.conf[0])
        detections.append(f"{cls_name} ({conf*100:.2f}%)")

    detection_text = "No objects detected." if not detections else "Objects detected:\n" + ", ".join(detections)

    prompt = f"""
    The following objects have been detected:
    {detection_text}

    Please summarize these detections and comment on how the model performed.
    """

    with st.spinner("Asking GPT for a summary..."):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that interprets detection results."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )

    gpt_answer = response["choices"][0]["message"]["content"]
    st.markdown(gpt_answer)

    os.remove(tmp_img_path)

st.markdown("---")
st.markdown("Built by **Team 6**")
