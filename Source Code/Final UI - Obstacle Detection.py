# ──────────────────────────────────────────────────────────────────────────────
#  Drone In‑flight Path Intrusion Detection – Team 6  ·  DATA 298B
#  Streamlit app with webcam support (no experimental_rerun)
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
import tempfile, os, json, cv2, time
from PIL import Image
from ultralytics import YOLO
import openai

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG & AUTHENTICATION
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Final Project – Team 6", layout="centered")

openai.api_key  = "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"
OPENAI_MODEL    = "gpt-4o-mini"
MODEL_PATH      = r"C:\Users\vabss\Downloads\bestmodel\Yolov11\best.pt"
ALLOWED_TYPES   = ["jpg", "jpeg", "png", "mp4", "avi", "mov"]

st.title("🎮🚁 Drone In‑flight Path Intrusions Detection")

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR – choose input source
# ──────────────────────────────────────────────────────────────────────────────
src_mode = st.sidebar.radio(
    "Choose input source",
    options=["Upload files", "Webcam (real‑time)"],
    index=0
)

# ──────────────────────────────────────────────────────────────────────────────
# COMMON: load YOLO model once
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Loading YOLO…")
def load_model(path: str):
    return YOLO(path)

model = load_model(MODEL_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# 1) UPLOAD mode
# ──────────────────────────────────────────────────────────────────────────────
if src_mode == "Upload files":
    st.markdown("Upload **one or many** images / videos. YOLO will detect obstacles")
    files = st.file_uploader("Choose files for inference",
                             type=ALLOWED_TYPES,
                             accept_multiple_files=True)

    if files:
        summaries, temp_paths = [], []

        for f in files:
            ext = os.path.splitext(f.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(f.read())
                temp_paths.append(tmp.name)

            # ─── IMAGE ────────────────────────────────────────────────────────
            if ext in [".jpg", ".jpeg", ".png"]:
                st.subheader(f"🖼️ {f.name}")
                img = Image.open(tmp.name).convert("RGB")
                st.image(img, caption="Original", use_column_width=True)

                res = model.predict(img, save=False, imgsz=640)
                st.image(res[0].plot(), caption="YOLO Prediction",
                         use_column_width=True)

                classes = [model.names[int(c)] for c in res[0].boxes.cls]
                confs   = [float(s)           for s in res[0].boxes.conf]
                summaries.append({"file": f.name, "type": "image",
                                  "classes": classes, "confidences": confs})

            # ─── VIDEO ────────────────────────────────────────────────────────
            else:
                st.subheader(f"📹 {f.name}")
                stframe = st.empty()
                cap     = cv2.VideoCapture(tmp.name)
                frame_cnt, shown = 0, 0
                v_classes, v_confs = [], []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_cnt += 1
                    if frame_cnt % 3:                      # speed‑up
                        continue

                    res = model.predict(frame, save=False, imgsz=640)
                    annotated = res[0].plot()

                    if shown < 60:                         # preview first N
                        stframe.image(annotated, channels="BGR",
                                       use_column_width=True)
                        shown += 1

                    v_classes.extend(model.names[int(c)] for c in res[0].boxes.cls)
                    v_confs.extend(float(s)           for s in res[0].boxes.conf)

                cap.release()
                summaries.append({"file": f.name, "type": "video",
                                  "classes": v_classes, "confidences": v_confs})

        # ─── ASK GPT FOR FEEDBACK ────────────────────────────────────────────
        with st.spinner("🧠 Asking GPT for feedback…"):
            prompt = (
                "Below is a JSON list of YOLO detections. "
                "For each file, tell me which classes appear confidently detected "
                "and which might need improvement. Group low‑confidence or rare "
                "detections together if you like.\n\n"
                f"{json.dumps(summaries, indent=2)}"
            )

            chat = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a concise computer‑vision QA assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            feedback = chat.choices[0].message.content.strip()

        st.markdown("### 🤖 GPT feedback")
        st.write(feedback)

        # cleanup temp files
        for p in temp_paths:
            os.remove(p)

# ──────────────────────────────────────────────────────────────────────────────
# 2) WEBCAM mode  – real‑time without experimental_rerun
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("Turn on your webcam and watch YOLO spot intrusions live.")

    run_cam = st.toggle("▶️ Start / stop webcam", value=False)

    # keep camera & placeholders in session_state
    if "cap" not in st.session_state:
        st.session_state.cap = None
    frame_out  = st.empty()      # for the annotated image
    stop_area  = st.empty()      # for the stop button

    # open / close camera based on toggle
    if run_cam and st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
    if not run_cam and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        frame_out.empty()
        stop_area.empty()

    # streaming loop
    while run_cam and st.session_state.cap is not None and st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if not ret:
            break

        # YOLO inference & display
        res = model.predict(frame, imgsz=640)
        frame_out.image(res[0].plot(), channels="BGR", use_column_width=True)

        # stop button with UNIQUE key every frame
        if stop_area.button("⏹️ Stop webcam", key=f"stop_{int(time.time()*1e3)}"):
            run_cam = False
            break

        time.sleep(0.03)   # ~30 FPS cap

    # final cleanup
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        frame_out.empty()
        stop_area.empty()

# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built by **Team 6**")
