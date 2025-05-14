# 🛸 Drone Obstacle Detection for Flight Path Optimization

This project presents a real-time obstacle detection system designed to enhance drone navigation safety using deep learning. We implemented and benchmarked multiple state-of-the-art models to detect aerial obstacles under diverse environmental conditions. The project showcases skills in model training, annotation automation, cloud deployment, and real-time inferencing.

---

## 🎯 Project Objectives

- Develop a deep learning pipeline capable of detecting **11 types of aerial obstacles**.
- Annotate **34,000+ drone images** using **Grounding DINO + SAM**.
- Train and compare object detection models: **YOLOv11**, **Faster R-CNN**, **RT-DETR v3**, and **Detectron2**.
- Deploy a responsive web interface using **Streamlit** and **REST APIs** on **Google Cloud**.
- Evaluate models using **mAP**, **F1-score**, and **latency** benchmarks to identify the best-fit solution for real-time use.

---

## 🧰 Tools & Technologies Used

- **Frameworks**: PyTorch, TensorFlow, Detectron2, Grounding DINO  
- **Annotation**: Grounded SAM, LabelImg, Roboflow  
- **Cloud & DevOps**: Google Cloud (GCS, Compute Engine, BigQuery), Streamlit, Flask  
- **Visualization & Evaluation**: TensorBoard, Matplotlib, COCO Evaluator  
- **Programming**: Python 3.10

---

## 📦 Dataset: DDOS (Drone Depth and Obstacle Segmentation)

- **Source**: Hugging Face  
- **Size**: ~256 GB across 300+ drone flight missions  
- **Structure**: RGB, depth maps, segmentation masks, normals, metadata  
- **Classes**: 11 object types including cars, trees, poles, wires, mesh, bungalows  
- **Augmentation**: Brightness, rotation, noise — boosted total dataset size by 50%  
- **Split**: 70% train, 20% val, 10% test

---

## 🧪 Model Evaluation Summary

| Model            | mAP@0.5 | mAP@[0.5:0.95] | F1 Score | FPS | Notable Strengths              |
|------------------|---------|----------------|----------|-----|--------------------------------|
| **YOLOv11**       | **0.859** | **0.784**       | **0.81**  | 30  | Speed + accuracy, small objs   |
| Faster R-CNN     | 0.720   | 0.660          | 0.75     | 15  | Occlusion, clutter             |
| RetinaNet        | 0.760   | 0.670          | 0.76     | 20  | Balanced urban detection       |
| RT-DETR v3       | 0.695   | 0.610          | 0.705    | 25  | Transformer + mid-sized objs   |
| Grounding DINO   | ~0.85   | ~0.77          | ~0.79    | 10  | Semantic reasoning             |

---

## 🔁 Project Pipeline
📥 Dataset Ingestion →

🧠 Annotation (Grounded SAM) →

🧼 Preprocessing (resize, augment, normalize) →

🧪 Model Training + Tuning →

📊 Evaluation (mAP, F1, latency) →

🚀 Deployment (GCP + Streamlit)

---

## 🖼️ Interface & Deployment

- Upload images or videos for real-time detection.
- View bounding boxes, class labels, and confidence levels.
- Achieves **30+ FPS** with minimal latency (**<220 ms**).
- RESTful API enables integration with drone control systems.

![Streamlit UI Screenshot](media/streamlit_ui.png)

