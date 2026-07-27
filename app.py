import joblib
import gradio as gr
import numpy as np

# ==========================================================
# Load Model
# ==========================================================
try:
    model = joblib.load("breast_cancer_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ==========================================================
# Prediction Function
# ==========================================================
def predict(
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    concave_points_mean,
    symmetry_mean,
    fractal_dimension_mean,
    radius_se,
    texture_se,
    perimeter_se,
    area_se,
    smoothness_se,
    compactness_se,
    concavity_se,
    concave_points_se,
    symmetry_se,
    fractal_dimension_se,
    radius_worst,
    texture_worst,
    perimeter_worst,
    area_worst,
    smoothness_worst,
    compactness_worst,
    concavity_worst,
    concave_points_worst,
    symmetry_worst,
    fractal_dimension_worst,
):

    if model is None:
        return "❌ Model not loaded."

    try:
        features = np.array([[
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean,
            symmetry_mean,
            fractal_dimension_mean,
            radius_se,
            texture_se,
            perimeter_se,
            area_se,
            smoothness_se,
            compactness_se,
            concavity_se,
            concave_points_se,
            symmetry_se,
            fractal_dimension_se,
            radius_worst,
            texture_worst,
            perimeter_worst,
            area_worst,
            smoothness_worst,
            compactness_worst,
            concavity_worst,
            concave_points_worst,
            symmetry_worst,
            fractal_dimension_worst
        ]])

        prediction = model.predict(features)[0]

        if prediction == 1:
            return "🟢 Benign (Non-Cancerous)"
        else:
            return "🔴 Malignant (Cancerous)"

    except Exception as e:
        return f"Prediction Error: {e}"


# ==========================================================
# User Interface
# ==========================================================

inputs = [
    gr.Number(label="Radius Mean"),
    gr.Number(label="Texture Mean"),
    gr.Number(label="Perimeter Mean"),
    gr.Number(label="Area Mean"),
    gr.Number(label="Smoothness Mean"),
    gr.Number(label="Compactness Mean"),
    gr.Number(label="Concavity Mean"),
    gr.Number(label="Concave Points Mean"),
    gr.Number(label="Symmetry Mean"),
    gr.Number(label="Fractal Dimension Mean"),
    gr.Number(label="Radius SE"),
    gr.Number(label="Texture SE"),
    gr.Number(label="Perimeter SE"),
    gr.Number(label="Area SE"),
    gr.Number(label="Smoothness SE"),
    gr.Number(label="Compactness SE"),
    gr.Number(label="Concavity SE"),
    gr.Number(label="Concave Points SE"),
    gr.Number(label="Symmetry SE"),
    gr.Number(label="Fractal Dimension SE"),
    gr.Number(label="Radius Worst"),
    gr.Number(label="Texture Worst"),
    gr.Number(label="Perimeter Worst"),
    gr.Number(label="Area Worst"),
    gr.Number(label="Smoothness Worst"),
    gr.Number(label="Compactness Worst"),
    gr.Number(label="Concavity Worst"),
    gr.Number(label="Concave Points Worst"),
    gr.Number(label="Symmetry Worst"),
    gr.Number(label="Fractal Dimension Worst"),
]

custom_css = """
footer {
    visibility: hidden;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto;
}

h1 {
    text-align: center;
}

.developer {
    margin-top: 20px;
    padding: 15px;
    border-radius: 10px;
    background-color: #f5f5f5;
}
"""

description = """
### 🩺 Breast Cancer Prediction System

This application predicts whether a breast tumor is **Benign** or **Malignant**
using a trained Machine Learning model.

---

### 👨‍💻 Developer Information

**Developer:** **Shubham Sharma**

🎓 Bachelor of Computer Applications (BCA)

🛡️ Machine Learning & Cyber Security Enthusiast

📧 Email: **svats1310@gmail.com**

---

⚠️ **Disclaimer**

This application is developed for educational and demonstration purposes only.
It should not be used as a substitute for professional medical advice or diagnosis.
"""

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction Result"),
    title="🩺 Breast Cancer Prediction System",
    description=description,
    theme=gr.themes.Soft(),
    css=custom_css,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
