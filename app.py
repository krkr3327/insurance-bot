import gradio as gr
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("insurance_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

def predict_charges(age, sex, bmi, smoker, region, children):
    try:
        data = pd.DataFrame([{
            "age": float(age),
            "sex": label_encoders["sex"].transform([sex])[0],
            "bmi": float(bmi),
            "smoker": label_encoders["smoker"].transform([smoker])[0],
            "region": label_encoders["region"].transform([region])[0],
            "children": int(children)
        }])
        prediction = model.predict(data)[0]
        return f"💰 Estimated Insurance Charges: ₹{prediction:.2f}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio interface
demo = gr.Interface(
    fn=predict_charges,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="BMI"),
        gr.Radio(["yes", "no"], label="Smoker"),
        gr.Dropdown(["northeast", "northwest", "southeast", "southwest"], label="Region"),
        gr.Number(label="Number of Children")
    ],
    outputs="text",
    title="🏥 Insurance Charges Predictor Bot",
    description="Enter the details to estimate medical insurance charges."
)

demo.launch()
