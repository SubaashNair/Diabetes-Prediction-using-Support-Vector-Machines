import gradio as gr
import numpy as np
import joblib

# Function to load model and make prediction


def make_prediction(Pregnancies, Glucose, DiabetesPedigreeFunction, BMI):
    model = joblib.load('svc_model.pkl')
    scaler = joblib.load('scaler.pkl')

    input_data = np.asarray(
        [Pregnancies, Glucose, DiabetesPedigreeFunction, BMI]).reshape(1, -1)
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    # For binary classification, you might return something like this
    return 'Positive' if prediction[0] == 1 else 'Negative'


# Define the gradio interface
# iface = gr.Interface(
#     fn=make_prediction,
#     inputs={
#         'Pregnancies': gr.inputs.Number(),
#         'Glucose': gr.inputs.Number(),
#         'DiabetesPedigreeFunction': gr.inputs.Number(),
#         'BMI': gr.inputs.Number()
#     },
#     outputs='text'
# )
iface = gr.Interface(
    fn=make_prediction,
    inputs=[
        gr.inputs.Number(label="Pregnancies"),
        gr.inputs.Number(label="Glucose"),
        gr.inputs.Number(label="DiabetesPedigreeFunction"),
        gr.inputs.Number(label="BMI")
    ],
    outputs=gr.Textbox(label="Diabetes Prediction"),
    title="Diabetes Prediction App"
)


# Launch the interface
iface.launch()
