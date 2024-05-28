from google.cloud import run_v2
import streamlit as st
import requests
import pandas as pd


def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/my-personal-mlops-project/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri


BACKEND = get_backend_url()
if BACKEND is None:
    raise ValueError("Backend service not found")


def classify_image(image):
    """Send the image to the backend for classification."""
    predict_url = f"{BACKEND}/predict"
    response = requests.post(predict_url, files={"image": image})
    if response.status_code == 200:
        return response.json()
    else:
        return None


def main():
    """Main function for the Streamlit frontend."""
    st.title("Image Classification")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image)

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", prediction)

            # make a nice bar chart
            data = {"Class": [f"Class {i}" for i in range(10)], "Probability": probabilities}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
