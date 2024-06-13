import os
from google.cloud import run_v2
import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/my-personal-mlops-project/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri
    name = os.environ.get("BACKEND_SERVICE_NAME", None)

    if name is None:
        raise ValueError("Backend service not found")

    return name


def image_to_byte_array(image: Image) -> bytes:
    """Convert an image to a byte array."""
    # BytesIO is a file-like buffer stored in memory
    img_bytes = io.BytesIO()
    # image.save expects a file-like as a argument
    image.save(img_bytes, format="PNG")
    # Turn the BytesIO object back into a bytes object
    img_bytes = img_bytes.getvalue()
    return img_bytes


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict"
    response = requests.post(predict_url, files={"image": image})
    if response.status_code == 200:
        return response.json()
    else:
        return None


def main():
    """Main function for the Streamlit frontend."""
    backend = get_backend_url()
    if "initialized" not in st.session_state or not st.session_state.initialized:
        # waky waky up the backend
        requests.get(f"{backend}/health")
        st.session_state.initialized = True

    st.title("Image Classification")

    on = st.toggle("Draw image")

    if on:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=5,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=100,
            width=100,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result.image_data is not None:
            classify = st.button("Classify Image?")

            if classify:
                image = Image.fromarray(canvas_result.image_data)
                image = image.convert("RGB")
                image = image.resize((28, 28))
                result = classify_image(image_to_byte_array(image), backend)

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

    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = uploaded_file.read()
            result = classify_image(image, backend)

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
