import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Plantttttt.keras")

model = load_model()

def preprocess_image(image):
    image = image.resize((224, 224))  
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  
    image_array = image_array / 255.0  
    return image_array

def model_prediction(image_array):
    predictions = model.predict(image_array)
    return predictions

medicinal_uses = {
    'Aloevera': 'Aloevera has various medicinal uses including treating burns, soothing sunburns, moisturizing skin, and aiding digestion.',
    'Amla': 'Amla, also known as Indian gooseberry, is rich in vitamin C and antioxidants. It is used in Ayurvedic medicine to boost immunity, improve digestion, and promote hair health.',
    'Neem': 'Neem is a versatile medicinal plant with antibacterial, antifungal, and anti-inflammatory properties. It is used in treating skin conditions, dental issues, and boosting immunity.',
    'Tulsi': 'Tulsi, or holy basil, is revered in Ayurveda for its medicinal properties. It is used to treat respiratory disorders, fever, inflammation, and promote overall health.',
}

def home_page():
    st.header("MEDICINAL PLANT DETECTION SYSTEM")
    image_path = "homepage.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Medicinal Plant Detection System! 🌿🔍

    Our mission is to empower you to identify medicinal plants and discover their uses. Upload an image of a plant, and our system will analyze it to reveal its potential medicinal properties. Together, let's unlock the power of nature's pharmacy!

    ### How It Works

    **Upload Image**: Go to the Detect Plant page and upload an image.
    **Analysis**: Our system will process the image using advanced algorithms to identify the medicinal plant and provide information on its traditional uses.
    **Results**: View the results and explore recommendations for further research or responsible use.
   
    ### Why Choose Us?

    **Accuracy**: Our system utilizes state-of-the-art machine learning techniques for reliable medicinal plant identification.
    **User-Friendly**: Simple and intuitive interface for seamless user experience.
    **Fast and Efficient**: Receive results in seconds, allowing you to explore the world of medicinal plants quickly.

    Click on the Medicinal Plant Detection page in the sidebar to upload an image and experience the power of our system!

    Learn more about the project, our team, and our goals on the About page.
    """)

def about_page():
    st.header("About")
    st.markdown("""
        ### About the Project

        This project is dedicated to identifying medicinal plants and providing information on their uses through the power of machine learning and image processing. Our goal is to bridge the gap between traditional knowledge and modern technology, making it easier for people to explore and benefit from the natural properties of various plants.

        We believe that understanding and utilizing medicinal plants can contribute to better health and well-being, and we are committed to making this knowledge accessible to everyone.

        Thank you for using our system, and we hope you find it valuable in your journey of discovering the wonders of medicinal plants.
    """)

def detect_plant():
    st.header("Medicinal Plant Detection")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        st.image(test_image, use_column_width=True, caption="Selected Image")

        if st.button("Predict", key="predict_button"):
            image = Image.open(test_image)
            image_array = preprocess_image(image)

            predictions = model_prediction(image_array)

            max_index = np.argmax(predictions)

            classes = ['Aloevera and Coconut', 'Neem and Sweet Potato', 'Tulsi and Watermelon', 'Amla and Pineapple']

            plant_name = classes[max_index]

            medicinal_plant_name = plant_name.split(" and ")[0]

            st.write(f"Detected Medicinal Plant: {medicinal_plant_name}")

            if medicinal_plant_name in medicinal_uses:
                medicinal_plant_use = medicinal_uses[medicinal_plant_name]
                st.write("Medicinal Uses:", medicinal_plant_use)
            else:
                st.write("No medicinal uses found for the detected plant.")

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Detect Plant"])

if app_mode == "Home":
    home_page()
elif app_mode == "About":
    about_page()
elif app_mode == "Detect Plant":
    detect_plant()
