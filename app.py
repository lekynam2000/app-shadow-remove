import streamlit as st
from PIL import Image
from MaskShadowGAN.baseline import MaskShadowGAN_remover

model_list = {}

MSGAN_remover = MaskShadowGAN_remover()

# Set up the UI
st.title("Shadow Removal Application")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Display original image
    st.subheader("Original Image")
    st.image(img, use_column_width=True)

    # Choose filter and intensity
    # filter_name = st.selectbox("Choose a filter:", ('Blur', 'Contour', 'Brightness', 'Sharpness'))
    # if filter_name == 'Brightness' or filter_name == 'Sharpness':
    #     intensity = st.slider("Intensity", 0.0, 2.0, 1.0, 0.1)
    # else:
    #     intensity = None

    # Remove Shadow
    if st.button("Remove Shadow"):
        processed_img = MSGAN_remover.remove_shadow(img)
        st.subheader("Processed Image")
        st.image(processed_img, use_column_width=True)
