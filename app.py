import os
import streamlit as st
from PIL import Image
from MaskShadowGAN.baseline import MaskShadowGAN_remover
from G2R_ShadowRemoval.baseline import G2R_remover

model_list = {
    "MSGAN_remover": MaskShadowGAN_remover(),
    "MSGAN_remover addLoss": MaskShadowGAN_remover(pretrained_path=os.path.join("MaskShadowGAN","addLoss","net_addLoss.pth")),
    "MSGAN_remover change strategy": MaskShadowGAN_remover(pretrained_path=os.path.join("MaskShadowGAN","change_strategy","net_change_strategy.pth")),
    "G2R_remover": G2R_remover(device="cuda:3"),

}


# Set up the UI
st.title("Shadow Removal Application")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Display original image
    st.subheader("Original Image")
    st.image(img, use_column_width=True)

    algorithm = st.selectbox("Choose Removal Algorithm:", model_list.keys())
    

    # Remove Shadow
    if st.button("Remove Shadow"):
        output = model_list[algorithm].remove_shadow(img)
        st.subheader(f"{algorithm} output:")
        st.image(output["img"], use_column_width=True)
        if "mask" in output.keys():
            st.image(output["mask"], use_column_width=True)
