import streamlit as st
from skimage.io import imread
from skimage.util import img_as_float
import io
import keypoint_clone_detection as kcd

st.markdown("# Clone detection in images")


@st.cache(allow_output_mutation=True)
def load_image(uploaded_image):
    bytes_data = uploaded_image.getvalue()
    return imread(io.BytesIO(bytes_data))


@st.cache(allow_output_mutation=True)
def compute_matches(img):
    img_matches = kcd.extract_keypoints_and_draw_matches(
        img
    )
    return img_matches


uploaded_image = st.file_uploader("Choose an image file")
if uploaded_image is not None:
    img = load_image(uploaded_image)
    img_matches = compute_matches(img)

    show_matches = st.checkbox(label="show matches", value=True)
    if show_matches:
        st.image(img_matches)
    else:
        st.image(img)