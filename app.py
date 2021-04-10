import os
import argparse
from datetime import datetime
import streamlit as st
from skimage.io import imread, imsave
import keypoint_clone_detection as kcd


@st.cache(allow_output_mutation=True)
def load_image(uploaded_image, image_dir=None):
    img = imread(uploaded_image)
    if image_dir is not None:
        if not os.path.exists(image_dir):
            raise RuntimeError("Image save directory does not exist")
        output_name = os.path.join(
            image_dir,
            f"{str(datetime.now()).replace(' ', '_').split('.')[0]}_{uploaded_image.name}"
        )
        imsave(output_name, img)
    return img


@st.cache(allow_output_mutation=True)
def compute_matches(img):
    img_matches = kcd.extract_keypoints_and_draw_matches(
        img
    )
    return img_matches

@st.cache()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-image-dir", default=None, help="Image to save uploaded images in.")
    return parser.parse_args()

args = parse_args()

st.markdown("# Clone detection in images")

uploaded_image = st.file_uploader("Choose an image file")

if uploaded_image is not None:
    img = load_image(uploaded_image, args.save_image_dir)
    img_matches = compute_matches(img)

    show_matches = st.checkbox(label="show matches", value=True)
    if show_matches:
        st.image(img_matches)
    else:
        st.image(img)


st.sidebar.markdown("# Key point matching")
st.sidebar.markdown(
    "The algorithm uses `SIFT` to detect key points and compute local descriptors. "
    "Key points are matched based on their descriptors similarity. "
    "The lines join matching key points. The line opacity represents the similarity "
    "of the key point descriptors."
)
