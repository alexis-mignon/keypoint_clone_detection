import os
import argparse
from datetime import datetime
import streamlit as st
from skimage.io import imread, imsave
import keypoint_clone_detection as kcd


@st.cache(allow_output_mutation=True)
def load_image(uploaded_image, image_dir=None):
    img = imread(uploaded_image)
    img = kcd.check_image(img)
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
def equalize(img, radius, denoise):
    return kcd.equalize(img, radius, denoise)


@st.cache(allow_output_mutation=True)
def compute_matches(img_eq, n_keypoints):
    kpts, descr = kcd.extract_sift_descr(img_eq, n_keypoints=n_keypoints)

    matches = kcd.find_matches(descr)
    return (
        kcd.convert_keypoints_descriptors(kpts, descr),
        kcd.convert_matches(matches)
    )


@st.cache()
def draw_matches(img, keypoints, matches, threshold, damp_factor=5.0, color="jet"):
    return kcd.draw_matches(img, keypoints, matches,
                            color=color, damp_factor=damp_factor, threshold=threshold)


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

    radius = st.sidebar.number_input(label="Radius of the local contrast area (in pixels)",
                                     value=50)
    denoise = st.sidebar.checkbox(label="Denoise", value=True,
                                  help=("Apply a denoising filter before detecting key points."
                                        "Can degrade the results if the duplicated signal is in "
                                        "the high frequencies"))

    img_eq = equalize(img, radius, denoise)

    use_equal = st.sidebar.checkbox(label="Use equalized image ", value=True,
                            help="Use the image with enhanced local contrast to detect key points")
    n_keypoints = st.sidebar.number_input("Number of keypoints:", value=5000)

    keypoints, matches = compute_matches(img_eq if use_equal else img, n_keypoints=n_keypoints)

    st.markdown(f"Number of detected keypoints: {keypoints.n_keypoints}")
    st.markdown(f"Number of found matches: {matches.n_matches}")

    show_matches = st.sidebar.checkbox(label="show matches", value=True,
                               help="Draw a line between matching keypoints")


    show_equal = st.sidebar.checkbox(label="show equalized image ", value=False,
                             help="Show the image with enhanced local contrast")

    threshold_max = st.sidebar.slider(label="Descriptor max distance threshold (in quantile): ",
                              min_value=0.0, max_value=1.0, value=1.0, step=0.01)

    threshold_min = st.sidebar.slider(label="Descriptor min distance threshold",
                              min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    damp = st.sidebar.number_input("Opacity damping factor:", value=1.0, step=0.1,
                           help=(
                               "Set the opacity as a decreasing function of the "
                               "descriptor distance. A damp factor 0.0 means that "
                               "all lines have a full opacity. Higher values make "
                               "the opacity drop faster."
                           ))

    img_ = img_eq if show_equal else img
    img_matches = draw_matches(
        img_,
        keypoints,
        matches,
        threshold=(threshold_min, threshold_max),
        damp_factor=damp
    )

    st.image(img_matches if show_matches else img_)


st.sidebar.markdown("# Key point matching")
st.sidebar.markdown(
    "The algorithm uses `SIFT` to detect key points and compute local descriptors. "
    "Key points are matched based on their descriptors similarity. "
    "The lines join matching key points. The line opacity represents the similarity "
    "of the key point descriptors."
)
