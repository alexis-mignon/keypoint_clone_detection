import argparse
from scipy.spatial.distance import cdist
import numpy as np

from skimage.filters import rank
from skimage.morphology import disk
from skimage.color import color_dict, rgb2gray, gray2rgb
from skimage.feature import ORB
from skimage import draw
from skimage import io
from skimage import util


def self_match(descriptors, metric="hamming"):
    """Match descriptors to themselves

    Parameters
    ----------
    descriptors: array_like, shape=[n_keypoints, n_features]
        The array of descriptors

    metric: str, optional (default="hamming")
        Metric named to be passed to scipy.spatial.distance.cdist

    Returns
    -------
    ineighb: array of int, shape=[n_keypoints]
        The indices of the neighbor of each input point
    neighb_dist: array of flaot, shape=[n_keypoints]
        The distance to the nearest neighbor
    """
    dists = cdist(descriptors, descriptors, metric=metric)
    isort = dists.argsort(axis=1)
    ineighb = isort[:, 1]
    return ineighb, dists[np.arange(descriptors.shape[0]), ineighb]


def find_matches(img, n_keypoints=None):
    """Find keypoint matches

    Parameters
    ----------
    img: array of uint8, shape=[height, width, 3]
        The input image
    n_keypoints: int, optional (default=None)
        The number of keypoints. If not given it will be take as the max of height and width.

    Returns
    -------
    descr: skimage.feature.ORB
        The ORB feature extractor object
    ineighb: array of int, shape=[n_keypoints]
        The indices of the neighbor of each input point
    neighb_dist: array of flaot, shape=[n_keypoints]
        The distance to the nearest neighbor
    """
    min_size = min(*img.shape[:2])
    max_size = max(*img.shape[:2])
    radius = int(np.sqrt(min_size))

    img_gr = rgb2gray(img)
    img_eq = rank.equalize(util.img_as_ubyte(img_gr), disk(radius))

    if n_keypoints is None:
        n_keypoints = max_size

    descr = ORB(n_keypoints=n_keypoints)
    descr.detect_and_extract(img_eq)

    ineighb, neighb_dist = self_match(descr.descriptors)

    return descr, ineighb, neighb_dist


def draw_matches(img, color="lime", n_keypoints=None):
    """Draw matches

    Draw lines between matching keypoints on the image.

    Parameters
    ----------
    img: array of uint8, shape=[height, width, 3]
        The input image
    color: str, optional (default="lime")
        The name of the color (from the `skimage.color.color_dict` dictionary).
    n_keypoints: int, optional (default=None)
        The number of keypoints. If not given it will be take as the max of height and width.

    Returns
    -------
    img_match: array of uint8, shape=[height, width, 3]
        The original image with lines joining matching keypoints as overlay.
    """
    try:
        color = (np.array(color_dict[color]) * 255).clip(0, 255).astype("uint8")
    except KeyError:
        raise ValueError(f"{color!r} is not a valid color name.")

    descr, ineighb, neighb_dist = find_matches(img, n_keypoints=n_keypoints)

    weights = np.exp(-5.0 * (neighb_dist - neighb_dist.min()) / np.median(neighb_dist))
    keypoints = np.round(descr.keypoints).astype("int")

    img_draw = img.copy()

    for i, (j, w) in enumerate(zip(ineighb, weights)):
        (r0, c0), (r1, c1) = keypoints[[i, j]]
        li, lj, la = draw.line_aa(r0, c0, r1, c1)
        img_draw[li, lj] = (1 - w) * img_draw[li, lj] + w * la[:, None] * color

    return img_draw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", help="The image file.")
    parser.add_argument("output_image", help="The filename of the output image.")
    parser.add_argument("-c", "--color", default="lime", help="The color to use.")
    parser.add_argument("-n", "--n-keypoints", type=int, default=None, help="The number of keypoints.")

    args = parser.parse_args()

    img = io.imread(args.input_image)
    if img.ndim == 2:
        img = gray2rgb(img)
    img_match = draw_matches(img, color=args.color, n_keypoints=args.n_keypoints)

    io.imsave(args.output_image, img_match)