import argparse
import collections

from scipy.spatial import cKDTree
import numpy as np
import cv2
from skimage import draw
from skimage.color import color_dict

__version__ = "0.0.1"


Descriptor = collections.namedtuple(
    "Descriptor", ["keypoints", "descriptors", "orientations", "scales"]
)


def equalize(img, radius=None):
    if radius is None:
        min_size = min(img.shape[0], img.shape[1])
        radius = int(np.sqrt(min_size))

    img_gr = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    clahe = cv2.createCLAHE(tileGridSize=(radius, radius))
    return clahe.apply(img_gr)


def extract_sift_descr(img, n_keypoints=None):
    """ Extract orb descriptors

    Parameters
    ----------
    img: array of uint8, shape=[height, width, 3]
        The input image
    n_keypoints: int, optional (default=None)
        The number of keypoints. If not given it will be take as the max of height and width.

    Returns
    -------
    descr: Descriptor
        Object holding the keypoints and descriptors
    """
    max_size = max(*img.shape[:2])

    if n_keypoints is None:
        n_keypoints = max_size

    img_eq = equalize(img)

    sift = cv2.SIFT_create(nfeatures=n_keypoints)
    kpts = sift.detect(img_eq)
    kpts, descriptors = sift.compute(img, kpts)

    keypoints = np.array([kp.pt[::-1] for kp in kpts])

    descr = Descriptor(
        keypoints=keypoints,
        descriptors=descriptors,
        orientations=np.array([kp.angle for kp in kpts]),
        scales=np.array([kp.size for kp in kpts])
    )

    return descr


def find_matches(descriptors):
    """Match descriptors to themselves

    Parameters
    ----------
    descriptors: array_like, shape=[n_keypoints, n_features]
        The array of descriptors

    Returns
    -------
    ineighb: array of int, shape=[n_keypoints]
        The indices of the neighbor of each input point
    neighb_dist: array of flaot, shape=[n_keypoints]
        The distance to the nearest neighbor
    """
    kdtree = cKDTree(descriptors)
    dists, indices = kdtree.query(descriptors, k=2)
    return indices.T[1], dists.T[1]


def compute_weights(distances, damp_factor=5.0):
    """Compute weights based on distances

    Parameters
    ----------
    distances: array like
        The input distances
    damp_factor: float
        The damp factor used in the gaussian weight profile.

    Returns
    -------
    weights: array like
        The weights associated to each distance.
    """
    dists = distances - distances.min()
    return np.exp(- damp_factor * (dists / np.median(dists)) ** 2)


def draw_matches(img, descr, color="lime", damp_factor=5.0):
    """Add lines joining matching keypoints over the image.

    Parameters
    ----------
    img: array like, shape=[height, width, 3]
        The input image.
    descr: Descriptor
        The result of the keypoints extraction
    color: str
        The name of the color (from the `skimage.color.color_dict` dictionary).
    damp_factor: float
        The damp factor used in the gaussian weight profile.

    Returns
    -------
    img_match: array of uint8, shape=[height, width, 3]
        The original image with lines joining matching keypoints as overlay.

    """
    try:
        color = (np.array(color_dict[color]) * 255).clip(0, 255).astype("uint8")
    except KeyError:
        raise ValueError(f"{color!r} is not a valid color name.")

    ineighb, neighb_dist = find_matches(descr.descriptors)
    keypoints = descr.keypoints.astype("int")
    weights = compute_weights(neighb_dist, damp_factor=damp_factor)

    img_draw = img.copy()

    for i, (j, w) in enumerate(zip(ineighb, weights)):
        (r0, c0), (r1, c1) = keypoints[[i, j]]
        li, lj, la = draw.line_aa(r0, c0, r1, c1)
        img_draw[li, lj] = (1 - w) * img_draw[li, lj] + w * la[:, None] * color

    return img_draw


def extract_keypoints_and_draw_matches(img,
                                       n_keypoints=None,
                                       color="lime",
                                       damp_factor=5.0):
    descr = extract_sift_descr(img, n_keypoints=n_keypoints)
    return draw_matches(img, descr, color=color, damp_factor=damp_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", help="The image file.")
    parser.add_argument("output_image", help="The filename of the output image.")
    parser.add_argument("-c", "--color", default="lime", help="The color to use.")
    parser.add_argument("-n", "--n-keypoints", type=int, default=None, help="The number of keypoints.")
    parser.add_argument("-a", "--damp-factor", type=float, default=5.0,
                        help="Damping factor for match weights based on distances")
    args = parser.parse_args()

    img = cv2.cvtColor(
        cv2.imread(args.input_image),
        cv2.COLOR_BGR2RGB
    )

    img_match = extract_keypoints_and_draw_matches(
        img, n_keypoints=args.n_keypoints,
        color=args.color, damp_factor=args.damp_factor,
    )

    cv2.imwrite(
        args.output_image,
        cv2.cvtColor(img_match, cv2.COLOR_RGB2BGR)
    )

