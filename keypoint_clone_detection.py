import argparse
import collections

from matplotlib import cm
import numpy as np
import cv2
from skimage import draw
from skimage.color import color_dict

__version__ = "0.0.1"

FLANN_INDEX_KDTREE = 1


KeyPoints = collections.namedtuple(
    "KeyPoints", ["n_keypoints", "locations", "descriptors", "orientations", "scales"]
)

Matches = collections.namedtuple(
    "Matches", ["n_matches", "train_indexes", "query_indexes", "distances"]
)


def check_image(img):
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    return img


def equalize(img, radius=None):
    if radius is None:
        min_size = min(img.shape[0], img.shape[1])
        radius = int(np.sqrt(min_size))

    img_gr = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    clahe = cv2.createCLAHE(tileGridSize=(radius, radius))
    return clahe.apply(img_gr)


def extract_sift_descr(img, n_keypoints):
    """ Extract orb descriptors

    Parameters
    ----------
    img: array of uint8, shape=[height, width, 3]
        The input image
    n_keypoints: int, optional (default=None)
        The number of keypoints. If not given it will be take as the max of height and width.

    Returns
    -------
    kpts: list of cv2.Keypoint
        The list of detected keypoints

    descr: list of
        Object holding the keypoints and descriptors
    """
    sift = cv2.SIFT_create(nfeatures=n_keypoints)
    return sift.detectAndCompute(img, mask=None)


def drop_duplicate_matches(matches):
    indexes = set()
    dedup_matches = []
    for m in matches:
        i, j = m.queryIdx, m.trainIdx
        ij = (i, j) if i < j else (j, i)
        if ij not in indexes:
            indexes.add(ij)
            dedup_matches.append(m)
    return dedup_matches


def find_matches(descriptors, filter_matches=False):
    """Match descriptors to themselves

    Parameters
    ----------
    descriptors: array_like, shape=[n_keypoints, n_features]
        The array of descriptors

    Returns
    -------
    matches: list of cv2.Match
        The found matches
    """
    print(f"Number of descriptors: {len(descriptors)}")
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    if filter_matches:
        matches = flann.knnMatch(descriptors, descriptors, k=3)
        matches = [m1 for _, m1, m2 in matches if m1.distance < 0.7 * m2.distance]
    else:
        matches = flann.knnMatch(descriptors, descriptors, k=2)
        matches = [m for _, m in matches]
    print(len(matches))
    matches = drop_duplicate_matches(matches)
    print(len(matches))

    return matches


def convert_keypoints_descriptors(keypoints, descriptors):
    locations = np.array([kpt.pt[::-1] for kpt in keypoints])
    angles = np.array([kpt.angle for kpt in keypoints])
    sizes = np.array([kpt.size for kpt in keypoints])
    return KeyPoints(
        n_keypoints=len(keypoints),
        locations=locations,
        descriptors=descriptors,
        orientations=angles,
        scales=sizes,
    )


def convert_matches(matches):
    return Matches(
        n_matches=len(matches),
        train_indexes=np.array([m.trainIdx for m in matches]),
        query_indexes=np.array([m.queryIdx for m in matches]),
        distances=np.array([m.distance for m in matches])
    )


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


def get_colors(name, n):
    cmap = cm.get_cmap(name)
    if hasattr(cmap, "colors"): # Discrete
        c = cmap(np.arange(n) % len(cmap.colors))
    else:
        c = cmap(np.linspace(0, 1, n))
    return (c[:, :3] * 255).clip(0, 255).astype("uint8")


def draw_matches(img,
                 keypoints,
                 matches,
                 color="lime",
                 damp_factor=5.0,
                 threshold=1.0):
    """Add lines joining matching keypoints over the image.

    Parameters
    ----------
    img: numpy.array, shape=[height, width, 3]
        The input image.
    keypoints: KeyPoints
        The result of the keypoints extraction
    matches: Matches
        The result of the keypoint matching
    color: str
        The name of the color (from the `skimage.color.color_dict` dictionary).
    damp_factor: float
        The damp factor used in the gaussian weight profile.
    threshold: float, optional (default=1)
        The threshold on descriptor distance above which the matches are displayed expressed
        in quantile. By default all matches are displayed.

    Returns
    -------
    img_match: array of uint8, shape=[height, width, 3]
        The original image with lines joining matching keypoints as overlay.

    """
    try:
        color = (np.array(color_dict[color]) * 255).clip(0, 255).astype("uint8")
        colors = np.array([color] * matches.n_matches)
    except KeyError:
        try:
            colors = get_colors(color, matches.n_matches)
            ishuffle = np.random.permutation(len(colors))
            colors = colors[ishuffle]
        except ValueError:
            raise ValueError(f"{color!r} is not a valid color name or color map name.")

    weights = compute_weights(matches.distances, damp_factor=damp_factor)

    if threshold < 1.0:
        thresh_dist = np.quantile(matches.distances, threshold)
        mask = matches.distances < thresh_dist
        train_indexes = matches.train_indexes[mask]
        query_indexes = matches.query_indexes[mask]
        weights = weights[mask]
        colors = colors[mask]
    else:
        train_indexes = matches.train_indexes
        query_indexes = matches.query_indexes

    img_draw = img.copy()
    if img_draw.ndim == 2:
        img_draw = np.dstack([img_draw] * 3)

    locations = np.round(keypoints.locations).astype("int")

    for i, j, w, color in zip(train_indexes, query_indexes, weights, colors):
        (r0, c0), (r1, c1) = locations[[i, j]]
        li, lj, la = draw.line_aa(r0, c0, r1, c1)
        img_draw[li, lj] = (1 - w) * img_draw[li, lj] + w * la[:, None] * color

    return img_draw


def extract_keypoints_and_draw_matches(img,
                                       n_keypoints=None,
                                       color="lime",
                                       damp_factor=5.0,
                                       threshold=1.0):
    kpts, descr = extract_sift_descr(img, n_keypoints=n_keypoints)
    matches = find_matches(descr)

    keypoints = convert_keypoints_descriptors(kpts, descr)
    matches = convert_matches(matches)

    return draw_matches(img, keypoints, matches, color=color, damp_factor=damp_factor, threshold=threshold)


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

