import json
from os import path, walk, makedirs
from sys import exit, stderr

# This removes the massive amount of scikit warnings of "low contrast images"
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_dependencies():
    global fillPoly, imwrite, np, wkt, mapping, Polygon, imread, tqdm
    try:
        from cv2 import fillPoly as _fillPoly, imwrite as _imwrite
        import numpy as _np
        from shapely import wkt as _wkt
        from shapely.geometry import mapping as _mapping, Polygon as _Polygon
        from imageio import imread as _imread
        from tqdm import tqdm as _tqdm
    except ImportError as exc:
        raise SystemExit(
            'Missing xBD data-prep dependency. Install with: pip install -e .'
        ) from exc

    fillPoly = _fillPoly
    imwrite = _imwrite
    np = _np
    wkt = _wkt
    mapping = _mapping
    Polygon = _Polygon
    imread = _imread
    tqdm = _tqdm


def get_dimensions(file_path):
    """
    :param file_path: The path of the file
    :return: returns (width,height,channels)
    """
    # Open the image we are going to mask
    pil_img = imread(file_path)
    img = np.array(pil_img)
    w, h, c = img.shape
    return (w, h, c)


def mask_polygons_separately(size, shapes):
    """
    :param size: A tuple of the (width,height,channels)
    :param shapes: A list of points in the polygon from get_feature_info
    :returns: a dict of masked polygons with the shapes filled in from cv2.fillPoly
    """
    # For each WKT polygon, read the WKT format and fill the polygon as an image
    masked_polys = {}

    for u in shapes:
        sh = shapes[u]
        mask_img = np.zeros(size, np.uint8)
        i = fillPoly(mask_img, [sh], (255, 255, 255))
        masked_polys[u] = i

    return masked_polys


def mask_polygons_together(size, shapes):
    """
    :param size: A tuple of the (width,height,channels)
    :param shapes: A list of points in the polygon from get_feature_info
    :returns: A numpy array with the polygons filled 255s where there's a building and 0 where not
    """
    # For each WKT polygon, read the WKT format and fill the polygon as an image
    mask_img = np.zeros(size, np.uint8)

    for u in shapes:
        blank = np.zeros(size, np.uint8)
        poly = shapes[u][0]
        fillPoly(blank, [poly], (shapes[u][1], shapes[u][1], shapes[u][1]))
        mask_img += blank

    # Here we are taking the overlap (+=) and squashing it back to 0
    # mask_img[mask_img > 1] = 0

    # Finally we are taking all 1s and making it pure white (255)
    # mask_img[mask_img == 1] = 255
    # mask_img[mask_img == 2] = 128
    mask_img[mask_img > 4] = 255
    MINOR_DAMAGE_NUM = np.sum(mask_img[mask_img == 2])
    MAJOR_DAMAGE_NUM = np.sum(mask_img[mask_img == 3])
    DESTROY_DAMAGE_NUM = np.sum(mask_img[mask_img == 4])
    return mask_img, MINOR_DAMAGE_NUM, MAJOR_DAMAGE_NUM, DESTROY_DAMAGE_NUM


def mask_polygons_together_with_border(size, shapes, border):
    """
    :param size: A tuple of the (width,height,channels)
    :param shapes: A list of points in the polygon from get_feature_info
    :returns: a dict of masked polygons with the shapes filled in from cv2.fillPoly
    """

    # For each WKT polygon, read the WKT format and fill the polygon as an image
    mask_img = np.zeros(size, np.uint8)

    for u in shapes:
        blank = np.zeros(size, np.uint8)
        # Each polygon stored in shapes is a np.ndarray
        poly = shapes[u]

        # Creating a shapely polygon object out of the numpy array
        polygon = Polygon(poly)

        # Getting the center points from the polygon and the polygon points
        (poly_center_x, poly_center_y) = polygon.centroid.coords[0]
        polygon_points = polygon.exterior.coords

        # Setting a new polygon with each X,Y manipulated based off the center point
        shrunk_polygon = []
        for (x, y) in polygon_points:
            if x < poly_center_x:
                x += border
            elif x > poly_center_x:
                x -= border

            if y < poly_center_y:
                y += border
            elif y > poly_center_y:
                y -= border

            shrunk_polygon.append([x, y])

        # Transforming the polygon back to a np.ndarray
        ns_poly = np.array(shrunk_polygon, np.int32)

        # Filling the shrunken polygon to add a border between close polygons
        fillPoly(blank, [ns_poly], (1, 1, 1))
        mask_img += blank

    mask_img[mask_img > 4] = 0
    # mask_img[mask_img == 1] = 255
    return mask_img


def save_masks(masks, output_path, mask_file_name):
    """
    :param masks: dictionary of UID:masked polygons from mask_polygons_separately()
    :param output_path: path to save the masks
    :param mask_file_name: the file name the masks should have
    """
    # For each filled polygon, write out a separate file, increasing the name
    for m in masks:
        final_out = path.join(output_path,
                              mask_file_name + '_{}.png'.format(m))
        imwrite(final_out, masks[m])


def save_one_mask(masks, output_path, mask_file_name):
    """
    :param masks: list of masked polygons from the mask_polygons_separately function
    :param output_path: path to save the masks
    :param mask_file_name: the file name the masks should have
    """
    # For each filled polygon, write the mask shape out to the file per image
    mask_file_name = path.join(output_path, mask_file_name+ '.png')
    imwrite(mask_file_name, masks)


def read_json(json_path):
    """
    :param json_path: path to load json from
    :returns: a python dictionary of json features
    """
    annotations = json.load(open(json_path))
    return annotations


def get_feature_info(feature):
    """
    :param feature: a python dictionary of json labels
    :returns: a list mapping of polygons contained in the image
    """
    # Getting each polygon points from the json file and adding it to a dictionary of uid:polygons
    props = {}

    for feat in feature['features']['xy']:
        try:
            damage_type = feat['properties']['subtype']
        except:  # pre-disaster damage is default no-damage
            damage_type = "no-damage"
        feat_shape = wkt.loads(feat['wkt'])
        coords = list(mapping(feat_shape)['coordinates'][0])
        if damage_type == "no-damage":
            props[feat['properties']['uid']] = (np.array(coords, np.int32), 1)
        elif damage_type == "minor-damage":
            props[feat['properties']['uid']] = (np.array(coords, np.int32), 2)
        elif damage_type == "major-damage":
            props[feat['properties']['uid']] = (np.array(coords, np.int32), 3)
        elif damage_type == "destroyed":
            props[feat['properties']['uid']] = (np.array(coords, np.int32), 4)
        else:
            props[feat['properties']['uid']] = (np.array(coords, np.int32), 255)

    return props


def mask_chips(json_path, images_directory, output_directory, single_file, border):
    """
    :param json_path: path to find multiple json files for the chips
    :param images_directory: path to the directory containing the images to be masked
    :param output_directory: path to the directory where masks are to be saved
    :param single_file: a boolean value to see if masks should be saved a single file or multiple
    """
    # For each feature in the json we will create a separate mask
    # Getting all files in the directory provided for jsons

    MINOR_DAMAGE_NUM = 0
    MAJOR_DAMAGE_NUM = 0
    DESTROY_DAMAGE_NUM = 0

    jsons = [j for j in next(walk(json_path))[2] if '_post' in j]

    # After removing non-json items in dir (if any)
    for j in tqdm([j for j in jsons if j.endswith('json')],
                  unit='poly',
                  leave=False):
        # Our chips start off in life as PNGs
        chip_image_id = path.splitext(j)[0] + '.png'
        mask_file = path.splitext(j)[0]

        # Loading the per chip json
        j_full_path = path.join(json_path, j)
        chip_json = read_json(j_full_path)

        # Getting the full chip path, and loading the size dimensions
        chip_file = path.join(images_directory, chip_image_id)
        chip_size = get_dimensions(chip_file)

        # Reading in the polygons from the json file
        polys = get_feature_info(chip_json)

        # Getting a list of the polygons and saving masks as separate or single image files
        if len(polys) >= 0:
            if single_file:
                if border > 0:
                    masked_polys = mask_polygons_together_with_border(chip_size, polys, border)
                else:
                    masked_polys, SUB_MINOR_DAMAGE_NUM, SUB_MAJOR_DAMAGE_NUM, SUB_DESTROY_DAMAGE_NUM = mask_polygons_together(
                        chip_size, polys)
                    MINOR_DAMAGE_NUM += SUB_MINOR_DAMAGE_NUM
                    MAJOR_DAMAGE_NUM += SUB_MAJOR_DAMAGE_NUM
                    DESTROY_DAMAGE_NUM += SUB_DESTROY_DAMAGE_NUM
                save_one_mask(masked_polys, output_directory, mask_file)
            else:
                masked_polys = mask_polygons_separately(chip_size, polys)
                save_masks(masked_polys, output_directory, mask_file)
    print(MINOR_DAMAGE_NUM, MAJOR_DAMAGE_NUM, DESTROY_DAMAGE_NUM)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """generate_building_masks.py: Takes in xBD dataset and masks polygons in the image\n\n
        WARNING: This could lead to hundreds of output images per input\n""")

    parser.add_argument('--input',
                        metavar="./data/infra_damage/xBD/train",
                        default="./data/infra_damage/xBD/train",
                        help='Path to one xBD/xView2 split directory containing images/ and labels/')
    parser.add_argument('--single-file',
                        action='store_true',
                        default=True,
                        help='use to save all masked polygon instances to a single file rather than one polygon per mask file')
    parser.add_argument('--border',
                        default=0,
                        type=int,
                        metavar="positive integer for pixel border (e.g. 1)",
                        help='Positive integer used to shrink the polygon by')

    args = parser.parse_args()

    # Getting the list of the disaster types under the xBD directory
    # Create the full path to the images, labels, and mask output directories
    image_dir = path.join(args.input, 'images')
    json_dir = path.join(args.input, 'labels')
    output_dir = path.join(args.input, 'masks')

    if not path.isdir(image_dir):
        print(
            "Error, could not find image files in {}.\n\n"
                .format(image_dir),
            file=stderr)
        exit(2)

    if not path.isdir(json_dir):
        print(
            "Error, could not find labels in {}.\n\n"
                .format(json_dir),
            file=stderr)
        exit(3)

    if not path.isdir(output_dir):
        makedirs(output_dir)

    load_dependencies()
    mask_chips(json_dir, image_dir, output_dir, args.single_file, args.border)
