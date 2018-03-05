#
#   Preprocess bird image data.
#
#   @author Gleb Promokhov
#
import os

def load_images(project_root=-1):
    if project_root == -1:
        project_root = os.path.dirname(os.path.realpath(__file__+"/.."))

    image_data_path = project_root + "/data/CUB_200_2011/"
    segmentations_data_path = project_root + "/data/segmentations/"

    # get class labels vector
    labels = open(image_data_path + "classes.txt").readlines()
    labels = [r.split()[1].strip() for r in labels]

    # get source image classes
    classes = open(image_data_path + "image_class_labels.txt").readlines()
    classes = [int(r.split()[1].strip()) for r in classes]

    # get source+segmentations image paths
    source_image_paths = open(image_data_path + "images.txt").readlines()
    source_image_paths = [r.split()[1].strip() for r in source_image_paths]
    segmentations_image_paths = [r.replace("jpg", "png") for r in source_image_paths]
    source_image_paths = [image_data_path + 'images/' + r for r in source_image_paths]
    segmentations_image_paths = [segmentations_data_path + r for r in segmentations_image_paths]

    # zip all into list of tuples of source data
    # [(class, source_image_path, segmentation_image_path), ...]
    data = list(zip(classes, source_image_paths, segmentations_image_paths))

    return labels, data

__all__ = ['load_images']
