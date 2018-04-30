#
#   Preprocess bird image data.
#
#   @author Gleb Promokhov
#   @author Greg Goh
#

import os
import errno
import pathlib
from shutil import copyfile

import matplotlib.pyplot as plt
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__+"/.."))

def load_images():
    """
    :return: labels of birds as strings, [(class_as_index, image_path, segmentations_path), ...]
    """

    image_data_path = PROJECT_ROOT + "/data/CUB_200_2011/"
    segmentations_data_path = PROJECT_ROOT + "/data/segmentations/"

    # get class labels vector
    classes = open(image_data_path + "classes.txt").readlines()
    classes = [r.split()[1].strip() for r in classes]

    # get source image classes
    labels = open(image_data_path + "image_class_labels.txt").readlines()
    labels = [int(r.split()[1].strip())-1 for r in labels]

    # get source+segmentations image paths
    source_image_paths = open(image_data_path + "images.txt").readlines()
    source_image_paths = [r.split()[1].strip() for r in source_image_paths]
    segmentations_image_paths = [r.replace("jpg", "png") for r in source_image_paths]
    source_image_paths = [image_data_path + 'images/' + r for r in source_image_paths]
    segmentations_image_paths = [segmentations_data_path + r for r in segmentations_image_paths]

    # zip all into list of tuples of source data
    # [(class, source_image_path, segmentation_image_path), ...]
    data = list(zip(labels, source_image_paths, segmentations_image_paths))

    return classes, data

def convert_to_edges(source_folder):
    """
    Convert directory of images into edges.
    """
    image_paths = []
    for filepath in pathlib.Path(source_folder).glob('**/*png'):
        image_paths.append(filepath.absolute())

    for i in image_paths:
        img = cv2.imread(str(i))
        edges = cv2.Canny(img,100,200)
        cv2.imwrite(str(i), img)

def apply_segmentations(classes, data):
    """
    Apply segmentations to images, save them into /data/
    return classes, [(label, segmented_image_path), ...]
    """

    segmented_data =[]

    output_dir = PROJECT_ROOT + "/data/segmented_images/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for r in data:
        print("~~~ SEGMENTING: "+r[1])
        seg = cv2.imread(r[2])
        img = cv2.imread(r[1])
        img2 = cv2.bitwise_and(img, seg)
        filename = r[1].split("/")
        filename = filename[len(filename)-1].split(".")[0]
        if not os.path.exists(output_dir+classes[r[0]]):
            os.makedirs(output_dir+classes[r[0]])
        cv2.imwrite(output_dir+classes[r[0]]+"/"+filename+".png", img2)
        segmented_data.append((r[0],output_dir+classes[r[0]]+"/"+filename+".png"))

    return segmented_data

def plot_class_distribution(data):
    """
    Plot class distribution to catch class imbalances
    """
    classes = [r[0] for r in data]
    plt.hist(classes)
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title('Histogram of class counts')
    plt.show()

def split_dir(dirr, output_dir, dirs=['train', 'validation', 'test'], split=(.5,.25,.25)):
    """
    Split a labeled image directory into train/validation/test dirs.
    """

    # get all image paths
    image_paths = []
    for filepath in pathlib.Path(dirr).glob('**/*'):
        image_paths.append(filepath.absolute())

    # organize into {class_name:[class_image_paths, ...], ...}
    class_dict = {}
    for i in image_paths:
        fname = str(i).split("/")
        file_name = fname[len(fname)-1]
        class_name = fname[len(fname)-2]
        if class_name not in class_dict.keys():
            class_dict[class_name] = []
        class_dict[class_name].append(str(i))

    del class_dict['segmented_head_images'] #I don't know why

    # organize into {class_name:[[train_paths],[validation_paths],[test_paths]], ...}
    # by given
    for k in class_dict.keys():
        paths = class_dict[k]

        train_split = int(len(paths)*split[0])
        validation_split = int(len(paths)*split[1])

        train_paths = paths[train_split:]
        validation_paths = paths[train_split:validation_split+train_split]
        test_paths = paths[validation_split+train_split:]

        class_dict[k] = [train_paths, validation_paths, test_paths]

    # make output dirs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir+"/"+dirs[0])
        os.makedirs(output_dir+"/"+dirs[1])
        os.makedirs(output_dir+"/"+dirs[2])

    # move everything
    for k in class_dict.keys():
        for d_i,d in enumerate(dirs):

            if not os.path.exists(output_dir+"/"+d+"/"+k):
                os.makedirs(output_dir+"/"+d+"/"+k)

            for path in class_dict[k][d_i]:
                file_name = path.split("/")
                file_name = file_name[len(file_name)-1]
                copyfile(path, output_dir+"/"+d+"/"+k+"/"+file_name)

def segment_heads(classes, data):
    """
    Remove 'non-head' parts of birds.
    """

    segmented_data =[]

    # gather and organize needed data
    output_dir = PROJECT_ROOT + "/data/segmented_head_images/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_ids_file = open(PROJECT_ROOT + '/data/CUB_200_2011/images.txt').readlines()
    img_ids_file = [i.strip().split(' ') for i in img_ids_file]

    parts_file = open(PROJECT_ROOT +'/data/CUB_200_2011/parts/part_locs.txt').readlines()
    parts_file = [i.strip().split(' ') for i in parts_file]

    # <image_id> <x> <y> <width> <height>
    bounding_file = open(PROJECT_ROOT +'/data/CUB_200_2011/bounding_boxes.txt').readlines()
    bounding_file = [i.strip().split(' ') for i in bounding_file]

    img_ids = {}
    for i in img_ids_file:
        img_ids[i[1]] = int(i[0])

    part_ids = {}
    for i in parts_file:
        part_ids[(int(i[0]), int(i[1]))] = list(map(lambda x:int(float(x)), i[2:]))

    boudning_ids = {}
    for i in bounding_file:
        boudning_ids[int(i[0])] = list(map(lambda x:int(float(x)), i[1:]))

    for r in data:
        print("~~~SEGMENTING HEAD: ", r[1])

        img_id = r[1].split('/')
        img_id = img_id[len(img_id)-2] + '/' + img_id[len(img_id)-1].replace('png', 'jpg')
        img_id = img_ids[img_id]

        # get location of bird parts
        nape = part_ids[(img_id, 10)]
        tail = part_ids[(img_id, 14)]
        throat = part_ids[(img_id, 15)]
        bounds = boudning_ids[img_id]

        # if any of that parts not visible
        if nape[2] == 0 or tail[2] == 0 or throat[2] == 0 or nape[1] - throat[1] == 0:
            continue

        # compute on what side of nape-throat line tail is on
        tail_side = (tail[1] - throat[0])*(nape[1] - throat[1]) - (tail[0] - throat[1])*(throat[0] - nape[0])

        img = cv2.imread(r[1])
        (rows, cols, _) = img.shape

        # all pixels on same side of nape-throat line as tail turn off
        for y in range(0,rows):
            for x in range(0,cols):
                v1 = (nape[0]-throat[0], nape[1] - throat[0])
                v2 = (x - throat[0], y - throat[1])
                c_p = v1[0]*v2[1]-v1[1]*v2[0]
                if np.sign(tail_side) != np.sign(c_p):
                    img[y, x, :] = 0

        # crop by boudning box
        img = img[bounds[1]:bounds[1]+bounds[3], bounds[0]:bounds[0]+bounds[2], :]

        # save
        filename = r[1].split("/")
        filename = filename[len(filename)-1].split(".")[0]
        if not os.path.exists(output_dir+classes[r[0]]):
            os.makedirs(output_dir+classes[r[0]])
        cv2.imwrite(output_dir+classes[r[0]]+"/"+filename+".png", img)
        segmented_data.append((r[0],output_dir+classes[r[0]]+"/"+filename+".png"))

    return segmented_data

def image_classes():
    """
    Read in class labels of the images
    """

    image_data_path = PROJECT_ROOT + "/data/CUB_200_2011/"

    # <class_id> <class_name>
    classes = open(image_data_path + "classes.txt").readlines()
    classes = [i.strip().split() for i in classes]

    # <image_id> <class_id>
    labels = open(image_data_path + "image_class_labels.txt").readlines()
    labels = [i.strip().split() for i in labels]

    class_ids = {}
    for i in classes:
        class_ids[i[1]] = int(i[0])

    label_ids = {}
    for i in labels:
        label_ids[int(i[0])] = int(i[1])

    return class_ids, label_ids

def load_attributes():
    """
    Read in attribute labels and certainties.
    """

    # <attribute_id> <attribute_name>
    attributes_file = open(PROJECT_ROOT +'/data/attributes.txt').readlines()
    attributes_file = [i.strip().split() for i in attributes_file]

    # <certainty_id> <certainty_name>
    certainties_file = open(PROJECT_ROOT +'/data/CUB_200_2011/attributes/certainties.txt').readlines()
    certainties_file = [i.strip().split() for i in certainties_file]

    # <image_id> <attribute_id> <is_present> <certainty_id> <time>
    labels_file = open(PROJECT_ROOT +'/data/CUB_200_2011/attributes/image_attribute_labels.txt').readlines()
    labels_file = [i.strip().split() for i in labels_file]

    attribute_ids = {}
    for i in attributes_file:
        attribute_ids[i[1]] = int(i[0])

    certainty_ids = {}
    for i in certainties_file:
        certainty_ids[i[1]] = int(i[0])

    label_ids = {}
    for i in labels_file:
        label_ids[(int(i[0]), int(i[1]))] = list(map(lambda x:int(float(x)), i[2:]))

    return attribute_ids, certainty_ids, labels_file, label_ids

def check_data_struct():
    """
    Check that all data is in place first
    """
    if not os.path.exists(PROJECT_ROOT+'/data'):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), PROJECT_ROOT+'/data')

    if not os.path.exists(PROJECT_ROOT+'/data/CUB_200_2011'):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), PROJECT_ROOT+'/data/CUB_200_2011')

    if not os.path.exists(PROJECT_ROOT+'/data/segmentations'):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), PROJECT_ROOT+'/data/segmentations')

    if not os.path.exists(PROJECT_ROOT+'/data/attributes.txt'):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), PROJECT_ROOT+'/data/attributes.txt')

def main():

    check_data_struct()

    # classes = ['001.Black_footed_Albatross', '002.Laysan_Albatross', ...]
    # data = [(class, source_image_path, segmentation_image_path), ...]
    classes, data = load_images()

    # plot_class_distribution(data)

    # data = [(class, segmented_image_path), ...]
    data = apply_segmentations(classes, data)

    # data = [(class, segmented_head_image_path), ...]
    data = segment_heads(classes, data)

    # split each class subdirectory into train/validation/test subdirectories
    split_dir(PROJECT_ROOT + "/data/segmented_head_images/", PROJECT_ROOT + "/data/split_segmented_head_images/")

    # convert_to_edges("/Users/gpro/gpc/rit/compvis/BirdNet/data/split_segmented_images_5_edges")


if __name__ == "__main__":
    main()
