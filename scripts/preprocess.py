#
#   Preprocess bird image data.
#
#   @author Gleb Promokhov
#

import os
import errno
import pathlib
from shutil import copyfile

import matplotlib.pyplot as plt
import cv2

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

def apply_segmentations(classes, data):
    """
    Apply segmentations to images, save them into /data/
    return classes, [(label, segmented_image_path), ...]
    """

    segmented_data =[]

    output_dir = PROJECT_ROOT + "/data/segmented_images/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_list = PROJECT_ROOT + "/data/segmented_images/segmented_image_paths.txt"
    output_file_list = open(output_file_list, "w+")

    for r in data:
        print("~~~ SEGMENTING: "+r[1])
        seg = cv2.imread(r[2])
        img = cv2.imread(r[1])
        img2 = cv2.bitwise_and(img, seg)
        filename = r[1].split("/")
        filename = filename[len(filename)-1].split(".")[0]
        if not os.path.exists(output_dir+classes[r[0]]):
            os.makedirs(output_dir+classes[r[0]])
        cv2.imwrite(output_dir+classes[r[0]]+"/"+filename+"_SEGMENTED.png", img2)
        output_file_list.write(output_dir+classes[r[0]]+"/"+filename+"_SEGMENTED.png\n")
        segmented_data.append((r[0],output_dir+classes[r[0]]+"/"+filename+"_SEGMENTED.png"))

    output_file_list.close()

    return classes, segmented_data

def augment_images():
    """
    Augment image data set
    """
    if not os.path.exists(PROJECT_ROOT+'/data/segmented_images'):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), PROJECT_ROOT+'/data/segmented_images')

    # load segmented images
    segmented_data_path = PROJECT_ROOT + "/data/segmented_images"
    image_paths = []
    for filepath in pathlib.Path(segmented_data_path).glob('**/*'):
        image_paths.append(filepath.absolute())

    #
    output_dir = PROJECT_ROOT + "/data/segmented_images/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


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

    del class_dict['segmented_images'] #I don't know why

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
                print(k)
                copyfile(path, output_dir+"/"+d+"/"+k+"/"+file_name)

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

def main():

    check_data_struct()

    classes, data = load_images()

    # plot_class_distribution(data)

    apply_segmentations(classes, data)

    #augment_images()

    split_dir("/Users/gpro/gpc/rit/compvis/BirdNet/data/segmented_images", "/Users/gpro/gpc/rit/compvis/BirdNet/data/split_segmented_images")

if __name__ == "__main__":
    main()
