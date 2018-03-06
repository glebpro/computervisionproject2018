#
#   Some general utilities to use throughout project.
#
#   @author Gleb Promokhov
#


def show_images(images, titles=[]):
    """
    Show some images in a single frame
    """
    if len(titles) == 0:
        titles = list(range(len(images)))

    fig = plt.figure()
    for i,image in enumerate(images):
        sub = fig.add_subplot(1,len(images),i+1)
        imgplot = plt.imshow(image)
        plt.axis('off')
        sub.set_title(titles[i])
    plt.show()

__all__ = ['show_images']
