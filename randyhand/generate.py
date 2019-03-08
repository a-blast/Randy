"""
randyhand -> generate

Giving you a random hand to write some training data for OCR.

GOAL: To try to emulate pictures of human handwriting using synthesized
text from the emnist data set

TODO:
1. Create base img canvas
2. Add random background noise/shapes/objects
3. Calculate how many lines of text & characters per line from uniform distribution
4. Calculate offset between lines (vertically) from narrow normal distribution
5. Get random text (random_word package), or use user supplied text
--> Could be word, number, or word with number appended.
6. Map letters to randomly selected members of the character's class
7. Impose letter spacing value sampled from a normal distribution
8. create word images based on selected characters and imposed variations
9. Merge images together w/ spaces to make lines
--> if more lines are remaining, go to next line at step 5
10. Set random base transformations (rotations, affine, grow/shrink) & transform canvas
##### In an image of handwriting, the direction, perspective,
##### and size will be "roughly" homogeneous, I would think
11. Update all x,y,w,h values for the bounding boxes
12. Translate annotation into XML
12. return image and XML

misc. TODO:
-> wrap this in a CLI
-> add to docker file w/ emnist
-> hook data directly into darkflow??? (save disk IO time...)
"""

import imutils
import cv2
import numpy as np


def generate(text=None, size=(500,500)):
    """User facing function for handling generation & annotation of images.

    :param text: User supplied text to put in image. If None, it is randomly generated
    :param size: Size of exported canvas
    :returns: the img and YOLO compatible XML annotation
    :rtype: {img: img, XML: XML}

    """
    base_canvas = np.zeros(size)
    regions_remaining = [size]

    #TODO implement the rest...
    pass



def get_letter_emnist_index(letter):
    """Get a random index for a specific letter class to slice the emnist dataset accordingly

    :param letter: single character to get a slice for
    :returns: a slice() object that will get a random member of the appropriate class in emnist
    :rtype: slice object

    """
    pass

