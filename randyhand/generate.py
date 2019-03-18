# -*- coding: utf-8 -*-

"""randyhand -> generate.

Giving you a random hand to write some training data for OCR.

GOAL: To try to emulate pictures of human handwriting using synthesized
text from the emnist data set

TODO:
DONE 1. Create base img canvas
2. Add random background noise/shapes/objects
DONE 3. Calculate how many lines of text & characters per line from uniform distribution DONE
DONE 4. Calculate offset between lines (vertically) from narrow normal distribution DONE
DONE 5. Get random text (random_word package), or use user supplied text DONE
--> Could be word, number, or word with number appended.
DONE 6. Map letters to randomly selected members of the character's class DONE
DONE 7. Impose letter spacing value sampled from a normal distribution DONE
DONE 8. create word images based on selected characters and imposed variations DONE
DONE 9. Merge images together w/ spaces to make lines DONE
--> if more lines are remaining, go to next line at step 5
10. Set random base transformations (rotations, affine, grow/shrink) & transform
##### In an image of handwriting, the direction, perspective,
##### and size will be "roughly" homogeneous, I would think
11. Update all x,y,w,h values for the bounding boxes
DONE 12. Translate annotation into XML
DONE 12. return image and XML

misc. TODO:
DOING -> wrap this in a CLI
-> add to docker file w/ emnist
"""

import numpy as np
import requests
from PIL import Image
import math
import string
import pandas as pd
import xml.etree.cElementTree as ET



def getGenerator(emnist_path, by_merge, text=None, size=(608, 608)):
    """User facing function for handling generation & annotation of images.

    :param text: User supplied text to put in image.
    If None, it is randomly generated
    :param size: Size of exported canvas
    :returns: the img and YOLO compatible XML annotation
    :rtype: dict, {img: img, XML: XML}

    """
    init_char_size = 28
    width, height = size
    data_ext =  "/emnist-bymerge-train.csv" if by_merge else  "/emnist-balanced-train.csv"
    emnist = pd.read_csv(emnist_path + data_ext, header=None)
    class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
    # next_word is a function
    next_word = get_next_word_function(text)

    emnist_dict = {}
    for index, letter in enumerate(class_mapping):
        is_letter = lambda df: (df[0] == index)
        emnist_dict[letter] = emnist.loc[is_letter, 1:]

    def generator():
        annotations = []
        base_canvas = Image.fromarray(np.zeros(size))

        while True:
            char_size, space_between_lines, num_lines = calculate_line_parameters(size, init_char_size)
            max_letters_per_line = width/char_size
            if max_letters_per_line >= 3:
                break

        letter_index = lambda letter: letter if letter in class_mapping else letter.upper()

        # NOTE: might want to make xOffset dynamically initialized
        xOffset = 0
        yOffset = space_between_lines
        num_characters_remaining = width//char_size

        while(num_lines > 0):
            while(num_characters_remaining >= 3):
                word = next_word()
                if len(word) > max_letters_per_line:
                    continue
                elif (len(word) > num_characters_remaining
                      or yOffset+char_size > height):
                    break

                for letter in word:
                    letter = letter_index(letter)
                    imgIn = Image.fromarray(np.uint8(np.reshape
                                                     (emnist_dict[letter] \
                                                      .sample().values,
                                                      (init_char_size,init_char_size)))) \
                                 .transpose(Image.TRANSPOSE) \
                                 .resize((char_size,char_size))
                    base_canvas.paste(imgIn, (xOffset, yOffset))
                    annotations.append((letter, (xOffset, yOffset,
                                                 xOffset+char_size, yOffset+char_size)))
                    xOffset = xOffset+char_size

                annotations.append(("_", (xOffset, yOffset,
                                          xOffset+char_size, yOffset+char_size)))
                xOffset = xOffset + char_size

                num_characters_remaining = num_characters_remaining - len(word) - 1
            
            num_lines = num_lines - 1
            yOffset = yOffset + space_between_lines + char_size
            xOffset = 0
            num_characters_remaining = max_letters_per_line
        return {"img":base_canvas, "annotations":annotations}

    return generator


def to_XML(annotations, imgSize):
    """get XML based on annotation format of generate
    """
    root = ET.Element("annotation")
    tree = ET.ElementTree(root)
    ET.SubElement(root, "folder").text = "annotations"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(imgSize[0])
    ET.SubElement(size, "height").text = str(imgSize[1])
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented"). text = "0"

    for annotation in annotations:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = annotation[0]
        ET.SubElement(obj, "pose").text = "Frontal"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "occluded").text = "0"
        box = ET.SubElement(obj, "bndbox")
        ET.SubElement(box, "xmin").text = str(annotation[1][0])
        ET.SubElement(box, "xmax").text = str( annotation[1][2] )
        ET.SubElement(box, "ymin").text = str( annotation[1][1] )
        ET.SubElement(box, "ymax").text = str( annotation[1][3] )

    return tree

def calculate_line_parameters(size, letter_size):
    """Get random params (that make sense!) for lines to be written

    :param height: height of canvas
    :param width: width of canvas
    :returns: base character height & line spacing
    :rtype: (float,float)

    """
    height, width = size
    max_chars_per_line = 16
    default_num_lines = height/letter_size
    new_num_lines     = int(np.random.uniform(1, 10))

    height_per_line     = height/new_num_lines
    space_between_lines = np.random.uniform(0, int(height_per_line/3))
    character_height    = height_per_line - space_between_lines

    return (int(math.ceil(character_height)), int(math.floor(space_between_lines)), new_num_lines)


def get_next_word_function(text):
    """Function generator closure that gets the next word depending if text was provided or not

    :param text: list of strings or None
    :returns: a function to get the next word
    :rtype: function

    """
    punctuation_stripper = str.maketrans({key: None for key in string.punctuation})
    if text:
        text.reverse()
        next_word = lambda: text.pop().translate(punctuation_stripper) if text else "FIN"
    else:
        word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
        response = requests.get(word_site)
        text = response.content.splitlines()
        next_word = lambda: np.random.choice(text) \
                                     .decode("UTF-8") \
                                     .translate(punctuation_stripper)
    return next_word

