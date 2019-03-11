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

import numpy as np
import requests
from PIL import Image
from functools import partial, reduce
import math
import string
import pandas as pd
import xml.etree.cElementTree as ET



def getGenerator(text=None, size=(500,500)):
    """User facing function for handling generation & annotation of images.

    :param text: User supplied text to put in image. If None, it is randomly generated
    :param size: Size of exported canvas
    :returns: the img and YOLO compatible XML annotation
    :rtype: dict, {img: img, XML: XML}

    """
    init_char_size = 28
    width, height = size
    emnist = pd.read_csv("../data/emnist-balanced-train.csv", header=None)
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
                    imgIn = Image.fromarray(np.uint8(np.reshape
                                                     (emnist_dict[letter_index(letter)] \
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

def apply_random_transform(imgObj):
    img = imgObj["img"]
    annotations = imgObj["annotations"]
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)

        #res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        res = np.dot(np.linalg.inv(A), B)
        return np.array(res).reshape(8)

    width, height = img.size
    pa = [(0, 0), (width, 0), (width, height), (0, height)]

    sF1 = int(np.random.uniform(0,0.4)*width)
    sF2 = int(np.random.uniform(0,0.4)*height)
    pivot = int(np.random.uniform(0, min(sF1,sF2)))
    left_in = [(-sF1, -sF2-pivot), (width+sF1, -sF2+pivot),
               (width+sF1, height+sF2-pivot), (-sF1, height+sF2+pivot)]
    right_in = [(-sF1, -sF2+pivot), (width+sF1, -sF2-pivot),
                (width+sF1, height+sF2+pivot), (-sF1, height+sF2-pivot)]
    bottom_in = [(-sF1+pivot, -sF2), (width+sF1-pivot, -sF2),
                 (width+sF1+pivot, height+sF2), (-sF1-pivot, height+sF2)]

    top_in = [(-sF1-pivot, -sF2), (width+sF1+pivot, -sF2),
              (width+sF1-pivot, height+sF2), (-sF1+pivot, height+sF2)]

    lc_in = [(-sF1-pivot, -sF2-pivot), (width+sF1, -sF2),
                  (width+sF1+pivot, height+sF2+pivot), (-sF1, height+sF2)]
    rc_in = [(-sF1+pivot, -sF2+pivot), (width+sF1, -sF2),
                  (width+sF1-pivot, height+sF2-pivot), (-sF1, height+sF2)]

    transforms = [left_in, right_in, bottom_in, top_in, lc_in, rc_in]

    left_in = [(-100, 0), (width, 0),
               (width, height), (0, height)]

    # coeffs = find_coeffs(pa, transforms[np.random.random_integers(0,len(transforms)-1)])
    coeffs = find_coeffs(pa, left_in)
    print(coeffs)
    img = img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    annotations = list(map(lambda annotation: apply_transform_annotations(coeffs, annotation),
                      annotations))
    return {"img":img, "annotations": annotations}

def apply_transform_annotations(coeffs, annotation):
    a, b, c, d, e, f, g, h = coeffs
    x_min, y_min, x_max, y_max = annotation[1]

    calc_new_point = lambda p: ((a*p[0]+b*p[1]+c)/(g*p[0]+h*p[1]+1),(d*p[0]+e*p[1]+f)/(g*p[0]+h*p[1]+1))

    bounding_points = [(x_min, y_min), (y_min, x_max), (x_min, y_max), (x_max, y_max)]
    new_bounding_points = list(map(calc_new_point, bounding_points))
    print(new_bounding_points)
    x_min = min(new_bounding_points[:][0])
    x_max = max(new_bounding_points[:][0])
    y_min = min(new_bounding_points[:][1])
    y_max = max(new_bounding_points[:][1])

    #return (annotation[0], (x_min, y_min, x_max, y_max))
    return (annotation[0], new_bounding_points)

def to_XML(annotations, imgSize):
    root = ET.Element("annotation")
    tree = ET.ElementTree(root)
    folder = ET.SubElement(root, "folder").text = "randyhand_data"
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width").text = str(imgSize[0])
    height = ET.SubElement(size, "height").text = str(imgSize[1])
    depth = ET.SubElement(size, "depth").text = "3"
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
        ET.SubElement(box, "xmax").text = str( annotation[1][1] )
        ET.SubElement(box, "ymin").text = str( annotation[1][2] )
        ET.SubElement(box, "ymax").text = str( annotation[1][3] )
        
    return tree

def calculate_line_parameters(size, letter_size):
    """Get random params (that make sense!) for lines to be written

    :param height: height of canvas
    :param width: width of canvas
    :returns: base character height & line spacing
    :rtype: (float,float)

    """
    # Assuming character height to be 32
    height, width = size
    default_num_lines = height/letter_size
    new_num_lines     = np.random.normal(default_num_lines,
                                           default_num_lines/2)

    new_num_lines = 1 if (new_num_lines < 1) else int(math.floor(new_num_lines))

    height_per_line     = height/new_num_lines
    space_between_lines = np.random.uniform(0, int(height_per_line/2))
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

