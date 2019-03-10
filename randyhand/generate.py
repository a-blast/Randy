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


s=time.time()
img=generate(emnist)
o=time.time()-s

def generate(letterSource, text=None, size=(500,500)):
    """User facing function for handling generation & annotation of images.

    :param text: User supplied text to put in image. If None, it is randomly generated
    :param size: Size of exported canvas
    :returns: the img and YOLO compatible XML annotation
    :rtype: dict, {img: img, XML: XML}

    """
    init_char_size = 28
    width, height = size
    base_canvas = Image.fromarray(np.zeros(size))

    while True:
        char_size, space_between_lines, num_lines = calculate_line_parameters(size, init_char_size)
        max_letters_per_line = width/char_size
        if max_letters_per_line >= 3:
            break

    # next_word is a function
    next_word = get_next_word_function(text)

    # emnist = pd.read_csv("../data/emnist-balanced-train.csv", header=None)
    class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

    annotation_dict = {}
    emnist_dict = {}
    for index, letter in enumerate(class_mapping):
        is_letter = lambda df: (df[0] == index)
        emnist_dict[letter] = letterSource.loc[is_letter, 1:]

    letter_index = lambda letter: letter if letter in class_mapping else letter.upper()
    # NOTE: might want to make xOffset dynamically initialized
    xOffset = 0
    yOffset = space_between_lines
    num_characters_remaining = width//char_size
    punctuation_stripper = str.maketrans({key: None for key in string.punctuation})


    while(num_lines > 0):
        while(num_characters_remaining >= 3):
            word = next_word().decode("UTF-8").translate(punctuation_stripper)
            if len(word) > max_letters_per_line:
                print("HELP!")
                continue
            elif len(word) > num_characters_remaining:
                break

            for (index, letter) in enumerate(word):
                imgIn = Image.fromarray(np.uint8(np.reshape
                                                 (emnist_dict[letter_index(letter)].sample().values,
                                                  (init_char_size,init_char_size)))) \
                             .transpose(Image.TRANSPOSE) \
                             .resize((char_size,char_size))
                base_canvas.paste(imgIn, (xOffset, yOffset))
                xOffset = xOffset+char_size

            xOffset = xOffset + char_size

            num_characters_remaining = num_characters_remaining - len(word) - 1
    
        num_lines = num_lines - 1
        yOffset = yOffset + space_between_lines + char_size
        xOffset = 0
        num_characters_remaining = max_letters_per_line
    return base_canvas

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
    if text:
        text.reverse()
        next_word = lambda: text.pop() if text else "FIN"
    else:
        word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
        response = requests.get(word_site)
        text = response.content.splitlines()
        next_word = lambda: np.random.choice(text)
    return next_word

