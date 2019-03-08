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


def generate(text=None, size=(500,500)):
    """User facing function for handling generation & annotation of images.

    :param text: User supplied text to put in image. If None, it is randomly generated
    :param size: Size of exported canvas
    :returns: the img and YOLO compatible XML annotation
    :rtype: dict, {img: img, XML: XML}

    """
    width, height = size
    img_out = np.zeros((width, height, 3))
    base_canvas = np.zeros(size)

    character_height, space_between_lines = calculate_line_parameters(height)

    next_word = get_next_word_function(text)

    


    #TODO implement the rest...
    pass

def calculate_line_parameters(height):
    """Get random params (that make sense!) for lines to be written

    :param height: height of canvas
    :param width: width of canvas
    :returns: base character height & line spacing
    :rtype: (float,float)

    """
    # Assuming character height to be 32
    default_num_lines = height/32
    new_num_lines     = np.random.normal(default_num_lines,
                                           default_num_lines/2)

    new_num_lines = 1 if (new_num_lines < 1) else new_num_lines

    height_per_line     = height/new_num_lines
    space_between_lines = np.random.uniform(0, int(height_per_line/2))
    character_height    = height_per_line - space_between_lines

    return (character_height, space_between_lines)


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


def get_letter_emnist_index(letter):
    """Get a random index for a specific letter class to slice the emnist dataset accordingly

    :param letter: single character to get a slice for
    :returns: a slice() object that will get a random member of the appropriate class in emnist
    :rtype: slice object

    """
    pass

