from .generate import *
import os, sys, functools, math
import xml.etree.cElementTree as ET

def run(numImgs):
    cwd = os.getcwd()
    if not os.path.exists(cwd+"/randyhand_data"):
        os.mkdir(cwd+"/randyhand_data")
        os.mkdir(cwd+"/randyhand_data/img")
        os.mkdir(cwd+"/randyhand_data/annotations")
        os.mkdir(cwd+"/randyhand_data/text")

    emnist_path = cwd+"/emnist"
    gen_data = getGenerator(emnist_path)

    for imgNum in range(numImgs):

        print(imgNum)
        out = gen_data()

        entry_name = str(imgNum).zfill(math.floor(math.log(numImgs))+1)

        out["img"].convert("RGB").save(cwd+"/randyhand_data/img/"+entry_name+".jpg")

        xml = out["annotations"]
        xml = to_XML(xml,out["img"].size)
        root = xml.getroot()

        ET.SubElement(root, "filename").text = entry_name+".jpg"

        xml.write(open(cwd+"/randyhand_data/annotations/"+entry_name+".xml", "w+"),
                  encoding="unicode")

        text_in_img = functools.reduce(
            lambda text, letter: text+letter,
            list(map(lambda annotation: annotation[0], out["annotations"])))

        string_file = open(cwd+"/randyhand_data/text/"+entry_name+".txt", "w+")
        string_file.write(text_in_img)
        string_file.close()
