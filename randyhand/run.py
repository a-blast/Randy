from .generate import *
import os, sys, functools

def run(numImgs):
    cwd = os.getcwd()
    if not os.path.exists(cwd+"/randyhand_data"):
        os.mkdir(cwd+"/randyhand_data")
        os.mkdir(cwd+"/randyhand_data/img")
        os.mkdir(cwd+"/randyhand_data/annotations")
        os.mkdir(cwd+"/randyhand_data/text")

    emnist_path = cwd+"/emnist"
    gen = getGenerator(emnist_path)

    for imgNum in range(numImgs):
        print(imgNum)
        out = gen()
        out["img"].convert("RGB").save(cwd+"/randyhand_data/img/"+str(imgNum).zfill(3)+".jpg")

        xml = out["annotations"]
        xml = to_XML(xml,out["img"].size)
        xml.write(open(cwd+"/randyhand_data/annotations/"+str(imgNum).zfill(3)+".xml", "w+"),
                  encoding="unicode")

        string = functools.reduce(
            lambda text, letter: text+letter,
            list(map(lambda annotation: annotation[0], out["annotations"])))

        string_file = open(cwd+"/randyhand_data/text/"+str(imgNum).zfill(3)+".txt", "w+")
        string_file.write(string)
        string_file.close()
