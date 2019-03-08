# Code based on https://github.com/CentralLabFacilities/object_recognition/blob/master/scripts/darknet_to_darkflow.py

import sys
import os
from shutil import copyfile
from lxml import etree

def createXML(annotations_list, height, width, depth, ID):
    annotation = etree.Element('annotation')

    fo = etree.Element('folder')
    fo.text = '/Images'

    annotation.append(fo)

    f = etree.Element('filename')
    f.text = ID + '.jpg'

    annotation.append(f)

    size = etree.Element('size')
    w = etree.Element('width')
    w.text = str(width)
    h = etree.Element('height')
    h.text = str(height)
    d = etree.Element('depth')
    d.text = str(depth)

    size.append(w)
    size.append(h)
    size.append(d)

    annotation.append(size)

    seg = etree.Element('segmented')
    seg.text = str(0)

    annotation.append(seg)

    for j in range(len(annotations_list)): # Each bounding box

        object = etree.Element('object')
        n = etree.Element('name')
        p = etree.Element('pose')
        t = etree.Element('truncated')
        d_1 = etree.Element('difficult')
        bb = etree.Element('bndbox')

        n.text = str(annotations_list[j][0])#classname
        p.text = 'center'
        t.text = str(1)
        d_1.text = str(0)

        xmi = etree.Element('xmin')
        ymi = etree.Element('ymin')
        xma = etree.Element('xmax')
        yma = etree.Element('ymax')

        xmi.text = str(annotations_list[j][1])
        yma.text = str(annotations_list[j][4])
        ymi.text = str(annotations_list[j][3])
        xma.text = str(annotations_list[j][2])

        bb.append(xmi)
        bb.append(ymi)
        bb.append(xma)
        bb.append(yma)

        object.append(n)
        object.append(p)
        object.append(t)
        object.append(d_1)
        object.append(bb)

        annotation.append(object)

    return annotation

def saveXML(xml, filename):
    path = filename + '.xml'
    debug=False

    if(debug):
        print ('Creating file ' + path + ':')

    with open(path, "wb") as file:
        file.write(etree.tostring(xml, pretty_print=True))











