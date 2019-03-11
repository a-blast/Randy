"""

randyhand -> generate

Functions to handle transformation of the generated text image & annotations

TODO: Make annotation transforms work!
The homography matrix can be calculated, (after alot of trial & error, this may be a PIL nuance, notice the weird negative points in the pb (destination vertices) required to make the perspective transform work).

However, when applying the homography matrix to the points of the bounding boxes the output is wildly incorrect (out of bounds, negative, etc).

TODO: refactor once transformations are properly applied. May need to use openCV instead of PIL.

"""

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
