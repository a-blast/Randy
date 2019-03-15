# Randyhand

## This is how you can run randyhand!

1. Get the emnist dataset (FROM KAGGLE) https://www.kaggle.com/crawford/emnist
2. Move it to a local dir named emnist/.
3. Run:

```{bash}
mkvirtualenv -p python3 randyhand
workon randyhand
pip install randyhand
python
```
4. From the python terminal type

```{python}
import randyhand
randyhand.run(100)
exit()
```
for 100 generated text images, xml annotations, & corresponding strings.

They will be saved in a local dir named randyhand_data. 

## If you run into errors using the synthesized data with YOLO, please check PyPi and make sure you are running the most recent version. 

#### This repo is still undergoing developement, if you find something that isnt working as expected please open a issue ticket here.

### Classes ouput in the annotations:
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, a, b, d, e, f, g, h, n, q, r, t, _

(for the letter only script, see will_dev and follow the instructions there)
