# Randyhand

This is how you can run randyhand assuming you have the emnist dataset in a local dir (in a folder names emnist/).

### The data must have been downloaded from here: 
https://www.kaggle.com/crawford/emnist

Assuming you have virtualenv,

```{bash}
mkvirtualenv -p python3 randyhand
workon randyhand
pip install randyhand
python
```

Then, from the python terminal type

```{python}
import randyhand
randyhand.run(100)
exit()
```
for 100 generated text images, xml annotations, & corresponding strings.

(for the letter only script, see will_dev and follow the instructions there)
