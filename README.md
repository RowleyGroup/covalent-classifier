# How to use

Using the python environment manager of your choice (Python~3.10):

```
$ pip install -r requirements.txt
```
If you want to simply to train the model, run the following from the main directory:
```
$ python models/train.py
```
# Generating a GradCAM heatmap
You can generate a class activation heatmap by running the following command from the main directory:
```
$ python models/gradcam.py Your_Smiles_Or_InChI_String
```
This will make a file named `gradcam_heatmap.png`. If you want to adapt this to your needs, refer to `models/gradcam.py` file itself.

# AIMNet data
Non-aiment data is available as InChI strings in the `./data` folder.
AIMNet data is too large to be hosted here, and can be found at this address: