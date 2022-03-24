# This script generate a csv file according to the dataset.
# File contains two columns: the absolute path to each image and the image label.
# The CSV file can be later used to load data.

# import the necessary packages
import argparse
import os
import glob
import pandas as pd

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, \
    help="path to root of  the dataset")
ap.add_argument("-o", "--output", required=True, \
    help="path to the generated csv file")
args = vars(ap.parse_args())

# save the original working directory for later to restore
original_cwd = os.getcwd()

# change the current working directory to the path of the root of the dataset
os.chdir(args["dataset"])

# get the list of class names by getting the name of sub-directories of the root
class_list = glob.glob(os.getcwd() + "/*")

# for every class (i.e. sub-directory of root), add every image's path and label
# to the list
path = []
label = []
for class_label, class_path in enumerate(class_list):
    for img_path in glob.glob(class_path + "/*.jpg"):
        path.append(img_path)
        label.append(class_label)

# restore the current working directory and save csv file to the output path
os.chdir(original_cwd)
data = {'path':path, 'label': label}
df = pd.DataFrame(data)
df.to_csv(args["output"])
