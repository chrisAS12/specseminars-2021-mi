import os
import string
import sys
import names

def get_text( fileName ):

    if not os.path.isfile( fileName ):
        print( " no file found! " )
    else:
        f = open(fileName, encoding="utf-8")
        text = f.read()
        return text
    return 0

def format_text(text):

def save_text(fileName):

string = "".join(u for u in username if u not in ("?", ".", ";", ":", "!"))

files = ["data\train\LV_train_unformatted_1.txt"]
for i in files:
    get_text(files)


formattedFileNames = []