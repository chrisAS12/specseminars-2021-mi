import os
import string
import sys

def get_text(fileName):
    if not os.path.isfile(fileName):
        print(" no file found! ")
    else:
        f = open(fileName, encoding="utf-8")
        text = f.read().split("\n")
        return text
    return -1


def removePunctuation(allText):
    listUnformattedText = allText.split("\n")
    listFormattedText = []
    for i in listUnformattedText:
        text = "".join(t for t in i if t not in ("?", ".", ";", ":", "!", '"'))
        listFormattedText.append(text)
    return listFormattedText


def removeDoc(text):
    correctTextList = []
    for i,e in enumerate(text):
        if e.startswith("<") or "<" in e or "<doc" in e or e.find('source') > 0 or e.find('</') > 0 or e.find('<') > 0:
            print("deleted ", i ,  "   ", e)
        else:
            correctTextList.append(e)
    return correctTextList


def saveText(text, fileName):
    with open(fileName, 'w') as file:
        file.writelines("% s\n" % line for line in text)
        file.close()

file = "data\LVK2013.txt"
saveFileName = "data\LVK2013_FORMATTED.txt"
text = get_text(file)
print(text)
if text != (-1):
    saveText(removeDoc(text), saveFileName)

    #text = removeDoc(text[:1000])
    #for e in text:
    #    print(e)