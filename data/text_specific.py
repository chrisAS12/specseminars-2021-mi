# Data formatting for the required BERT specifics

from text_formatting_universal import save_text_string, get_text

def splitSentences(file):
    text = "".join(get_text(file, False))
    stuff_to_endline_after = ['.','!','?']
    for u in stuff_to_endline_after:
        text = text.replace(u, u +" \n")   
    return removeBlankLines(text)

def create_vocabulary(file):
    text = get_text(file)
    text_string = "".join(text)
    text_lower_case = text_string.lower()
    unwanted_stuff = "1 2 3 4 5 6 7 8 9 0 , ; : . ! ? * ( ) [ ] @ « ” / \\ _ …  – ' » “ - \" ' „" .strip(" ")
    for u in unwanted_stuff:
        text_lower_case = text_lower_case.replace(u, " ")
        print(u)
    return list(set(text_lower_case.split()))

files = ["data\LVK2013_FORMATTED.txt" , "data\TRAIN_SET_FORMATTED.txt"]

def removeBlankLines(text):
    newText = ""
    for i,e in enumerate(text.split("\n")):
        if not e.isspace() and any(c.isalpha() for c in e):
            newText += e.lstrip(" ") + "\n"
    return newText

#word_list = []
#for i in files:
#    word_list.append(create_vocabulary(i))
#
#text = ""
#for i in range(len(files)):
#    for w in word_list[i]:
#        if not len(w) > 25:
#            text += w + " "
#
#save_text_string(text, "word_list_v7.txt")

text = ""
for i in files:
    text += splitSentences(i)
print(text)

save_text_string(text, "sentences_seperated_by_lines_format.txt")
