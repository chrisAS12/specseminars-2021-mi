import os

def get_text(file_name, split):
    if not os.path.isfile(file_name):
        print(" no file found! ", file_name)
    else:
        f = open(file_name, encoding="utf-8")
        text = f.read()
        if split == True:
            text = text.split("\n")
        return text
    return -1


def remove_punctuation(allText):
    list_unformatted_text = allText.split("\n")
    list_formatted_text = []
    for i in list_unformatted_text:
        text = "".join(t for t in i if t not in ("?", ".", ";", ":", "!", '"'))
        list_formatted_text.append(text)
    return list_formatted_text


def remove_unnecessary_elements(text):
    correct_text_list = []
    for i, e in enumerate(text):
        if e.startswith("<") or "<" in e or "<doc" in e or e.find('source') > 0 or e.find('</') > 0 or e.find('<') > 0:
            print("deleted ", i,  "   ", e)
        else:
            correct_text_list.append(e)
    return correct_text_list


def save_text(text, file_name):
    with open(file_name, 'w') as file:
        file.writelines("% s\n" % line for line in text)
        file.close()


def save_text_string(text, file_name):
    with open(file_name, "w") as file:
        file.write(text)
        file.close()


def format_text_manually():
    file = "data\LVK2013.txt"
    save_file_name = "data\LVK2013_FORMATTED.txt"
    text = get_text(file)
    print(text)
    if text != (-1):
        save_text(remove_unnecessary_elements(text), save_file_name)
