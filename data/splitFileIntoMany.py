def save_text_string(text, file_name):
    with open(file_name, "w") as file:
        file.write(text)
        file.close()

file_path = "sentences_seperated_by_lines_format.txt"

new_path_variable = "splitted_data_v" 

f = open(file_path, encoding="utf-8")
text = f.read()
sentencesList = text.split("\n")
howManyFiles = 5
sentencesBeforeNewFile = round(len(sentencesList) / 5)

currentText = ""
for i in range(len(sentencesList)): 
    currentText += sentencesList[i]+"\n"
    if i != 0 and i % sentencesBeforeNewFile == 0:
        save_text_string(currentText, str(new_path_variable +  str(i/sentencesBeforeNewFile)))
        currentText = ""
        print("saved file 1")
save_text_string(currentText, str(new_path_variable + "0.0"))        

        
        