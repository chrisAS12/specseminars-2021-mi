def save_text_string(text, file_name):
    with open(file_name, "w") as file:
        file.write(text)
        file.close()

file_path = "sentences_seperated_by_lines_format.txt"

new_path_variable = "splitted_data_v" 

f = open(file_path, encoding="utf-8")
text = f.read()
sentences_list = text.split("\n")
file_count = 5
sentences_in_one_file = round(len(sentences_list) / 5)

currentText = ""
for i in range(len(sentences_list)): 
    currentText += sentences_list[i]+"\n"
    if i != 0 and i % sentences_in_one_file == 0:
        save_text_string(currentText, str(new_path_variable +  str(i/sentences_in_one_file)))
        currentText = ""
        print("saved file 1")
save_text_string(currentText, str(new_path_variable + "0.0"))        

        
        