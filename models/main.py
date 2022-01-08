from bert_0 import predict

def full_prediction(sentence, symbol_to_replace):
    print(sentence)
    print(f"Šajā tekstā mēs izmainīsim {symbol_to_replace} uz šo: ")
    prediction = predict(sentence,symbol_to_replace)
    print('\033[0m',"\033[1m",prediction[1]['sequence'],'\033[0m')
    print('Viss minējums: ', prediction)
    make_text_green()

def make_text_green():
    print('\033[92m')

make_text_green()

while(True):
    sentence = input("Ievadiet tekstu un simbolu, kuru vēlāk mainīsiet: ")
    if(sentence is None or sentence == '' or sentence == '-1'):
        break
    symbol = input("Simbols, kuru mainīt: ")
    full_prediction(sentence, symbol)
    print("Ievadiet '-1', lai izietu!")
    