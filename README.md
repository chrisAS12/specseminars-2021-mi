# Word prediction using BERT / RoBERTa and other models.
This is [Krišjāņa Mārtiņa Allika] course project for the NLP and AI course at the [University of Latvia], 2021/2022.

## Aim of this:
The goal of this project was to better understand NLP (Natural Language Processing), learn python, PyTorch, and machine learning models - how they work, how to create them, how are they used -  and create a model, that helps the user to write a text, by predicting what words could be put into a blank space. 

## Information:

This was built on PyTorch. The training data I used was publicly available. It contained HTML tags and other unwanted stuff, hence the data had to be formatted and split into the required files, which can be found in the "data" folder. 


# How to use this:

Install Git, then:	

``` git clone https://github.com/chrisAS12/specseminars-2021-mi ```

Afterwards, install all the requirements which are mentioned in the file "requiremenets.txt" using pip. 

### Then you can run ``` main.py ``` and see how it works!

### Example:
``` 
Ievadiet tekstu un simbolu, kuru vēlāk mainīsiet: viens * divi
Simbols, kuru mainīt: * 
viens * divi
Šajā tekstā mēs izmainīsim * uz šo:
  viens divi 
Viss minējums:  [{'sequence': 'viens, divi', 'score': 0.005533328279852867, 'token': 16, 'token_str': ','}, {'sequence': 'viens divi', 'score': 0.0010569952428340912, 'token': 2, 'token_str': '</s>'}, {'sequence': 'viensas divi', 'score': 0.0010153661714866757, 'token': 266, 'token_str': 'as'}, {'sequence': 'viensa divi', 'score': 0.0009730682941153646, 'token': 69, 'token_str': 'a'}, {'sequence': 'viensT divi', 'score': 0.0008624795009382069, 'token': 56, 'token_str': 'T'}] 
```
## Conclusion
After finishing this project, which took me a lot of hours, I see a lot of space for improvement. The input data could have been formatted much better. The model wasn't great, since it needs a lot of training input and computer resources that can be allocated. During the programming of this model, I learned a lot of information about NLP and how machine learning works in general, especially BERT and RoBERTa. This was great! 

