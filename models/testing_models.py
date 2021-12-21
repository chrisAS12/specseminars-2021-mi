from transformers import pipeline
pipeClassification = pipeline("text-classification")
print(pipeClassification("Checking how positive this thingy ma jig is. :D"))