# # import tabula

# # data = tabula.read_pdf("mrp.pdf", pages="all")
# # print(data)
# import re
# from pdfminer.high_level import extract_pages, extract_text


# # for page_layout in extract_pages("mrp.pdf"):
# #     for element in page_layout:
# #         print(element)
# text = extract_text("mrp.pdf")
# print(text)

# pattern = re.compile(r"[a-zA-Z]+,{1}\s{1}")
# matches = pattern.findall(text)
# print(matches)
# import tabula
# data = tabula.read_pdf("mrp.pdf", pages="all")
# print(data)
# from pdf2docx import converter, parse
# from converter import convert


# pdf_file = "mrp.pdf"
# word_file = "demo.docx"

# cv = converter(pdf_file)
# cv.convert(word_file, start=0, end=None)
# cv.close()
# from spacy.training import Example
import json
import random
# from spacy.util import minibatch, compounding
# from spacy.training import GoldParse
# import spacy
import pdfplumber
import csv


with pdfplumber.open("Jaquar.pdf") as pdf_file:

    for page in pdf_file.pages:

        text = page.extract_text()

        lines = text.split("\n")

        with open("output1.txt", "w", newline='') as csv_file:
            writer = csv.writer(csv_file)

            for line in lines:

                cells = line.split(" ")

                writer.writerow(cells)
# with open('output.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         print(row)
# import pytesseract
# from PIL import Image

# pdf = Image.open('mrp.pdf')
# text = pytesseract.image_to_string(pdf)
# print(text)


# Load the model
# nlp = spacy.blank('en')

# # Define the training data as a list of dictionaries in spaCy's expected JSON format
# TRAIN_DATA = [
#     {"text": "This is an example sentence.", "entities": [(0, 4, "LABEL")]},
#     {"text": "Another example sentence.", "entities": [(10, 17, "LABEL")]},
#     # add more training examples here
# ]

# # Convert the training data to spaCy's Example format
# examples = []
# for data in TRAIN_DATA:
#     text = data["text"]
#     entities = data["entities"]
#     examples.append(Example.from_dict(
#         nlp.make_doc(text), {"entities": entities}))

# # Train the model
# nlp.begin_training()
# for i in range(10):
#     random.shuffle(examples)
#     for batch in minibatch(examples, size=2):
#         nlp.update(batch, drop=0.5)

# # Test the model on some example text
# doc = nlp("This is a test sentence.")
# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)
