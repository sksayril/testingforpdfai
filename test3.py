import re
import fitz

with fitz.open("Jaquar.pdf") as doc:
    text = ""
    for page in doc:
        text += page.get_text()

print(text)


with open("output7.txt", "w", encoding="utf-8") as file:
    file.write(text)


with open("output7.txt", "r") as file:
    lines = file.readlines()


for i, line in enumerate(lines):
    if re.search("QQP-CHR-7001BPM", line):

        lines[i] = re.sub("\$200", "$250", line)


with open("output8.txt", "w") as file:
    file.writelines(lines)
