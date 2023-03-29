import re
import fitz  # install using: pip install PyMuPDF

# Open the PDF file

with fitz.open("Jaquar.pdf") as doc:
    text = ""
    for page in doc:
        text += page.get_text()

print(text)

# Save the extracted text to a text file
with open("output7.txt", "w", encoding="utf-8") as file:
    file.write(text)

# Read the file
with open("output7.txt", "r") as file:
    lines = file.readlines()

# Find the line with Model No: ABC456
for i, line in enumerate(lines):
    if re.search("QQP-CHR-7001BPM", line):
        # Modify the price
        lines[i] = re.sub("\$200", "$250", line)

# Write the modified file
with open("output8.txt", "w") as file:
    file.writelines(lines)
