# import csv
# from pdfminer.high_level import extract_text

# # Extract the text from the PDF file
# page_text = extract_text('Jaquar.pdf')

# # Split the text into lines and write them to a CSV file
# with open('file.csv', mode='w', encoding='utf-8', newline='') as file:
#     csv_writer = csv.writer(file)

#     for line in page_text.split('\n'):
#         csv_writer.writerow(['Header 1', 'Header 2'])
#         csv_writer.writerow(['Data 1', 'Data 2'])

# # Write the text to a text file
# with open('example.txt', 'w') as txt_file:
#     txt_file.write(page_text)
# import csv
# from pdfminer.high_level import extract_text

# # Extract the text from the PDF file
# page_text = extract_text('Jaquar.pdf')

# # Split the text into lines and write them to a CSV file
# with open('file.csv', mode='w', encoding='utf-8', newline='') as file:
#     csv_writer = csv.writer(file)

#     for line in page_text.split('\n'):
#         csv_writer.writerow(['Header 1', 'Header 2'])
#         csv_writer.writerow(['Data 1', 'Data 2'])

# # Write the text to a text file
# with open('example.txt', 'w', encoding='utf-8') as txt_file:
#     txt_file.write(page_text)
import pandas as pd
import pdfminer.high_level

# Extract the text from the PDF file
page_text = pdfminer.high_level.extract_text('Jaquar.pdf')

# Split the text into rows and columns
rows = page_text.split('\n')
columns = [row.split('\t') for row in rows]

# Create a pandas DataFrame from the columns
df = pd.DataFrame(columns)

# Save the DataFrame as an Excel file
df.to_excel('output.xlsx', index=False)
