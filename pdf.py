from PyPDF2 import PdfReader

# doc = PdfReader("mrp.pdf")
# pages = len(doc.pages)
reader = PdfReader("mrp.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()
print(text)
search = 'H947'

# list_pages = []
# print(pages)
# for i in range(page):
#     current_page = number_of_pages(i)
#     text = current_page.extractText()
#     print(text)
