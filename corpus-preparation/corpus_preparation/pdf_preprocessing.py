
from anyio import Path
from pypdf import PdfReader
import pdf2image
import pytesseract as pt

# https://pypdf.readthedocs.io/en/latest/user/extract-text.html
def preprocess_pdf_file(file_path):
    parts = []

    def visitor_body(text, cm, tm, font_dict, font_size):
        y = tm[5]
        if y > 50 and y < 720:
            yield text

    reader = PdfReader(file_path)
    for page in reader.pages:
        parts += list(page.extract_text(visitor_text=visitor_body))

    if len(parts) == 0:
        return pdf_image_reader(file_path)
    return parts


def pdf_image_reader(file_path: Path, image_folder_name="images"):
    """
    https://github.com/YiLi225/Import_PDF_Word_Python/blob/master/Pdf_Word_Reader.py
    error msg: TesseractNotFoundError: tesseract is not installed
    Download and install the Tesseract OCR: sudo apt install tesseract-ocr
    """
    ### 1) Initiate to store the converted images
    pages = pdf2image.convert_from_path(pdf_path=file_path, dpi=200, size=(1654, 2340))

    ### 2) Make the dir to store the images (if not already exists)
    saved_image_name = file_path.parent / image_folder_name / file_path.name
    saved_image_name.mkdir(exist_ok=True, parents=True)

    ### 3) Save each page as one image
    for i in range(len(pages)):
        pages[i].save(saved_image_name / f"{i}.jpg")

    ### 4) Extract the content by converting images to a list of strings
    content = ""
    for i in range(len(pages)):
        content += pt.image_to_string(pages[i])

    return content
