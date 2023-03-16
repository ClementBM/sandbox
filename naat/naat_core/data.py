import os
from pathlib import Path

import subprocess
from tqdm import tqdm
from docx import Document
from docx.oxml.shared import qn
from docx.text.paragraph import Paragraph, Run
from pypdf import PdfReader
from validator_collection import checkers
import gcld3
import pandas as pd
import pytesseract as pt
import pdf2image

ROOT = Path(__file__)

CORPUS_PATH = Path(os.environ.get("CORPUS_PATH"))
OUTPUT_PATH = CORPUS_PATH.parent / "test"

# check for file without format
# check for duplicate or quasi-duplicate files


# https://github.com/python-openxml/python-docx/issues/85
def GetParagraphRuns(paragraph):
    def _get(node, parent):
        for child in node:
            if child.tag == qn("w:r"):
                yield Run(child, parent)
            if child.tag == qn("w:hyperlink"):
                yield from _get(child, parent)

    return list(_get(paragraph._element, paragraph))


Paragraph.runs = property(lambda self: GetParagraphRuns(self))


def preprocess_docx_file(file_path):
    document = Document(file_path)

    text = []
    for paragraph in document.paragraphs:
        for run in paragraph.runs:
            if checkers.is_not_empty(run.text):
                text.append(run.text.replace("\xa0", " "))

    if len(text) == 0:
        return None
    return " ".join(text)


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


def get_empty_files(startpath):
    for root, dirs, files in os.walk(startpath):
        for file in files:
            file_path = Path(root) / file
            if os.path.getsize(file_path) == 0:
                print(file_path)


def list_subfolders(startpath, target_level=1):
    subfolders = {}
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        subfolder = os.path.basename(root)
        if level != target_level:
            continue
        if subfolder in subfolders:
            subfolders[subfolder] += 1
        else:
            subfolders[subfolder] = 1

    return subfolders


def list_files(startpath, verbose=False, extension="*"):
    indent_count = 4
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * indent_count * (level)
        if verbose:
            print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * indent_count * (level + 1)
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if extension in ["*", file_extension]:
                yield Path(root) / file
                if verbose:
                    print(f"{subindent}{file}")


def get_file_extensions(startpath):
    file_extensions = dict()
    for root, dirs, files in os.walk(startpath):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension in file_extensions:
                file_extensions[file_extension] += 1
            else:
                file_extensions[file_extension] = 1

    return file_extensions


def preprocess_dataset(startpath, copy_to=None):
    processing_methods = {
        ".pdf": preprocess_pdf_file,
        ".docx": preprocess_docx_file,
        ".doc": preprocess_doc_file,
    }

    for root, subdirs, files in tqdm(os.walk(startpath)):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension not in processing_methods:
                continue

            file_abs_path = Path(root) / file
            if copy_to:
                formated_text = processing_methods[file_extension](file_abs_path)
                if formated_text == None:
                    print(f"Warning: {file_abs_path}")
                    continue

                base_name = os.path.basename(startpath)
                new_file_path = str(file_abs_path).replace(base_name, copy_to)
                new_file_path = Path(new_file_path[: -len(file_extension)] + ".txt")

                if not new_file_path.parent.exists():
                    new_file_path.parent.mkdir(parents=True)
                with open(new_file_path, "w") as writer:
                    writer.write(formated_text)

            else:
                print(file_abs_path)


def detect_language(text):
    """
    sudo apt-get install -y --no-install-recommends g++ protobuf-compiler libprotobuf-dev
    https://github.com/google/cld3
    https://towardsdatascience.com/introduction-to-googles-compact-language-detector-v3-in-python-b6887101ae47
    """
    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=10, max_num_bytes=2147483647)
    results = detector.FindTopNMostFreqLangs(text=text, num_langs=5)
    for result in results:
        if result.language == "und":
            continue
        yield result.language


def get_file_languages(file_path):
    with open(file_path) as file:
        lines = file.readlines()
        languages = sorted(list(detect_language("".join(lines))))

    return "-".join(languages)


def get_files_languages(startpath):
    file_languages = dict()
    for root, dirs, files in os.walk(startpath):
        for file in files:
            file_abs_path = Path(root) / file
            languages = get_file_languages(file_abs_path)
            if languages in file_languages:
                file_languages[languages] += 1
            else:
                file_languages[languages] = 1

    return file_languages


def preprocess_doc_file(file: Path):
    """
    Processing the text paragraph into a list of substrings
    lowriter --convert-to docx *.doc
    By default, stdout and stderr are not captured,
    and those attributes will be None.
    Pass stdout=PIPE and/or stderr=PIPE in order to capture them.
    """
    completed_process = subprocess.run(
        ["lowriter", "--convert-to", "docx", str(file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=file.parent,
    )

    if completed_process.returncode != 0:
        raise Exception(completed_process.stdout, completed_process.stderr)

    filename, file_extension = os.path.splitext(file)

    return preprocess_docx_file(f"{filename}.docx")


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
