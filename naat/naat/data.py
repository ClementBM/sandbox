import os
from pathlib import Path

from tqdm import tqdm
from fasttext import load_model
from tqdm import tqdm
import numpy as np
from docx import Document
from docx.oxml.shared import qn
from docx.text.paragraph import Paragraph, Run
from pypdf import PdfReader
from validator_collection import checkers

ROOT = Path(__file__)

ROOT_PATH = os.environ.get("ROOT_PATH")
FOLDER_NAME = "Lettre_interne_justice_climatique"

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
        return None
    return "".join(parts)


def list_files(startpath):
    indent_count = 4
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * indent_count * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * indent_count * (level + 1)
        for f in files:
            print(f"{subindent}{f}")


def get_file_extensions(startpath):
    file_extensions = set()
    for root, dirs, files in os.walk(startpath):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            file_extensions.add(file_extension)

    return file_extensions


def preprocess_dataset(startpath, copy_to=None):
    processing_methods = {
        ".pdf": preprocess_pdf_file,
        ".docx": preprocess_docx_file,
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
                    continue

                language = "fr"
                if is_english([formated_text[:1000].replace("\n", " ")]):
                    language = "en"

                base_name = os.path.basename(startpath)
                new_file_path = str(file_abs_path).replace(base_name, copy_to)
                new_file_path = Path(
                    new_file_path[: -len(file_extension)] + f"_{language}.txt"
                )

                if not new_file_path.parent.exists():
                    new_file_path.parent.mkdir(parents=True)
                with open(new_file_path, "w") as writer:
                    writer.write(formated_text)

            else:
                print(file_abs_path)


def is_english(texts: list):
    path_to_pretrained_model = ROOT.parent.parent / "fasttext" / "lid.176.ftz"
    fmodel = load_model(str(path_to_pretrained_model))
    return np.array(sum(fmodel.predict(texts)[0], [])) == "__label__en"
