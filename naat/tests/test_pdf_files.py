from naat_data import PDF_PACIFIC_PATH, PDF_JONAH_PATH
from naat_core.data import preprocess_pdf_file
import re


def test_pdf_preprocessing_1():
    document = preprocess_pdf_file(PDF_PACIFIC_PATH)
    document_out = re.sub(r"\n\d+", "", document)
    assert len(document_out) > 0


def test_pdf_preprocessing_2():
    document = preprocess_pdf_file(PDF_JONAH_PATH)
    assert len(document) > 0
