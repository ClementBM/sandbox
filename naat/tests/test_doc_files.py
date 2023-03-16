from naat_data import DOC_PESTICIDES_PATH, DOC_PROTECTION_PATH
from naat_core.data import preprocess_doc_file


def test_doc_preprocessing_1():
    document = preprocess_doc_file(DOC_PESTICIDES_PATH)
    assert len(document) > 0


def test_doc_preprocessing_2():
    document = preprocess_doc_file(DOC_PROTECTION_PATH)
    assert len(document) > 0
