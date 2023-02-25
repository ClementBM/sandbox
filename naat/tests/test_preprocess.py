from naat.data import CORPUS_PATH, get_file_extensions, preprocess_dataset


def test_preprocess_get_extensions():
    extensions = get_file_extensions(CORPUS_PATH)
    assert len(extensions) > 0
