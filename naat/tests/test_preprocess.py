from naat.data import ROOT_PATH, get_file_extensions, preprocess_dataset


def test_preprocess_get_extensions():
    extensions = get_file_extensions(ROOT_PATH)
    assert len(extensions) > 0
