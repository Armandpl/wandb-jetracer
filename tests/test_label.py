import os

from label import label_img


def test_label_img(fs):
    path = "unique-id.jpg"
    fs.create_file(path)

    path = label_img(10, 20, path)

    assert os.path.exists("10_20_unique-id.jpg")

    label_img(30, 40, path)

    assert os.path.exists("30_40_unique-id.jpg")
