import gzip
import json

from PIL import Image


def write_json(data, path):
    with open(path, "w") as file:
        file.write(json.dumps(data))


def load_json(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data


def write_txt(data, path):
    with open(path, "w") as file:
        file.write("\n".join(data))


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save(file_name)


def load_dataset(path):
    with gzip.open(path, "rb") as file:
        data = json.loads(file.read(), encoding="utf-8")
    return data