import gradio as gr
import string
import random
import os
from PIL import Image
import json


def upload_func(img, label):
    def not_empty(s):
        return s and s.strip()

    doc_list = label.split("\n")
    doc_list = list(filter(not_empty, doc_list))
    doc_df = [[index, len(doc), doc] for index, doc in enumerate(doc_list)]
    return [[img], doc_df]


def gen_df(model_name, img, max_token, list_num):
    def generate_random_text(length):
        characters = string.ascii_letters + string.digits + string.punctuation + " "
        return "".join(random.choice(characters) for _ in range(length))

    return [[model_name, generate_random_text(max_token)] for _ in range(list_num)]


def get_test_gallery(*args, **kwargs):
    default_img_folder = "config/image/1"
    img_list = os.listdir(default_img_folder)
    ret_list = [Image.open(os.path.join(default_img_folder, img)) for img in img_list]
    return ret_list


def get_origin_dataset(dataset_path, dataset_name, dataset_split, top_k):
    with open(os.path.join(dataset_path, dataset_name), "r", encoding="utf-8") as f:
        data = json.load(f)[dataset_split * 10 : (dataset_split + 1) * 10]
    df = []
    for item in data[: top_k - 1]:
        image_path = os.path.join(
            "data/image",
            dataset_name.split(".")[0],
            f"{item['image_id']:012d}.jpg",
        )
        image_obj = Image.open(image_path)
        text = item["old_caption"]
        df.append([image_obj, text])
    return df


def get_enhance_dataset(dataset_path, dataset_name, dataset_split, top_k):
    with open(os.path.join(dataset_path, dataset_name), "r", encoding="utf-8") as f:
        data = json.load(f)[dataset_split * 10 : (dataset_split + 1) * 10]
    df = []
    for item in data[: top_k - 1]:
        image_path = os.path.join(
            "data/image",
            dataset_name.split(".")[0],
            f"{item['image_id']:012d}.jpg",
        )
        image_obj = Image.open(image_path)
        text = item["caption"]
        df.append([image_obj, text])
    return df
