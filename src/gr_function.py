import gradio as gr
import string
import random
import os
from PIL import Image


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
