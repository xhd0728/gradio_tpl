import os
from PIL import Image
import json
from utils import parse_json
import numpy as np

np.set_printoptions(precision=4, suppress=False)


def split_dataset(split):
    select = {"train": 0, "val": 1, "test": 2}
    assert split in select.keys()
    return select[split]


def get_dataset_image(dataset_path, dataset_name, dataset_split, top_k):
    split_chunk = split_dataset(dataset_split)
    with open(
        os.path.join(dataset_path, f"{dataset_name}.json"), "r", encoding="utf-8"
    ) as f:
        data = json.load(f)[split_chunk * 10 : (split_chunk + 1) * 10]
    img_list = []
    for item in data[:top_k]:
        image_id = item["image_id"]
        image_path = os.path.join("data/image", dataset_name, f"{image_id:012d}.jpg")
        image_obj = Image.open(image_path)
        img_list.append(image_obj)
    return img_list


def get_origin_dataset(dataset_path, dataset_name, dataset_split, top_k):
    split_chunk = split_dataset(dataset_split)
    with open(
        os.path.join(dataset_path, f"{dataset_name}.json"), "r", encoding="utf-8"
    ) as f:
        data = json.load(f)[split_chunk * 10 : (split_chunk + 1) * 10]
    df = []
    for index, item in enumerate(data[:top_k]):
        text = item["old_caption"]
        df.append([index, len(text), text])
    return df


def get_enhance_dataset(dataset_path, dataset_name, dataset_split, top_k):
    split_chunk = split_dataset(dataset_split)
    with open(
        os.path.join(dataset_path, f"{dataset_name}.json"), "r", encoding="utf-8"
    ) as f:
        data = json.load(f)[split_chunk * 10 : (split_chunk + 1) * 10]
    df = []
    for index, item in enumerate(data[:top_k]):
        text = item["caption"]
        df.append([index, len(text), text])
    return df


def fill_case_data(test_case):
    case_data = parse_json(f"data/embedding/{test_case}.json")
    image_path = os.path.join("data/image/webqa", f'{case_data["image_id"]:012d}.jpg')
    image_obj = Image.open(image_path)
    text_obj = case_data["caption"]
    return [image_obj, text_obj]


def get_vision_embedding(vision_embedding_model, test_case):
    case_data = parse_json(f"data/embedding/{test_case}.json")
    ret_df = []
    for model in vision_embedding_model:
        vector = case_data["vision_embedding_model"][model]
        ret_df.append([model, vector])
    return ret_df


def get_text_embedding(text_embedding_model, test_case):
    case_data = parse_json(f"data/embedding/{test_case}.json")
    ret_df = []
    for model in text_embedding_model:
        vector = case_data["text_embedding_model"][model]
        ret_df.append([model, vector])
    return ret_df


def get_combined_embedding(vision_embedding_model, text_embedding_model, test_case):
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    case_data = parse_json(f"data/embedding/{test_case}.json")
    ret_df = []
    for model_1 in vision_embedding_model:
        vector_1 = case_data["vision_embedding_model"][model_1]
        for model_2 in text_embedding_model:
            vector_2 = case_data["text_embedding_model"][model_2]
            max_dim = max(len(vector_1), len(vector_2))
            vector_1 = np.pad(vector_1, (0, max_dim - len(vector_1)), constant_values=0)
            vector_2 = np.pad(vector_2, (0, max_dim - len(vector_2)), constant_values=0)
            vector_3 = np.multiply(softmax(vector_1), softmax(vector_2)).tolist()
            ret_df.append([model_1, model_2, vector_3])
    return ret_df


def fill_retrieval_case_data(test_case):
    case_data = parse_json(f"data/retrieval/{test_case}.json")
    image_path = os.path.join("data/image/webqa", f'{case_data["image_id"]:012d}.jpg')
    image_obj = Image.open(image_path)
    text_obj = case_data["caption"]
    return [image_obj, text_obj]
