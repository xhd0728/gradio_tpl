import os
import json


def parse_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def parse_jsonl(jsonl_path):
    with open(jsonl_path, "r") as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return data
