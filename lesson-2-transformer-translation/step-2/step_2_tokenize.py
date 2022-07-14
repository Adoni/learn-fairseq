# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/05/07
@desc: 这只飞很懒
"""
import json


def tokenize():
    base_dir = "./easy-dataset"
    for split in ["train", "valid", "test"]:
        input_file_path = f"{base_dir}/raw_data/{split}.in"
        with open(input_file_path) as input_file:
            source_tokenized_data = [list(line.strip()) for line in input_file]
        input_file_path = f"{base_dir}/raw_data/{split}.out"
        with open(input_file_path) as input_file:
            target_tokenized_data = [list(line.strip()) for line in input_file]
        tokenized_data = []
        for source, target in zip(source_tokenized_data, target_tokenized_data):
            tokenized_data.append({
                "src": source,
                "tgt": target
            })
        output_file_path = f"{base_dir}/json_data/{split}.json"
        with open(output_file_path, "w") as output_file:
            json.dump(tokenized_data, output_file, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    tokenize()
