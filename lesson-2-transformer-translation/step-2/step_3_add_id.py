# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: xiaofei_sun@shannonai.com
@time: 2022/05/06
@desc: 这只飞很懒
"""
import json

from fairseq.data import Dictionary
from my_fairseq_module.tasks.my_translation_task import MyTranslationTask


def save_dict(src_dict: Dictionary, tgt_dict: Dictionary, base_dir):
    src_dict.save(f"{base_dir}/preprocessed/dict_src.txt")
    tgt_dict.save(f"{base_dir}/preprocessed/dict_tgt.txt")


def run():
    base_dir = "./easy-dataset"
    src_dict = MyTranslationTask.build_dictionary(f"{base_dir}/json_data/train.json", "src")
    tgt_dict = MyTranslationTask.build_dictionary(f"{base_dir}/json_data/train.json", "tgt")
    save_dict(src_dict, tgt_dict, base_dir)

    for split in ["train", "valid", "test"]:
        input_file_path = f"{base_dir}/json_data/{split}.json"
        with open(input_file_path) as input_file:
            data = json.load(input_file)
        for datum in data:
            datum["src_ids"] = src_dict.encode_line(
                datum["src"], line_tokenizer=lambda x: x, add_if_not_exist=False
            ).tolist()
            datum["tgt_ids"] = tgt_dict.encode_line(
                datum["tgt"], line_tokenizer=lambda x: x, add_if_not_exist=False
            ).tolist()
        output_file_path = f"{base_dir}/preprocessed/{split}.json"
        with open(output_file_path, "w") as output_file:
            json.dump(data, output_file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    run()
