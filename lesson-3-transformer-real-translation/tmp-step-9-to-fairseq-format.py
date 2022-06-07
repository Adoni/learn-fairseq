# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: xiaofei_sun@shannonai.com
@time: 2022/05/31
@desc: 这只飞很懒
"""
import json


def run():
    for split in ["train", "valid", "test"]:
        with open(f"./iwslt14/bpe/{split}.json") as input_file:
            data = json.load(input_file)
        src = [" ".join(datum["src"]) for datum in data]
        tgt = [" ".join(datum["tgt"]) for datum in data]
        with open(f"./iwslt14/tmp-bpe/{split}.src", "w") as output_file:
            output_file.write("\n".join(src))
        with open(f"./iwslt14/tmp-bpe/{split}.tgt", "w") as output_file:
            output_file.write("\n".join(tgt))
        ad = "adsfasdfasdfadsf"


if __name__ == '__main__':
    run()
