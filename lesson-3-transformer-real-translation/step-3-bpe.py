# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/05/17
@desc: 这只飞很懒
"""
import json

import fastBPE
from tqdm import tqdm


def run():
    src_bpe = fastBPE.fastBPE("./codes.bpe", "./vocab.src.40000")
    tgt_bpe = fastBPE.fastBPE("./codes.bpe", "./vocab.tgt.40000")

    for split in ["train", "valid", "test"]:
        with open(f"./iwslt14/tokenized/{split}.json") as output_file:
            data = json.load(output_file)
        bpe_data = []
        for datum in tqdm(data, desc=split):
            bpe_data.append(
                {
                    "src": src_bpe.apply([" ".join(datum["src"])])[0].split(" "),
                    "tgt": tgt_bpe.apply([" ".join(datum["tgt"])])[0].split(" "),
                }
            )
        with open(f"./iwslt14/bpe/{split}.json", "w") as output_file:
            json.dump(bpe_data, output_file, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    run()
