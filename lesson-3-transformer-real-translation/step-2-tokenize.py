# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/05/17
@desc: 这只飞很懒
"""
import json

from sacremoses import MosesTokenizer
from tqdm import tqdm


def run():
    print("Tokenize")
    de_tokenizer = MosesTokenizer(lang="de")
    en_tokenizer = MosesTokenizer(lang="en")
    for split in ["train", "valid", "test"]:
        with open(f"iwslt14.tokenized.de-en/tmp/{split}.de") as input_file:
            source_list = [line.strip() for line in input_file]
        with open(f"iwslt14.tokenized.de-en/tmp/{split}.en") as input_file:
            target_list = [line.strip() for line in input_file]
        data = []
        for s, t in tqdm(zip(source_list, target_list), total=len(source_list), desc=f"Tokenize {split}"):
            data.append({
                "src": de_tokenizer.tokenize(s, return_str=False),
                "tgt": en_tokenizer.tokenize(t, return_str=False),
            })
        with open(f"./iwslt14/tokenized/{split}.json", "w") as output_file:
            json.dump(data, output_file, ensure_ascii=False, indent=2)
        if split == "train":
            src = [" ".join(datum["src"]) for datum in data]
            tgt = [" ".join(datum["tgt"]) for datum in data]
            with open(f"./iwslt14/tokenized/train.for_bpe.src", "w") as output_file:
                output_file.write("\n".join(src))
            with open(f"./iwslt14/tokenized/train.for_bpe.tgt", "w") as output_file:
                output_file.write("\n".join(tgt))


if __name__ == '__main__':
    run()
