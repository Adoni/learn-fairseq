# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/06/19
@desc: 这只飞很懒
"""

import json

from sqlalchemy.orm import Session
from tqdm import tqdm

from db import PairRecord, get_dataset_engine


def run():
    engine = get_dataset_engine()
    for split in ["train", "valid", "test"]:
        path = f"./iwslt14/preprocessed/{split}.json"
        print("path", path)
        with open(path) as input_file:
            data = json.load(input_file)
        records = []
        for datum in tqdm(data):
            src = " ".join(datum["src"])[:900]
            tgt = " ".join(datum["tgt"])[:900]
            records.append(PairRecord(
                source=src,
                target=tgt,
                source_ids=datum["src_ids"],
                target_ids=datum["tgt_ids"],
                split=split
            ))
        batch_size = 1000
        for begin_id in tqdm(range(0, len(records), batch_size)):
            with Session(engine) as session:
                session.add_all(records[begin_id:begin_id + batch_size])
                session.commit()


if __name__ == '__main__':
    run()
