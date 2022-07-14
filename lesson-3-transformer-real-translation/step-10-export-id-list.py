# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/06/19
@desc: 这只飞很懒
"""

import json
from typing import List

from sqlalchemy import select, func
from sqlalchemy.orm import Session
from tqdm import tqdm

from db import PairRecord, get_dataset_engine


def run(split):
    engine = get_dataset_engine()
    with Session(engine) as session:
        min_id = session.scalar(select(func.min(PairRecord.id)))
        max_id = session.scalar(select(func.max(PairRecord.id)))
    batch_size = 1000
    id_list = []
    for begin_id in tqdm(range(min_id, max_id + 1, batch_size), ncols=120, desc=split):
        with Session(engine) as session:
            pair_list: List[PairRecord] = session.scalars(
                select(PairRecord)
                    .where(PairRecord.id >= begin_id)
                    .where(PairRecord.id < begin_id + batch_size)
            ).all()
        for pair in pair_list:
            if pair.split != split:
                continue
            id_list.append([pair.id, len(pair.source_ids), len(pair.target_ids)])
    with open(f"./iwslt14/preprocessed/{split}_id_list.json", "w") as output_file:
        json.dump(id_list, output_file, indent=2)


if __name__ == '__main__':
    run("train")
    run("valid")
    run("test")
