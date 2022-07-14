# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/06/22
@desc: 这只飞很懒
"""
import json
from typing import List

import numpy.random
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from db import PairRecord, get_dataset_engine


def run():
    with open("iwslt14/preprocessed/train_id_list.json") as input_file:
        data = json.load(input_file)
    data = [datum[0] for datum in data]
    numpy.random.shuffle(data)
    print(data[:10])
    batch_size = 1
    engine = get_dataset_engine()
    for begin in tqdm(range(0, len(data), batch_size)):
        id_list = data[begin:begin + batch_size]
        # id_list = [62851]
        with Session(engine) as session:
            pair_list: List[PairRecord] = session.scalars(
                select(PairRecord).where(PairRecord.id.in_(id_list))
            ).all()
            assert len(pair_list) > 0


if __name__ == '__main__':
    run()
