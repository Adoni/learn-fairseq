# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: xiaofei_sun@shannonai.com
@time: 2022/06/19
@desc: 这只飞很懒
"""

import json

from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy import create_engine
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import QueuePool

Base = declarative_base()


def get_dataset_engine():
    engine = create_engine(
        "postgresql+psycopg2://sunxiaofei:123atPitaya@172.32.0.16/datasets",
        json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
        pool_size=10, poolclass=QueuePool
    )
    return engine


class PairRecord(Base):
    __tablename__ = "learn_fairseq_lesson_3_pair_table"

    id = Column(Integer, primary_key=True, doc="id", comment="id")
    source = Column(String(1000), nullable=False, doc="src句子", comment="src句子")
    target = Column(String(1000), nullable=True, doc="tgt句子", comment="tgt句子")
    source_ids = Column(MutableList.as_mutable(JSON), nullable=False, doc="src的id列表", comment="src的id列表")
    target_ids = Column(MutableList.as_mutable(JSON), nullable=True, doc="tgt的id列表", comment="tgt的id列表")
    split = Column(String(100), nullable=True, doc="是train、valid还是test", comment="是train、valid还是test")


if __name__ == '__main__':
    PairRecord.metadata.create_all(get_dataset_engine())
