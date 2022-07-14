# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: xiaofei_sun@shannonai.com
@time: 2022/05/07
@desc: 这只飞很懒
"""
import json
import os
from collections import Counter
from typing import List

import numpy
import torch
from db import PairRecord, get_dataset_engine
from fairseq import utils
from fairseq.data import Dictionary, LanguagePairDataset, FairseqDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from sqlalchemy.orm import Session

engine = get_dataset_engine()


class MyDBDataset(FairseqDataset):
    """
    从json文件中读取数据
    """

    def __init__(self, id_list: List[int], length_list: List[int], src_or_tgt):
        self.id_list = id_list
        self.sizes = length_list
        self.size = len(self.id_list)
        # self.engine = get_dataset_engine()
        assert src_or_tgt in ["src", "tgt"]
        self.src_or_tgt = src_or_tgt
        self.index_list = []
        self.index_to_index = dict()
        self.cache_size = 1000
        self.db_select_size = 1000
        self.cache = dict()

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    def __getitem__(self, i):
        self.check_index(i)
        pair_id = self.id_list[i]
        if pair_id not in self.cache:
            if i in self.index_to_index:
                idx = self.index_to_index[i]
                to_cache_id_list = [self.id_list[i] for i in self.index_list[idx:idx + self.cache_size]]
            else:
                to_cache_id_list = [pair_id]
            self.cache = dict()
            with Session(engine) as session:
                pair_list: List[PairRecord] = session.query(PairRecord).where(PairRecord.id.in_(to_cache_id_list)).all()
            self.cache = {pair.id: pair for pair in pair_list}
        pair = self.cache[pair_id]
        if self.src_or_tgt == "src":
            return torch.from_numpy(numpy.array(pair.source_ids))
        else:
            return torch.from_numpy(numpy.array(pair.target_ids))

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        self.index_list = indices
        self.index_to_index = dict(
            zip(
                indices,
                range(len(indices))
            )
        )


@register_task("my_db_translation")
class MyDBTranslationTask(TranslationTask):
    @staticmethod
    def build_dictionary(file_name, split):
        with open(file_name) as input_file:
            data = json.load(input_file)
        dict = Dictionary()
        counter = Counter()
        for datum in data:
            counter.update(datum[split])
            counter.update([dict.eos_word])
        for w, c in sorted(counter.items()):
            dict.add_symbol(w, c)
        dict.finalize()
        return dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) == 1
        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], f"dict_src.txt")
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], f"dict_tgt.txt")
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        print("my load_dataset")
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        path = os.path.join(paths[0], f"{split}_id_list.json")
        print("path", path)
        with open(path) as input_file:
            id_length_list = json.load(input_file)
        id_list = [datum[0] for datum in id_length_list]
        src_size_list = [datum[1] for datum in id_length_list]
        tgt_size_list = [datum[2] for datum in id_length_list]
        src = MyDBDataset(id_list, src_size_list, "src")
        tgt = MyDBDataset(id_list, tgt_size_list, "tgt")
        self.datasets[split] = LanguagePairDataset(
            src=src,
            src_sizes=src.sizes,
            src_dict=self.src_dict,
            tgt=tgt,
            tgt_sizes=tgt.sizes,
            tgt_dict=self.tgt_dict,
        )
