# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/05/09
@desc: 这只飞很懒
"""
from fairseq.models.transformer import TransformerEncoder


class MyTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super(MyTransformerEncoder, self).__init__(args, dictionary, embed_tokens)
        print("Here is my encoder")
