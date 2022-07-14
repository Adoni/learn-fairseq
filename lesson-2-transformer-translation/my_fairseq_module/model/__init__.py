# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/05/09
@desc: 这只飞很懒
"""

from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture

# noinspection PyUnresolvedReferences
from .my_transformer_model import MyTransformerModel


@register_model_architecture("my_transformer", "my_small_transformer")
def build_small_transformer(args):
    # 这里我们先调用原本transformer的architecture
    base_architecture(args)
    # 然后我们修改层数，当然这个参数也可以通过直接外部参数传入，我们这边只是演示一下修改
    args.encoder_layers = 6
