# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: xiaofei_sun@shannonai.com
@time: 2022/05/03
@desc: 这只飞很懒
"""

from fairseq.hub_utils import GeneratorHubInterface

from my_fairseq_module.model.simple_lstm_model import SimpleLSTMModel


def run():
    model: GeneratorHubInterface = SimpleLSTMModel.from_pretrained(
        model_name_or_path="checkpoints",
        checkpoint_file="checkpoint_best.pt",
        tokenizer=None,
        # user_dir="./my_fairseq_module/"
    )
    print(type(model))

    while True:
        sentence = input('\nInput: ')
        translation = model.translate([" ".join(sentence)])
        print(translation)


if __name__ == '__main__':
    run()
