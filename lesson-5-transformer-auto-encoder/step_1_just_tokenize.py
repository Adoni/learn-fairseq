# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/06/06
@desc: 这只飞很懒
"""
import jieba


def run():
    with open("") as input_file:
        for line in input_file:
            " ".join(jieba.cut(line))


if __name__ == '__main__':
    run()
