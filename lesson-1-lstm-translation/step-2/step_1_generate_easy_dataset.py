# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/05/02
@desc: 这只飞很懒
"""
import cn2an
import numpy


def run():
    number_list = numpy.random.randint(100000, 1000000 - 1, 10000)
    number_list = list(set(number_list))
    print(len(number_list))
    dataset = {
        "train": [[], []],
        "valid": [[], []],
        "test": [[], []],
    }
    for num in number_list:
        chinese_num = cn2an.an2cn(str(num))
        p = numpy.random.random()
        if p < 0.8:
            dataset_type = "train"
        elif p < 0.9:
            dataset_type = "valid"
        else:
            assert p <= 1
            dataset_type = "test"
        dataset[dataset_type][0].append(" ".join(list(str(num))))
        dataset[dataset_type][1].append(" ".join(list(chinese_num)))
    for dataset_type in dataset:
        with open(f"./easy-dataset/raw_data/{dataset_type}.in", "w") as output_file:
            output_file.write("\n".join(dataset[dataset_type][0]))
        with open(f"./easy-dataset/raw_data/{dataset_type}.out", "w") as output_file:
            output_file.write("\n".join(dataset[dataset_type][1]))


if __name__ == '__main__':
    run()
