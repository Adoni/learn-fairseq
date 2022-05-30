# Lesson-2-Transformer翻译

进入正题，我们还是依据阿拉伯数字到中文数字描述的转换，完成以下目标：

* 自行完成前处理，完成tokenize和bpe，以可视化字符存储词表和数据
* 自定义task，能够完成load数据的动作
* 使用Transformer结构
* 完成相应demo

## Step-1 准备数据

这次的raw_data和之前一样，但是这次我们不进行分词了，放在第二个脚本里做。

raw_data数据目录如下：

```
./easy-dataset/raw_data
├── test.in
├── test.out
├── train.in
├── train.out
├── valid.in
└── valid.out
```

数据格式如下：

train.in

```
819209
294923
557072
```

train.out

```
八十一万九千二百零九
二十九万四千九百二十三
五十五万七千零七十二
```

预处理程序，我们直接裸写tokenize和转id的过程，这里顺道要说一下我们是通过新定义的task完成的。

tokenize脚本见`step-2/step_2_tokenize.json`，完事儿后，目录长这样：

```
easy-dataset/json_data
├── test.json
├── train.json
└── valid.json
```

数据长这样：

```json
[
  {
    "src": [
      "6",
      "5",
      "5",
      "3",
      "8",
      "9"
    ],
    "tgt": [
      "六",
      "十",
      "五",
      "万",
      "五",
      "千",
      "三",
      "百",
      "八",
      "十",
      "九"
    ]
  }
]
```

接下来需要增加token id，详见`step-2/step_3_add_id.py`。
这里我们其实就是自己主动调用了`MyTranslationTask.build_dictionary`这个函数构造dictionary，然后去调用`encode_line`来获取id。

## Step-3 构造Encoder和Decoder

这一次，我们不再完整地实现Transformer，而是尝试继承Transformer，然后做一些不涉及核心代码的小修改，这样做的理由是：

1. 很多情况下，我们不需要大改Transformer
2. 尽量以集成的方式完成修改会令代码量少一些
3. 这不是我们这次的重点

代码见：`my_fairseq_module/model`

## Step-4 注册model和architecture

代码见：`my_fairseq_module/model/__init__.py`

## Step-5 Train

见`step-5-train.sh`

原始代码见`from fairseq_cli.train import cli_main`

特殊情况下我们可以拷贝这里的代码进行修改，这就是后话了

这里提一个需要注意的点：transformer训练一定要加warmup，一开始没加warmup发现总是不收敛，原因待查证，咨询过昱先，他表示的确有这样的trick存在

## Step-6 Evaluate

这里我们使用fairseq-generate的命令，直接在已经处理好的目录下进行predict，看一下结果

见`step-6-evaluate.sh`

## Step-7 Python版demo

运行代码之前，我们需要将bpe用到的字典和checkpoint放到一起

```
cp easy-dataset/preprocessed/dict* ./checkpoints
```

然后直接运行`python step-7-demo.py`
