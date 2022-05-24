Here to add that I found the download address of the **ctb5** dataset.

https://download.csdn.net/download/weixin_41194129/85448924?spm=1001.2014.3001.5503



## A Unified Model for Joint Chinese Word Segmentation and Dependency Parsing

This is the code for the paper [A Unified Model for Joint Chinese Word Segmentation and Dependency Parsing](https://arxiv.org/abs/1904.04697)

#### Requirements
This project needs the natural language processing python package 
[fastNLP](https://github.com/fastnlp/fastNLP). You can install by
the following command

```bash
pip install fastNLP
```


### Data
Your data should in the format as following
```
1	中国	_	NR	NR	_	4	nn	_	_
2	残疾人	_	NN	NN	_	4	nn	_	_
3	体育	_	NN	NN	_	4	nn	_	_
4	事业	_	NN	NN	_	5	nsubj	_	_
5	方兴未艾	_	VV	VV	_	0	root	_	_

1	新华社	_	NR	NR	_	12	dep	_	_
```
The 1st, 3rd, 6th, 7th(starts from 0) column should be words, pos tags,
 dependency heads and dependency labels, respectively. Empty line separate
  two instances.

You should place your data like the following structure
```
-JointCwsParser
    ...
    -train.py
    -train_bert.py
-data
    -ctb5
        -train.conll
        -dev.conll
        -test.conll
    -ctb7
        -...
    -ctb9
        -...
```
We use code from https://github.com/hankcs/TreebankPreprocessing to convert the original format into the conll format.


### Run the code
You can directly run by
```
python train.py --dataset ctb5
```
or 
```
python train_bert.py --dataset ctb5
```
FastNLP will download pretrained embeddings or BERT weight automatically.

## 训练记录
```bash
python train.py --dataset ctb5
Read cache from caches/ctb5-epo25.pkl.
In total 3 datasets:
        test has 1910 instances.
        train has 16091 instances.
        dev has 803 instances.
In total 7 vocabs:
        char_labels has 13 entries.
        pre_chars has 4296 entries.
        pre_bigrams has 178218 entries.
        pre_trigrams has 467820 entries.
        chars has 3748 entries.
        bigrams has 47556 entries.
        trigrams has 20308 entries.

/opt/conda/lib/python3.6/site-packages/fastNLP/core/field.py:625: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
  return np.array(contents)
input fields after batch(if batch size is 2):
        chars: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22])
        bigrams: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22])
        trigrams: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22])
        char_heads: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22])
        char_labels: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22])
        seq_lens: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2])
        pre_chars: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22])
        pre_bigrams: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22])
        pre_trigrams: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22])
target fields after batch(if batch size is 2):
        char_heads: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22])
        char_labels: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 22])
        seg_targets: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 21])
        seg_masks: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 21])
        seq_lens: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2])
        pun_masks: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 21])
        gold_word_pairs: (1)type:numpy.ndarray (2)dtype:object, (3)shape:(2,)
        gold_label_word_pairs: (1)type:numpy.ndarray (2)dtype:object, (3)shape:(2,)

training epochs started 2022-05-14-00-52-53
Evaluate data in 3.06 seconds!
Evaluate data in 6.99 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.5632, u_p=0.5682, u_r/uas=0.5584, l_f1=0.526, l_p=0.5306, l_r/las=0.5215
CWSMetric: rec=0.9051, pre=0.9192, f1=0.9121
Evaluation on dev at Epoch 1/25. Step:503/12575:
SegAppCharParseF1Metric: u_f1=0.5836, u_p=0.5896, u_r/uas=0.5776, l_f1=0.5503, l_p=0.556, l_r/las=0.5447
CWSMetric: rec=0.9115, pre=0.9289, f1=0.9201

Evaluate data in 3.06 seconds!
Evaluate data in 7.15 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.6687, u_p=0.6708, u_r/uas=0.6666, l_f1=0.6338, l_p=0.6358, l_r/las=0.6318
CWSMetric: rec=0.9342, pre=0.9398, f1=0.937
Evaluation on dev at Epoch 2/25. Step:1006/12575:
SegAppCharParseF1Metric: u_f1=0.6814, u_p=0.6845, u_r/uas=0.6783, l_f1=0.6504, l_p=0.6534, l_r/las=0.6475
CWSMetric: rec=0.9386, pre=0.9472, f1=0.9429

Evaluate data in 3.09 seconds!
Evaluate data in 7.06 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7103, u_p=0.7122, u_r/uas=0.7083, l_f1=0.6779, l_p=0.6797, l_r/las=0.676
CWSMetric: rec=0.9468, pre=0.9518, f1=0.9493
Evaluation on dev at Epoch 3/25. Step:1509/12575:
SegAppCharParseF1Metric: u_f1=0.7231, u_p=0.7263, u_r/uas=0.72, l_f1=0.6956, l_p=0.6986, l_r/las=0.6926
CWSMetric: rec=0.9491, pre=0.9572, f1=0.9531

Evaluate data in 3.16 seconds!
Evaluate data in 7.76 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7358, u_p=0.7388, u_r/uas=0.7329, l_f1=0.7074, l_p=0.7102, l_r/las=0.7046
CWSMetric: rec=0.9504, pre=0.9574, f1=0.9539
Evaluation on dev at Epoch 4/25. Step:2012/12575:
SegAppCharParseF1Metric: u_f1=0.7414, u_p=0.746, u_r/uas=0.7367, l_f1=0.7157, l_p=0.7202, l_r/las=0.7112
CWSMetric: rec=0.9509, pre=0.9622, f1=0.9565

Evaluate data in 3.17 seconds!
Evaluate data in 7.06 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7457, u_p=0.7485, u_r/uas=0.7428, l_f1=0.718, l_p=0.7208, l_r/las=0.7153
CWSMetric: rec=0.9525, pre=0.9592, f1=0.9558
Evaluation on dev at Epoch 5/25. Step:2515/12575:
SegAppCharParseF1Metric: u_f1=0.7569, u_p=0.7613, u_r/uas=0.7526, l_f1=0.7307, l_p=0.7349, l_r/las=0.7265
CWSMetric: rec=0.9532, pre=0.9636, f1=0.9584

Evaluate data in 2.96 seconds!
Evaluate data in 7.1 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7511, u_p=0.7548, u_r/uas=0.7476, l_f1=0.7246, l_p=0.7281, l_r/las=0.7212
CWSMetric: rec=0.952, pre=0.9603, f1=0.9561
Evaluation on dev at Epoch 6/25. Step:3018/12575:
SegAppCharParseF1Metric: u_f1=0.7594, u_p=0.7654, u_r/uas=0.7535, l_f1=0.7363, l_p=0.7421, l_r/las=0.7306
CWSMetric: rec=0.952, pre=0.9657, f1=0.9588

Evaluate data in 3.06 seconds!
Evaluate data in 7.05 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7618, u_p=0.7629, u_r/uas=0.7607, l_f1=0.7354, l_p=0.7365, l_r/las=0.7343
CWSMetric: rec=0.9572, pre=0.9602, f1=0.9587
Evaluation on dev at Epoch 7/25. Step:3521/12575:
SegAppCharParseF1Metric: u_f1=0.7719, u_p=0.7751, u_r/uas=0.7687, l_f1=0.7492, l_p=0.7523, l_r/las=0.7461
CWSMetric: rec=0.9587, pre=0.9664, f1=0.9625

Evaluate data in 2.99 seconds!
Evaluate data in 6.87 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7653, u_p=0.7676, u_r/uas=0.763, l_f1=0.7388, l_p=0.741, l_r/las=0.7366
CWSMetric: rec=0.9565, pre=0.962, f1=0.9592
Evaluation on dev at Epoch 8/25. Step:4024/12575:
SegAppCharParseF1Metric: u_f1=0.7705, u_p=0.7745, u_r/uas=0.7666, l_f1=0.7455, l_p=0.7493, l_r/las=0.7417
CWSMetric: rec=0.9574, pre=0.9668, f1=0.9621

Evaluate data in 3.1 seconds!
Evaluate data in 7.36 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7742, u_p=0.7756, u_r/uas=0.7729, l_f1=0.7477, l_p=0.749, l_r/las=0.7464
CWSMetric: rec=0.9595, pre=0.9629, f1=0.9612
Evaluation on dev at Epoch 9/25. Step:4527/12575:
SegAppCharParseF1Metric: u_f1=0.7744, u_p=0.7775, u_r/uas=0.7714, l_f1=0.7511, l_p=0.754, l_r/las=0.7482
CWSMetric: rec=0.9583, pre=0.9657, f1=0.962

Evaluate data in 2.96 seconds!
Evaluate data in 6.81 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7783, u_p=0.7811, u_r/uas=0.7756, l_f1=0.7535, l_p=0.7561, l_r/las=0.7508
CWSMetric: rec=0.9587, pre=0.965, f1=0.9618
Evaluation on dev at Epoch 10/25. Step:5030/12575:
SegAppCharParseF1Metric: u_f1=0.7827, u_p=0.7864, u_r/uas=0.7791, l_f1=0.7609, l_p=0.7645, l_r/las=0.7573
CWSMetric: rec=0.9599, pre=0.9686, f1=0.9642

Evaluate data in 3.1 seconds!
Evaluate data in 7.01 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7805, u_p=0.7816, u_r/uas=0.7793, l_f1=0.7555, l_p=0.7566, l_r/las=0.7544
CWSMetric: rec=0.9596, pre=0.9625, f1=0.961
Evaluation on dev at Epoch 11/25. Step:5533/12575:
SegAppCharParseF1Metric: u_f1=0.7833, u_p=0.7857, u_r/uas=0.7809, l_f1=0.7623, l_p=0.7647, l_r/las=0.76
CWSMetric: rec=0.9604, pre=0.9664, f1=0.9634

Evaluate data in 3.06 seconds!
Evaluate data in 7.07 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7853, u_p=0.7871, u_r/uas=0.7835, l_f1=0.7615, l_p=0.7632, l_r/las=0.7598
CWSMetric: rec=0.9601, pre=0.9644, f1=0.9622
Evaluation on dev at Epoch 12/25. Step:6036/12575:
SegAppCharParseF1Metric: u_f1=0.7889, u_p=0.7919, u_r/uas=0.786, l_f1=0.7678, l_p=0.7707, l_r/las=0.7649
CWSMetric: rec=0.9614, pre=0.9686, f1=0.965

Evaluate data in 3.0 seconds!
Evaluate data in 7.02 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7876, u_p=0.7887, u_r/uas=0.7865, l_f1=0.7626, l_p=0.7637, l_r/las=0.7615
CWSMetric: rec=0.9603, pre=0.9631, f1=0.9617
Evaluation on dev at Epoch 13/25. Step:6539/12575:
SegAppCharParseF1Metric: u_f1=0.7917, u_p=0.7951, u_r/uas=0.7882, l_f1=0.7693, l_p=0.7727, l_r/las=0.766
CWSMetric: rec=0.9614, pre=0.9695, f1=0.9654

Evaluate data in 3.08 seconds!
Evaluate data in 7.05 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7891, u_p=0.7898, u_r/uas=0.7885, l_f1=0.765, l_p=0.7657, l_r/las=0.7644
CWSMetric: rec=0.961, pre=0.9629, f1=0.9619
Evaluation on dev at Epoch 14/25. Step:7042/12575:
SegAppCharParseF1Metric: u_f1=0.7953, u_p=0.7979, u_r/uas=0.7927, l_f1=0.7728, l_p=0.7753, l_r/las=0.7703
CWSMetric: rec=0.9627, pre=0.9689, f1=0.9658

Evaluate data in 2.97 seconds!
Evaluate data in 6.86 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.789, u_p=0.7896, u_r/uas=0.7885, l_f1=0.7647, l_p=0.7652, l_r/las=0.7641
CWSMetric: rec=0.9621, pre=0.9637, f1=0.9629
Evaluation on dev at Epoch 15/25. Step:7545/12575:
SegAppCharParseF1Metric: u_f1=0.791, u_p=0.7939, u_r/uas=0.788, l_f1=0.7686, l_p=0.7715, l_r/las=0.7657
CWSMetric: rec=0.9624, pre=0.9694, f1=0.9659

Evaluate data in 3.06 seconds!
Evaluate data in 6.99 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7921, u_p=0.7932, u_r/uas=0.791, l_f1=0.7683, l_p=0.7694, l_r/las=0.7673
CWSMetric: rec=0.9631, pre=0.9658, f1=0.9644
Evaluation on dev at Epoch 16/25. Step:8048/12575:
SegAppCharParseF1Metric: u_f1=0.7982, u_p=0.8015, u_r/uas=0.795, l_f1=0.7782, l_p=0.7814, l_r/las=0.775
CWSMetric: rec=0.962, pre=0.9696, f1=0.9658

Evaluate data in 2.98 seconds!
Evaluate data in 6.87 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7914, u_p=0.7919, u_r/uas=0.7909, l_f1=0.7682, l_p=0.7687, l_r/las=0.7678
CWSMetric: rec=0.9624, pre=0.9639, f1=0.9631
Evaluation on dev at Epoch 17/25. Step:8551/12575:
SegAppCharParseF1Metric: u_f1=0.7936, u_p=0.7963, u_r/uas=0.791, l_f1=0.7736, l_p=0.7762, l_r/las=0.7711
CWSMetric: rec=0.9638, pre=0.9702, f1=0.967

Evaluate data in 3.01 seconds!
Evaluate data in 6.87 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7936, u_p=0.7945, u_r/uas=0.7927, l_f1=0.7704, l_p=0.7713, l_r/las=0.7695
CWSMetric: rec=0.9629, pre=0.9653, f1=0.9641
Evaluation on dev at Epoch 18/25. Step:9054/12575:
SegAppCharParseF1Metric: u_f1=0.7994, u_p=0.8019, u_r/uas=0.7969, l_f1=0.7797, l_p=0.7822, l_r/las=0.7773
CWSMetric: rec=0.9636, pre=0.9697, f1=0.9666

Evaluate data in 3.05 seconds!
Evaluate data in 7.09 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7928, u_p=0.7942, u_r/uas=0.7913, l_f1=0.7698, l_p=0.7712, l_r/las=0.7684
CWSMetric: rec=0.9616, pre=0.965, f1=0.9633
Evaluation on dev at Epoch 19/25. Step:9557/12575:
SegAppCharParseF1Metric: u_f1=0.8018, u_p=0.8055, u_r/uas=0.7982, l_f1=0.7805, l_p=0.7841, l_r/las=0.777
CWSMetric: rec=0.9624, pre=0.9707, f1=0.9665

Evaluate data in 3.03 seconds!
Evaluate data in 7.06 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7987, u_p=0.7996, u_r/uas=0.7977, l_f1=0.7766, l_p=0.7775, l_r/las=0.7757
CWSMetric: rec=0.9633, pre=0.9657, f1=0.9645
Evaluation on dev at Epoch 20/25. Step:10060/12575:
SegAppCharParseF1Metric: u_f1=0.8024, u_p=0.8054, u_r/uas=0.7994, l_f1=0.7807, l_p=0.7836, l_r/las=0.7778
CWSMetric: rec=0.9641, pre=0.9711, f1=0.9676

Evaluate data in 3.09 seconds!
Evaluate data in 6.8 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7998, u_p=0.8005, u_r/uas=0.7991, l_f1=0.7771, l_p=0.7777, l_r/las=0.7764
CWSMetric: rec=0.9639, pre=0.9659, f1=0.9649
Evaluation on dev at Epoch 21/25. Step:10563/12575:
SegAppCharParseF1Metric: u_f1=0.8065, u_p=0.8093, u_r/uas=0.8037, l_f1=0.7866, l_p=0.7893, l_r/las=0.7839
CWSMetric: rec=0.9643, pre=0.9709, f1=0.9676

Evaluate data in 3.01 seconds!
Evaluate data in 6.93 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.7994, u_p=0.8007, u_r/uas=0.7981, l_f1=0.7765, l_p=0.7777, l_r/las=0.7753
CWSMetric: rec=0.9631, pre=0.9662, f1=0.9646
Evaluation on dev at Epoch 22/25. Step:11066/12575:
SegAppCharParseF1Metric: u_f1=0.8051, u_p=0.8077, u_r/uas=0.8024, l_f1=0.785, l_p=0.7875, l_r/las=0.7824
CWSMetric: rec=0.9643, pre=0.9707, f1=0.9675

Evaluate data in 3.07 seconds!
Evaluate data in 7.03 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.8018, u_p=0.8021, u_r/uas=0.8015, l_f1=0.7794, l_p=0.7798, l_r/las=0.7791
CWSMetric: rec=0.9645, pre=0.9657, f1=0.9651
Evaluation on dev at Epoch 23/25. Step:11569/12575:
SegAppCharParseF1Metric: u_f1=0.8087, u_p=0.811, u_r/uas=0.8065, l_f1=0.7877, l_p=0.7898, l_r/las=0.7855
CWSMetric: rec=0.9648, pre=0.9703, f1=0.9675

Evaluate data in 3.03 seconds!
Evaluate data in 7.0 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.8004, u_p=0.8016, u_r/uas=0.7993, l_f1=0.7778, l_p=0.7789, l_r/las=0.7767
CWSMetric: rec=0.9633, pre=0.9662, f1=0.9647
Evaluation on dev at Epoch 24/25. Step:12072/12575:
SegAppCharParseF1Metric: u_f1=0.8065, u_p=0.8093, u_r/uas=0.8038, l_f1=0.7858, l_p=0.7885, l_r/las=0.7831
CWSMetric: rec=0.964, pre=0.9705, f1=0.9672

Evaluate data in 3.02 seconds!
Evaluate data in 7.0 seconds!
EvaluateCallback evaluation on data-test:
SegAppCharParseF1Metric: u_f1=0.8023, u_p=0.803, u_r/uas=0.8017, l_f1=0.7804, l_p=0.781, l_r/las=0.7797
CWSMetric: rec=0.9639, pre=0.9657, f1=0.9648
Evaluation on dev at Epoch 25/25. Step:12575/12575:
SegAppCharParseF1Metric: u_f1=0.8064, u_p=0.809, u_r/uas=0.8037, l_f1=0.7868, l_p=0.7894, l_r/las=0.7843
CWSMetric: rec=0.9647, pre=0.971, f1=0.9678

Best test performance(may not correspond to the best dev performance):{'SegAppCharParseF1Metric': {'u_f1': 0.8023, 'u_p': 0.803, 'u_r/uas': 0.8017, 'l_f1': 0.7804, 'l_p': 0.781, 'l_r/las': 0.7797}, 'CWSMetric': {'rec': 0.9639, 'pre': 0.9657, 'f1': 0.9648}} achieved at Epoch:25.
Best test performance(correspond to the best dev performance):{'SegAppCharParseF1Metric': {'u_f1': 0.8018, 'u_p': 0.8021, 'u_r/uas': 0.8015, 'l_f1': 0.7794, 'l_p': 0.7798, 'l_r/las': 0.7791}, 'CWSMetric': {'rec': 0.9645, 'pre': 0.9657, 'f1': 0.9651}} achieved at Epoch:23.

In Epoch:23/Step:11569, got best dev performance:
SegAppCharParseF1Metric: u_f1=0.8087, u_p=0.811, u_r/uas=0.8065, l_f1=0.7877, l_p=0.7898, l_r/las=0.7855
CWSMetric: rec=0.9648, pre=0.9703, f1=0.9675


```
 
