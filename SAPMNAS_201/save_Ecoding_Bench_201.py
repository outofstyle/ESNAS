
import numpy as np
from nats_bench import create
from typing import List
import pandas as pd
api = create('D:\Download\TSNAS\\NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=True)

def savefile(my_list):
       """
      把文件存成csv格式的文件，header 写出列名，index写入行名称
        :param my_list: 要存储的一条列表数据
        :return:
        """
       df = pd.DataFrame(data=[my_list])
       df.to_csv("./Bench_201_ImageNet_X.csv", encoding="utf-8-sig", mode="a", header=False, index=False)


def savefile1(my_list):
    """
   把文件存成csv格式的文件，header 写出列名，index写入行名称
     :param my_list: 要存储的一条列表数据
     :return:
     """
    df = pd.DataFrame(data=[my_list])
    df.to_csv("./Bench_201_cifar100"
              "_vaild_y.csv", encoding="utf-8-sig", mode="a", header=False, index=False)

def savefile2(my_list):
    """
   把文件存成csv格式的文件，header 写出列名，index写入行名称
     :param my_list: 要存储的一条列表数据
     :return:
     """
    df = pd.DataFrame(data=[my_list])
    df.to_csv("./Bench_201_cfar100_test_y.csv", encoding="utf-8-sig", mode="a", header=False, index=False)



# def offline_econding(archive):
#     out_archive1 = []
#     for indi in archive:
#         # channel_list = re.findall(r"\d+", indi['channels'])
#         # gen_list = re.findall(r"|.*?|", indi['arch_str'])
#         # channel_list = list(map(int, channel_list))
#         # channel1, channel2, channel3, channel4, channel5 = map(int, channel_list)
#         gen = indi['arch_str']
#         gen_list = []
#         i = 0
#         while i < len(gen):
#             gen_list1 = []
#             j = i
#             gen_list1 = gen[j:j + 3]
#             if gen_list1 == 'nor':
#                 if gen[j + 9] == '1':
#                     gen_list.append(2)
#                 if gen[j + 9] == '3':
#                     gen_list.append(3)
#             if gen_list1 == 'non':
#                 gen_list.append(0)
#             if gen_list1 == 'ski':
#                 gen_list.append(1)
#             if gen_list1 == 'avg':
#                 gen_list.append(4)
#             i = i + 1
#         out_archive1.append(gen_list)
#     out_archive = np.array(out_archive1)
#     return out_archive

def offline_econding(archive):
    out_archive1 = []
    gen = archive
    gen_list = []
    i = 0
    while i < len(gen):
        gen_list1 = []
        j = i
        gen_list1 = gen[j:j + 3]
        if gen_list1 == 'nor':
            if gen[j + 9] == '1':
                gen_list.append(2)
            if gen[j + 9] == '3':
                gen_list.append(3)
        if gen_list1 == 'non':
            gen_list.append(0)
        if gen_list1 == 'ski':
            gen_list.append(1)
        if gen_list1 == 'avg':
            gen_list.append(4)
        i = i + 1
    out_archive1.append(gen_list)
    out_archive = np.array(out_archive1)
    return out_archive

edge_spots: int = 4 * (4 - 1) // 2

# the operation must be none rather than zeroize
allowed_ops: List[str] = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

def _encode(arch: str):
    # encode architecture phenotype to genotype
    # a sample architecture
    # '|none~0|+|avg_pool_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_1x1~2|'
    ops = []
    for node in arch.split('+'):
        op = [o.split('~')[0] for o in node.split('|') if o]
        ops.extend(op)
    x_ops = np.empty(edge_spots, dtype=np.int64)
    for i, op in enumerate(ops):
        x_ops[i] = (np.array(allowed_ops) == op).nonzero()[0][0]
    return x_ops


def search():
    acclist = []
    valid_acclist = []
    test_acclist = []
    netlist = []
    for i in range(0, 15625):
        info = api.get_more_info(i, 'cifar100', hp='200', is_random=False)
        config = api.get_net_config(i, 'cifar100')  #ImageNet16-120
        valid_acclist.append(info['valid-accuracy'])
        test_acclist.append(info['test-accuracy'])
        gen = config['arch_str']
        arch = offline_econding(gen)
        arch1 = _encode(gen)
        netlist.append(arch)

    #max(valid_acclist)
    #X = offline_econding(netlist)
    #X_all = offline_econding(netlist)
    valid = np.array(valid_acclist)
    test = np.array(test_acclist)
    for num, data in enumerate(netlist, start=1):
        savefile(data)
    for num, data in enumerate(valid):
        savefile1(data)
    for num, data in enumerate(test):
        savefile2(data)


if __name__ == '__main__':
    search()