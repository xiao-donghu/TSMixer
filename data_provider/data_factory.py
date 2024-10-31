
from data_provider.data_loader import Dataset_Custom, Dataset_Pred

data_dict = {
    'custom': Dataset_Custom,
    'ZhouShanIn': Dataset_Custom,
    'ZhouShanPred': Dataset_Custom,
    'ZhouShanIn_yuan': Dataset_Custom,
    'ZhouShanIn_tian': Dataset_Custom,
    "ZhuJiaJianIn": Dataset_Custom,
    "ZhuJiaJianIn_yuan": Dataset_Custom,
    "ZhouShanOut": Dataset_Custom,
    "ZhouShanOut_OT": Dataset_Custom,
    "Overall_re": Dataset_Custom,
    "Overall_merge": Dataset_Custom,
    "Overall_merge_hour":Dataset_Custom,
    "ZhouShanIn_merge": Dataset_Custom,
    "ZhouShanOut_merge": Dataset_Custom,
    "ZhuJiaJianIn_merge": Dataset_Custom,
    "ZhuJiaJianOut_merge": Dataset_Custom,
    'Overall_jintang_re':Dataset_Custom,
    'Overall_jintang':Dataset_Custom,
    'Overall_jintang_hour':Dataset_Custom,
}

import tensorflow as tf

def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size
        freq = args.freq
        Data = data_dict[args.data]
    elif flag == 'pred':
        shuffle_flag = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred  # 使用预测数据集
    else:
        shuffle_flag = True
        batch_size = args.batch_size
        freq = args.freq
        Data = data_dict[args.data]

    # 创建数据集
    if flag == 'pred':
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path=args.data_path,
            target=args.target,
            scale=True,  # 默认值
            inverse=True,  # 默认值
            timeenc=timeenc,
            freq=freq
        )
    else:
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )

    print(flag, len(data_set))

    # 使用 tf.data.Dataset 从 numpy 数组中创建数据集
    def generator():
        for i in range(len(data_set)):
            yield data_set[i]

    # 使用生成器创建 TensorFlow 数据集
    output_signature = (
        tf.TensorSpec(shape=data_set[0][0].shape, dtype=tf.float32),
        tf.TensorSpec(shape=data_set[0][1].shape, dtype=tf.float32),
        tf.TensorSpec(shape=data_set[0][2].shape, dtype=tf.float32),
        tf.TensorSpec(shape=data_set[0][3].shape, dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if shuffle_flag:
        dataset = dataset.shuffle(buffer_size=len(data_set))

    # 设置批处理和预取
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return data_set, dataset