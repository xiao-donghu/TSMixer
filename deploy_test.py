import os
import sys

sys.path.append('./')

import pandas as pd

import warnings
import datetime
from run_test import run_model

current_script_path = os.path.abspath(__file__)
model_path = os.path.dirname(current_script_path)
from datetime import datetime
from datetime import datetime
import itertools
from datetime import datetime, timedelta

# 获取当前的日期和时间
now = datetime.now()


# 定义基于时间的两种不同的函数调用
def conditional_run(now):
    if now.hour == 7 and now.minute == 0:
        # 如果是正好上午7点，运行 exp1
        predicted_data_a, predicted_data_a_1 = run_model('long_term_forecast_a', 'Overall_re', './dataset/Overall/', 'Overall_re.csv', 192, 102,
                  '2023-01-01 00:00',
                  now, 1)
    else:
        # 如果不是上午7点，则运行 exp0，并且使用当前时间减去一分钟作为结束时间
        '''
        one_minute_ago = now - timedelta(minutes=1)
        predicted_data_b_30, predicted_data_b_1_30 = run_model('long_term_forecast_b', 'Overall_merge', './dataset/Overall/', 'Overall_merge.csv', 96, 1,
                  '2024-09-10 18:36:00',now, 0)  # 使用当前时间减去一分钟作为结束时间

        # 再次运行 exp0，但是使用不同的参数，并且同样使用当前时间减去一分钟作为结束时间
        predicted_data_b_60, predicted_data_b_1_60 =run_model('long_term_forecast_b_hour', 'Overall_merge_hour', './dataset/Overall/', 'Overall_merge_hour.csv', 36,
                  1,
                  '2024-09-10 18:36:00',
                  now, 0)
        # 合并 predicted_data_b_1_30 和 predicted_data_b_1_60
        combined_data = pd.concat([predicted_data_b_30,predicted_data_b_60],ignore_index=True)
        combined_data_1 = pd.concat([predicted_data_b_1_30, predicted_data_b_1_60], ignore_index=True)
        # 输出为 CSV 文件
        output_filename = 'combined_predicted_data.csv'
        combined_data.to_csv(output_filename, index=False)

        output_filename = 'combined_predicted_data_1.csv'
        combined_data_1.to_csv(output_filename, index=False)
        
        '''
        # 1. 定义参数列表
        seq_len_list = [160,168,176,184,192]  # 可调整不同的输入序列长度
        d_model_list = [8]  # 批次大小
        d_ff_list = [16]  # 隐藏层维度
        activation_list = ['gelu']  # 激活函数
        patience_list = [5]
        learning_rate_list = [0.001]  # 学习率
        epoch_list  = [20]

        # 2. 使用 itertools.product 生成参数组合
        param_combinations = list(itertools.product(seq_len_list, d_model_list, d_ff_list,activation_list, patience_list, learning_rate_list,epoch_list))

        # 3. 循环执行网格搜索
        for params in param_combinations:
            seq_len, d_model, d_ff, activation, patience, learning_rate,epoch = params
            predicted_data_a, predicted_data_a_1 = run_model('long_term_forecast_a', 'Overall_re', './dataset/Overall/',
                                                         'Overall_re.csv', seq_len, 102,
                                                         '2023-01-01 00:00',
                                                         '2024-08-11 23:50', 2,d_model,d_ff,activation,patience,learning_rate,epoch)
            output_filename = 'combined_predicted_data.csv'
            predicted_data_a.to_csv(output_filename, index=False)
            output_filename = 'combined_predicted_data_1.csv'
            predicted_data_a_1.to_csv(output_filename, index=False)


# todo 多个10分钟的数据，一次跑完
# todo 缺少少量数据时，用差值补全
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    conditional_run(now)

