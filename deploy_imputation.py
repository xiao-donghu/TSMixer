import os
import sys

sys.path.append('./')

import pandas

import warnings
import datetime
from run import run_model
import pandas as pd

current_script_path = os.path.abspath(__file__)
model_path = os.path.dirname(current_script_path)
from datetime import datetime
from datetime import datetime

from datetime import datetime, timedelta

# 获取当前的日期和时间
now = datetime.now()
# 将秒和微秒设置为零，并更新 now 变量
now = now.replace(second=0, microsecond=0)


# run_model(task_name=None, data=None, root_path=None, data_path=None, seq_len=None, pred_len=None,enc_in=None,dec_in=None,c_out=None, start_time=None,end_time=None, itr=None):
# 定义基于时间的两种不同的函数调用
def conditional_run(now):
    seven_am = now.replace(hour=7, minute=0, second=0, microsecond=0)

    if now.time() == seven_am.time():
        # 如果是正好上午7点，运行 exp1

        df1, df2 = run_model('long_term_forecast_a', 'Overall_re', './dataset/Overall/', 'Overall_re.csv', 192, 102, 4,
                             4, 4,
                             '2023-01-01 00:00:00',
                             now, 1)
        df3, df4 = run_model('long_term_forecast_jintang_a', 'Overall_jintang_re', './dataset/Overall/',
                             'Overall_jintang_re.csv', 192, 102, 2, 2, 2,
                             '2023-01-01 00:00',
                             '2024-09-23 23:59', 1)

    else:
        '''
        # 如果不是上午7点，则运行 exp0，并且使用当前时间减去一分钟作为结束时间
        one_minute_ago = now - timedelta(minutes=1)
        df1, df2 = run_model('long_term_forecast_b', 'Overall_merge', './dataset/Overall/', 'Overall_merge_30min.csv',
                             96, 1, 4, 4, 4,
                             '2022-01-01 00:00',
                             '2024-09-12 23:59:00', 0)  # 使用当前时间减去一分钟作为结束时间

        # 再次运行 exp0，但是使用不同的参数，并且同样使用当前时间减去一分钟作为结束时间
        df3, df4 = run_model('long_term_forecast_b_hour', 'Overall_merge_hour', './dataset/Overall/',
                             'Overall_merge_hour.csv', 36,
                             1, 4, 4, 4,
                             '2023-01-01 00:00',
                             '2024-09-12 17:43', 0)

        df5, df6 = run_model('long_term_forecast_jintang_b', 'Overall_jintang', './dataset/Overall/',
                             'Overall_jintang_30min.csv', 102,
                             1, 2, 2, 2,
                             '2023-01-01 00:00',
                             '2024-09-23 23:59', 0)

        df7, df8 = run_model('long_term_forecast_jintang_b_hour', 'Overall_jintang_hour', './dataset/Overall/',
                             'Overall_jintang_hour.csv', 34,
                             1, 2, 2, 2,
                             '2023-01-01 00:00',
                             '2024-09-23 23:59', 1)
        '''
        # 读取输入的 CSV 文件
        input_csv = './dataset/all_anomaly_points.csv'  # 输入文件路径
        output_csv = 'predicted_times.csv'  # 输出文件路径

        df = pd.read_csv(input_csv)

        # 创建一个新的 CSV 文件，记录每次预测的第一个时间
        with open(output_csv, 'a') as f:
            f.write("predicted_time\n")  # 写入表头

        for index, row in df.iterrows():
            # 读取第一列的时间，并将其转换为 datetime 对象
            row_time = pd.to_datetime(row[0])

            # 计算前十分钟的时间
            end_time = row_time - timedelta(minutes=10)

            # 调用 run_model 函数，将 end_time 作为结束时间
            df1, _ = run_model('long_term_forecast_a', 'Overall_re', './dataset/Overall/', 'Overall_re.csv',
                               192, 102, 4, 4, 4,
                               '2023-01-01 00:00:00', end_time, 1)

            # 根据 kk_type 决定追加 y_pred_in 或 y_pred_out
            kk_type = row.get('kk_type')  # 假设 kk_type 是输入 CSV 的一列

            if kk_type in [1, 7]:
                # 追加 y_pred_in 的值到新的 CSV 文件
                predicted_value = df1['y_pred_in'].iloc[0]
            else:
                # 追加 y_pred_out 的值到新的 CSV 文件
                predicted_value = df1['y_pred_out'].iloc[0]

            # 追加预测的值到新的 CSV 文件中
            with open(output_csv, 'a') as f:
                f.write(f"{predicted_value}\n")
        '''
        df1, df2 = run_model('long_term_forecast_a', 'Overall_re', './dataset/Overall/', 'Overall_re.csv',
                             192, 102, 4,4, 4,
                             '2023-01-01 00:00:00',
                             '2023-01-24 07:00',
                              1)
        '''


# todo 多个10分钟的数据，一次跑完
# todo 缺少少量数据时，用差值补全
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    conditional_run(now)