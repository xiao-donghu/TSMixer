import argparse
import os
from audioop import avg

import tensorflow as tf

from exp.exp_long_term_forecasting_a import Exp_Long_Term_Forecast_a
from exp.exp_long_term_forecasting_jintang_a import Exp_Long_Term_Forecast_jintang_a
from exp.exp_long_term_forecasting_b import Exp_Long_Term_Forecast_b
from exp.exp_long_term_forecasting_b_hour import Exp_Long_Term_Forecast_b_hour
from exp.exp_long_term_forecasting_jintang_b import Exp_Long_Term_Forecast_jintang_b
from exp.exp_long_term_forecasting_jintang_b_hour import Exp_Long_Term_Forecast_jintang_b_hour

import pandas as pd
from data_provider.data_factory import data_provider
from utils.print_args import print_args
import random
import numpy as np
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_model(task_name=None, data=None, root_path=None, data_path=None, seq_len=None, pred_len=None, enc_in=None,
              dec_in=None, c_out=None, start_time=None,
              end_time=None, itr=None):
    # Fix seed
    global predicted_data_1
    fix_seed = 2021
    set_seed(fix_seed)

    if pred_len == 1 and (task_name == 'long_term_forecast_b' or task_name == 'long_term_forecast_b_hour'):
        # 读取数据
        df = pd.read_csv('dataset/Overall_b.csv', encoding='gbk')
        df['window_start'] = pd.to_datetime(df['window_start'])

        # 创建完整的日期时间范围
        full_range = pd.date_range(start=start_time, end=end_time, freq='1min')
        full_df = pd.DataFrame({'window_start': full_range})

        # 定义处理函数
        def create_complete_sequence(df, full_df):
            merged_df = pd.merge(full_df, df, on='window_start', how='left')
            merged_df['kk_flow'].fillna(0, inplace=True)
            return merged_df

        # 分别处理进出舟山西和朱家尖的数据
        df_zhoushanin = df[df['kk_type'] == 1][['window_start', 'kk_flow']]
        df_zhoushanout = df[df['kk_type'] == 2][['window_start', 'kk_flow']]
        df_zhujiajianin = df[df['kk_type'] == 7][['window_start', 'kk_flow']]
        df_zhujiajianout = df[df['kk_type'] == 8][['window_start', 'kk_flow']]

        df_zhoushanin_complete = create_complete_sequence(df_zhoushanin, full_df)
        df_zhoushanout_complete = create_complete_sequence(df_zhoushanout, full_df)
        df_zhujiajianin_complete = create_complete_sequence(df_zhujiajianin, full_df)
        df_zhujiajianout_complete = create_complete_sequence(df_zhujiajianout, full_df)

        combined_df = pd.DataFrame({
            'date': df_zhoushanin_complete['window_start'],
            'ZhouShanIn': df_zhoushanin_complete['kk_flow'],
            'ZhouShanOut': df_zhoushanout_complete['kk_flow'],
            'ZhuJiaJianIn': df_zhujiajianin_complete['kk_flow'],
            'OT': df_zhujiajianout_complete['kk_flow']
        })

        print(combined_df.head())

        # 删除连续的零
        def remove_consecutive_zeros(df):
            rows_to_remove = set()
            columns_to_check = ['ZhouShanIn', 'ZhouShanOut', 'ZhuJiaJianIn', 'OT']

            for column in columns_to_check:
                i = 0
                while i <= len(df) - 60:
                    if (df[column].iloc[i:i + 60] == 0).all():
                        rows_to_remove.update(range(i, i + 60))
                        i += 60
                    else:
                        i += 1

            return df.drop(rows_to_remove)

        filtered_combined_df = remove_consecutive_zeros(combined_df)
        filtered_combined_df['date'] = pd.to_datetime(filtered_combined_df['date'])

        # 根据 task_name 执行不同的合并逻辑
        if task_name == 'long_term_forecast_b':
            # 合并 30 分钟数据逻辑
            merged_df_30 = pd.DataFrame()
            remainder_rows_30 = len(filtered_combined_df) % 30

            if remainder_rows_30 > 0:
                remainder_group_30 = filtered_combined_df.iloc[:remainder_rows_30]
                merged_remainder_row_30 = remainder_group_30.drop(columns=['date']).sum()
                merged_remainder_row_30['date'] = remainder_group_30.iloc[0]['date']
                merged_df_30 = pd.concat([merged_df_30, merged_remainder_row_30.to_frame().T], ignore_index=True)

            for i in range(remainder_rows_30, len(filtered_combined_df), 30):
                group_30 = filtered_combined_df.iloc[i:i + 30]
                merged_row_30 = group_30.drop(columns=['date']).sum()
                merged_row_30['date'] = group_30.iloc[0]['date']
                merged_df_30 = pd.concat([merged_df_30, merged_row_30.to_frame().T], ignore_index=True)

            merged_df_30 = merged_df_30[['date'] + [col for col in merged_df_30.columns if col != 'date']]
            merged_df_30.to_csv('./dataset/Overall/Overall_merge.csv', index=False)

        elif task_name == 'long_term_forecast_b_hour':
            # 合并 60 分钟数据逻辑
            merged_df_60 = pd.DataFrame()
            remainder_rows_60 = len(filtered_combined_df) % 60

            if remainder_rows_60 > 0:
                remainder_group_60 = filtered_combined_df.iloc[:remainder_rows_60]
                merged_remainder_row_60 = remainder_group_60.drop(columns=['date']).sum()
                merged_remainder_row_60['date'] = remainder_group_60.iloc[0]['date']
                merged_df_60 = pd.concat([merged_df_60, merged_remainder_row_60.to_frame().T], ignore_index=True)

            for i in range(remainder_rows_60, len(filtered_combined_df), 60):
                group_60 = filtered_combined_df.iloc[i:i + 60]
                merged_row_60 = group_60.drop(columns=['date']).sum()
                merged_row_60['date'] = group_60.iloc[0]['date']
                merged_df_60 = pd.concat([merged_df_60, merged_row_60.to_frame().T], ignore_index=True)

            merged_df_60 = merged_df_60[['date'] + [col for col in merged_df_60.columns if col != 'date']]
            merged_df_60.to_csv('./dataset/Overall/Overall_merge_hour.csv', index=False)

    elif pred_len == 1 and (task_name == 'long_term_forecast_jintang_b' or task_name == 'long_term_forecast_jintang_b_hour'):
        # 这里执行相应的代码块
        # 读取数据
        df = pd.read_csv('dataset/Overall_jintang_b.csv', encoding='gbk')

        # 直接使用 pd.to_datetime 自动处理有秒和无秒的情况
        df['window_start'] = pd.to_datetime(df['window_start'], errors='coerce')

        # 创建完整的日期时间范围
        full_range = pd.date_range(start=start_time, end=end_time, freq='1min')
        full_df = pd.DataFrame({'window_start': full_range})

        # 定义处理函数
        def create_complete_sequence(df, full_df):
            merged_df = pd.merge(full_df, df, on='window_start', how='left')
            merged_df['kk_flow'].fillna(0, inplace=True)
            return merged_df

        # 分别处理进出金塘
        df_jintangin = df[df['kk_type'] == 15][['window_start', 'kk_flow']]
        df_jintangout = df[df['kk_type'] == 16][['window_start', 'kk_flow']]

        df_jintangin_complete = create_complete_sequence(df_jintangin, full_df)
        df_jintangout_complete = create_complete_sequence(df_jintangout, full_df)

        combined_df = pd.DataFrame({
            'date': df_jintangin_complete['window_start'],
            'JinTangIn': df_jintangin_complete['kk_flow'],
            'OT': df_jintangout_complete['kk_flow']
        })

        print(combined_df.head())

        # 删除连续的零
        def remove_consecutive_zeros(df):
            rows_to_remove = set()
            columns_to_check = ['JinTangIn', 'OT']

            for column in columns_to_check:
                i = 0
                while i <= len(df) - 60:
                    if (df[column].iloc[i:i + 60] == 0).all():
                        rows_to_remove.update(range(i, i + 60))
                        i += 60
                    else:
                        i += 1

            return df.drop(rows_to_remove)

        filtered_combined_df = remove_consecutive_zeros(combined_df)
        filtered_combined_df['date'] = pd.to_datetime(filtered_combined_df['date'])

        # 根据 task_name 执行不同的合并逻辑
        if task_name == 'long_term_forecast_jintang_b':
            # 合并 30 分钟数据逻辑
            merged_df_30 = pd.DataFrame()
            remainder_rows_30 = len(filtered_combined_df) % 30

            if remainder_rows_30 > 0:
                remainder_group_30 = filtered_combined_df.iloc[:remainder_rows_30]
                merged_remainder_row_30 = remainder_group_30.drop(columns=['date']).sum()
                merged_remainder_row_30['date'] = remainder_group_30.iloc[0]['date']
                merged_df_30 = pd.concat([merged_df_30, merged_remainder_row_30.to_frame().T], ignore_index=True)

            for i in range(remainder_rows_30, len(filtered_combined_df), 30):
                group_30 = filtered_combined_df.iloc[i:i + 30]
                merged_row_30 = group_30.drop(columns=['date']).sum()
                merged_row_30['date'] = group_30.iloc[0]['date']
                merged_df_30 = pd.concat([merged_df_30, merged_row_30.to_frame().T], ignore_index=True)

            merged_df_30 = merged_df_30[['date'] + [col for col in merged_df_30.columns if col != 'date']]
            merged_df_30.to_csv('./dataset/Overall/Overall_jintang_30min.csv', index=False)

        elif task_name == 'long_term_forecast_jintang_b_hour':
            # 合并 60 分钟数据逻辑
            merged_df_60 = pd.DataFrame()
            remainder_rows_60 = len(filtered_combined_df) % 60

            if remainder_rows_60 > 0:
                remainder_group_60 = filtered_combined_df.iloc[:remainder_rows_60]
                merged_remainder_row_60 = remainder_group_60.drop(columns=['date']).sum()
                merged_remainder_row_60['date'] = remainder_group_60.iloc[0]['date']
                merged_df_60 = pd.concat([merged_df_60, merged_remainder_row_60.to_frame().T], ignore_index=True)

            for i in range(remainder_rows_60, len(filtered_combined_df), 60):
                group_60 = filtered_combined_df.iloc[i:i + 60]
                merged_row_60 = group_60.drop(columns=['date']).sum()
                merged_row_60['date'] = group_60.iloc[0]['date']
                merged_df_60 = pd.concat([merged_df_60, merged_row_60.to_frame().T], ignore_index=True)

            merged_df_60 = merged_df_60[['date'] + [col for col in merged_df_60.columns if col != 'date']]
            merged_df_60.to_csv('./dataset/Overall/Overall_jintang_hour.csv', index=False)

    elif task_name == 'long_term_forecast_a':
        # 读取数据
        df = pd.read_csv('dataset/Overall.csv', encoding='gbk')
        # 确保 window_start 列是 datetime 类型
        df['window_start'] = pd.to_datetime(df['window_start'])

        # 创建完整的日期时间范围
        full_range = pd.date_range(start=start_time, end=end_time, freq='10min')
        full_df = pd.DataFrame({'window_start': full_range})

        # 定义处理函数
        def create_complete_sequence(df, full_df):
            # 将 full_df 左连接到 df
            merged_df = pd.merge(full_df, df, on='window_start', how='left')
            merged_df['kk_flow'].fillna(0, inplace=True)
            return merged_df

        # 对每个 DataFrame 执行操作
        df_zhoushanin = df[df['kk_type'] == 1][['window_start', 'kk_flow']]
        df_zhoushanout = df[df['kk_type'] == 2][['window_start', 'kk_flow']]
        df_zhujiajianin = df[df['kk_type'] == 7][['window_start', 'kk_flow']]
        df_zhujiajianout = df[df['kk_type'] == 8][['window_start', 'kk_flow']]

        # 处理每个 DataFrame
        df_zhoushanin_complete = create_complete_sequence(df_zhoushanin, full_df)
        df_zhoushanout_complete = create_complete_sequence(df_zhoushanout, full_df)
        df_zhujiajianin_complete = create_complete_sequence(df_zhujiajianin, full_df)
        df_zhujiajianout_complete = create_complete_sequence(df_zhujiajianout, full_df)

        combined_df = pd.DataFrame({
            'date': df_zhoushanin_complete['window_start'],
            'ZhouShanIn': df_zhoushanin_complete['kk_flow'],
            'ZhouShanOut': df_zhoushanout_complete['kk_flow'],
            'ZhuJiaJianIn': df_zhujiajianin_complete['kk_flow'],
            'OT': df_zhujiajianout_complete['kk_flow']
        })

        # 输出合并后的 DataFrame
        print(combined_df.head())

        # 定义处理函数，直接对 DataFrame 进行操作
        def remove_consecutive_zeros(df):
            # 初始化一个空的索引列表，用于存储需要删除的行的索引
            rows_to_remove = set()

            # 获取所有需要检查的列名
            columns_to_check = ['ZhouShanIn', 'ZhouShanOut', 'ZhuJiaJianIn', 'OT']

            # 遍历数据，查找连续六个零的情况
            for column in columns_to_check:
                i = 0
                while i <= len(df) - 6:
                    # 检查是否存在连续6个零
                    if (df[column].iloc[i:i + 6] == 0).all():
                        # 如果存在，将这6个索引添加到需要删除的列表中
                        rows_to_remove.update(range(i, i + 6))
                        # 跳过这6行，继续检查
                        i += 6
                    else:
                        i += 1

            # 删除这些行
            df_filtered = df.drop(rows_to_remove)

            return df_filtered

        # 直接在内存中处理 DataFrame
        filtered_combined_df = remove_consecutive_zeros(combined_df)
        # 确保 window_start 列是 datetime 类型
        filtered_combined_df['date'] = pd.to_datetime(filtered_combined_df['date'])

        # 保存合并后的数据到新的 CSV 文件
        filtered_combined_df.to_csv('./dataset/Overall/Overall_re.csv', index=False)

    else:
        # 读取数据
        df = pd.read_csv('./dataset/Overall_jintang.csv', encoding='gbk')

        df['window_start'] = pd.to_datetime(df['window_start'])

        # 创建完整的日期时间范围
        full_range = pd.date_range(start=start_time, end=end_time, freq='10min')
        full_df = pd.DataFrame({'window_start': full_range})

        # 定义处理函数
        def create_complete_sequence(df, full_df):
            # 将 full_df 左连接到 df
            merged_df = pd.merge(full_df, df, on='window_start', how='left')
            merged_df['kk_flow'].fillna(0, inplace=True)
            return merged_df

        # 对每个 DataFrame 执行操作
        df_jintangin = df[df['kk_type'] == 15][['window_start', 'kk_flow']]
        df_jintangout = df[df['kk_type'] == 16][['window_start', 'kk_flow']]

        # 处理每个 DataFrame
        df_jintangin_complete = create_complete_sequence(df_jintangin, full_df)
        df_jintangout_complete = create_complete_sequence(df_jintangout, full_df)

        combined_df = pd.DataFrame({
            'date': df_jintangin_complete['window_start'],
            'JinTangIn': df_jintangin_complete['kk_flow'],
            'OT': df_jintangout_complete['kk_flow']
        })

        # 输出合并后的 DataFrame
        print(combined_df.head())

        # 定义处理函数，直接对 DataFrame 进行操作
        def remove_consecutive_zeros(df):
            # 初始化一个空的索引列表，用于存储需要删除的行的索引
            rows_to_remove = set()

            # 获取所有需要检查的列名
            columns_to_check = ['JinTangIn', 'OT']

            # 遍历数据，查找连续六个零的情况
            for column in columns_to_check:
                i = 0
                while i <= len(df) - 6:
                    # 检查是否存在连续6个零
                    if (df[column].iloc[i:i + 6] == 0).all():
                        # 如果存在，将这6个索引添加到需要删除的列表中
                        rows_to_remove.update(range(i, i + 6))
                        # 跳过这6行，继续检查
                        i += 6
                    else:
                        i += 1

            # 删除这些行
            df_filtered = df.drop(rows_to_remove)

            return df_filtered

        # 直接在内存中处理 DataFrame
        filtered_combined_df = remove_consecutive_zeros(combined_df)
        # 确保 window_start 列是 datetime 类型
        filtered_combined_df['date'] = pd.to_datetime(filtered_combined_df['date'])

        # 保存合并后的数据到新的 CSV 文件
        filtered_combined_df.to_csv('./dataset/Overall/Overall_jintang_re.csv', index=False)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast_a',
                        help='task_name, options:[long_term_forecast_a,long_term_forecast_b,long_term_forecast_b_hour, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model_id', type=str, default='Overall_tf_merge', help='model id')
    parser.add_argument('--model', type=str, default='TSMixer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--do_predict', action='store_true', default=1, help='whether to predict unseen future data')
    parser.add_argument('--time_interval', action='store_true', default=30, help='predict time_interval')
    # data loader
    parser.add_argument('--data', type=str, default='Overall_merge', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/Overall/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Overall_merge_1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='30min',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=160, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=20, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=2, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=60, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="ShapeDTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA warp preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discriminative DTW warp preset augmentation")
    parser.add_argument('--tunediscdtw', default=False, action="store_true",
                        help="Tuned Discriminative DTW warp preset augmentation")

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("使用 GPU")
    else:
        print("GPU 不可用，使用 CPU")
    args = parser.parse_args()

    if data_path:
        args.data_path = data_path
    if root_path:
        args.root_path = root_path
    if seq_len:
        args.seq_len = seq_len
    if pred_len:
        args.pred_len = pred_len
    if task_name:
        args.task_name = task_name
    if data:
        args.data = data
    if enc_in:
        args.enc_in = enc_in
    if dec_in:
        args.dec_in = dec_in
    if c_out:
        args.c_out = c_out

    args.use_gpu = tf.config.list_physical_devices('GPU') != []

    if args.use_gpu and args.use_multi_gpu:
        devices = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in devices]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # Select the appropriate experiment class based on task_name
    if args.task_name == 'long_term_forecast_b':
        Exp = Exp_Long_Term_Forecast_b
    elif args.task_name == 'long_term_forecast_a':
        Exp = Exp_Long_Term_Forecast_a
    elif args.task_name == 'long_term_forecast_b_hour':
        Exp = Exp_Long_Term_Forecast_b_hour
    elif args.task_name == 'long_term_forecast_jintang_b':
        Exp = Exp_Long_Term_Forecast_jintang_b
    elif args.task_name == 'long_term_forecast_jintang_b_hour':
        Exp = Exp_Long_Term_Forecast_jintang_b_hour
    elif args.task_name == 'long_term_forecast_jintang_a':
        Exp = Exp_Long_Term_Forecast_jintang_a
    if args.is_training:
        args.seq_len = seq_len  # Set the current seq_len value
        print(f"Using seq_len={seq_len}")  # Add this line to print the current seq_len
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_{}_{}_sl{}_ll{}_pl{}_in{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.time_interval,
                args.enc_in,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.inverse,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, 1)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                predicted_data, predicted_data_1 = exp.predict(setting, True)

    else:
        args.seq_len = seq_len  # Set the current seq_len value
        print(f"Using seq_len={seq_len}")  # Add this line to print the current seq_len
        setting = '{}_{}_{}_{}_ft{}_{}_{}_sl{}_ll{}_pl{}_in{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.time_interval,
            args.enc_in,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.inverse,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, itr)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        #exp.test(setting, test=1)
        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            predicted_data, predicted_data_1 = exp.predict(setting, True)

    tf.keras.backend.clear_session()
    return predicted_data, predicted_data_1


