import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np
import os
import time
import warnings
from datetime import datetime
import pandas as pd
import logging
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.dtw_metric import dtw, accelerated_dtw
import holidays
from utils.augmentation import run_augmentation, run_augmentation_single

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_jintang_b(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_jintang_b, self).__init__(args)
        self.model = self._build_model()
        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = model  # 使用策略作用域内的模型

        return model

    def _get_data(self, flag):
        # Adapted data provider function to return TensorFlow datasets
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)

    def _select_criterion(self):
        return tf.keras.losses.MeanSquaredError()

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # Create decoder input
        dec_inp = tf.zeros_like(batch_y[:, -self.args.pred_len:, :], dtype=tf.float32)
        dec_inp = tf.concat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1)

        # Run model
        outputs = self.model([batch_x, batch_x_mark, dec_inp, batch_y_mark], training=False)

        if self.args.output_attention:
            outputs = outputs[0]  # Extract attention output if needed

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []

        # Use tf.function for better performance, though it's optional for evaluation
        @tf.function
        def eval_step(batch_x, batch_y, batch_x_mark, batch_y_mark):
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            loss = criterion(batch_y, outputs)
            return loss

        for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
            batch_x = tf.cast(batch_x, tf.float32)
            batch_y = tf.cast(batch_y, tf.float32)
            batch_x_mark = tf.cast(batch_x_mark, tf.float32)
            batch_y_mark = tf.cast(batch_y_mark, tf.float32)

            # Decoder input
            dec_inp = tf.zeros_like(batch_y[:, -self.args.pred_len:, :], dtype=tf.float32)
            dec_inp = tf.concat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1)

            # Evaluate
            loss = eval_step(batch_x, batch_y, batch_x_mark, batch_y_mark)
            total_loss.append(loss.numpy())

        total_loss = np.mean(total_loss)
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 创建模型保存路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # EarlyStopping 设置
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 选择优化器和损失函数
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # AMP 自动混合精度设置
        if self.args.use_amp:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()

            # 遍历每个批次
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                # 数据转换为 tf.float32 格式
                batch_x = tf.cast(batch_x, tf.float32)
                batch_y = tf.cast(batch_y, tf.float32)
                batch_x_mark = tf.cast(batch_x_mark, tf.float32)
                batch_y_mark = tf.cast(batch_y_mark, tf.float32)

                # decoder input
                dec_inp = tf.zeros_like(batch_y[:, -self.args.pred_len:, :], dtype=tf.float32)
                dec_inp = tf.concat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1)

                # 梯度计算与反向传播
                with tf.GradientTape() as tape:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, training=True)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, training=True)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    # 计算损失
                    loss = criterion(batch_y, outputs)
                    train_loss.append(loss.numpy())

                # 反向传播和优化器更新
                gradients = tape.gradient(loss, self.model.trainable_variables)
                model_optim.apply_gradients(zip(gradients, self.model.trainable_variables))

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.numpy():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    iter_count = 0
                    time_now = time.time()

            # 计算平均训练损失
            train_loss = np.average(train_loss)

            # 验证集和测试集损失
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1} | "
                  f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            # 早停
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 加载最佳模型
        assert isinstance(path, str)
        best_model_path = os.path.join(path, 'checkpoint.h5')
        self.model.load_weights(best_model_path)

        return self.model

    '''
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd1 = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd1, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999


        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{},mape:{}'.format(mse, mae, dtw,mape))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{},mape:{}'.format(mse, mae, dtw,mape))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # 转换为 CSV 文件
        import pandas as pd

        def save_to_csv(data, filename):
            B, T, N = data.shape

            # 对于前 B-1 个样本，取第一个时间步的数据
            first_time_steps_data = data[:-1, 0, :]  # 形状为 (B-1, N)

            # 对于最后一个样本，取完整的时间步数据
            last_sample_data = data[-1, :, :]  # 形状为 (T, N)

            # 将这两部分数据拼接起来
            combined_data = np.vstack((first_time_steps_data, last_sample_data))

            # 将拼接后的数据转换为 DataFrame 并保存为 CSV
            df = pd.DataFrame(combined_data)
            df.to_csv(filename, index=False, header=False)

        save_to_csv(preds, folder_path + 'pred.csv')
        save_to_csv(trues, folder_path + 'true.csv')

        return
    '''

    def test(self, setting, test=0):
        # 获取测试数据和测试数据加载器
        test_data, test_loader = self._get_data(flag='test')

        # 创建保存结果的文件夹
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 加载模型
        if test:
            print(f"Setting: {setting}")
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.h5')
            if os.path.exists(model_path):
                print('Loading model...')

                # 假设输入的形状是 (batch_size, seq_len, sample_enc_in)
                sample_batch_size = 1  # 假设的批次大小，通常设为1就可以
                sample_seq_len = self.args.seq_len  # 根据模型定义的序列长度
                sample_enc_in = self.args.enc_in  # 根据输入的特征维度

                # 创建虚拟输入以初始化模型变量
                dummy_input_x = tf.zeros((sample_batch_size, sample_seq_len, sample_enc_in), dtype=tf.float32)
                dummy_input_mark = tf.zeros((sample_batch_size, sample_seq_len, sample_enc_in), dtype=tf.float32)
                dummy_output_x = tf.zeros((sample_batch_size, sample_seq_len, sample_enc_in), dtype=tf.float32)
                dummy_output_mark = tf.zeros((sample_batch_size, sample_seq_len, sample_enc_in), dtype=tf.float32)
                # 调用模型进行一次前向传播以初始化权重
                self.model(dummy_input_x, dummy_input_mark, dummy_output_x, dummy_output_mark, training=False)

                # 加载权重
                self.model.load_weights(model_path)
                print("Model weights loaded successfully.")
            else:
                print(f"Error: Model file does not exist at {model_path}")
                return

        preds = []
        trues = []

        # 使用 tf.function 装饰器提高性能
        @tf.function
        def predict_step(batch_x, batch_y, batch_x_mark, batch_y_mark):
            # decoder 输入
            dec_inp = tf.zeros_like(batch_y[:, -self.args.pred_len:, :], dtype=tf.float32)
            dec_inp = tf.concat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1)

            # 运行模型
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, training=False)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            return outputs, batch_y

        # 测试循环
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = tf.cast(batch_x, tf.float32)
            batch_y = tf.cast(batch_y, tf.float32)
            batch_x_mark = tf.cast(batch_x_mark, tf.float32)
            batch_y_mark = tf.cast(batch_y_mark, tf.float32)

            # 预测
            outputs, batch_y = predict_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

            # 逆标准化
            if test_data.scale and self.args.inverse:
                shape = outputs.shape
                outputs = test_data.inverse_transform(outputs.numpy().reshape(shape[0] * shape[1], -1)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.numpy().reshape(shape[0] * shape[1], -1)).reshape(shape)

            preds.append(outputs)
            trues.append(batch_y)

            if i % 20 == 0:
                input = batch_x.numpy()
                if test_data.scale and self.args.inverse:
                    shape = input.shape
                    input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                pd1 = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                visual(gt, pd1, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 结果保存
        result_folder_path = './results/' + setting + '/'
        if not os.path.exists(result_folder_path):
            os.makedirs(result_folder_path)

        # DTW 计算
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999

        mae, mse, rmse, mape, mspe = metric(trues, preds)
        print('mse:{}, mae:{}, dtw:{}, mape:{}'.format(mse, mae, dtw, mape))

        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, dtw:{}, mape:{}'.format(mse, mae, dtw, mape))
            f.write('\n')
            f.write('\n')

        np.save(result_folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(result_folder_path + 'pred.npy', preds)
        np.save(result_folder_path + 'true.npy', trues)

        # Squeeze 维度并保存到 CSV
        preds = np.squeeze(preds)
        trues = np.squeeze(trues)
        print("Preds shape after squeeze:", preds.shape)
        print("Trues shape after squeeze:", trues.shape)

        if len(preds.shape) == 1:
            df_preds = pd.DataFrame(preds, columns=['Predictions'])
            df_trues = pd.DataFrame(trues, columns=['True Values'])
        else:
            num_columns = preds.shape[1]
            pred_columns = [f'Prediction_{i + 1}' for i in range(num_columns)]
            true_columns = [f'True_Value_{i + 1}' for i in range(num_columns)]

            df_preds = pd.DataFrame(preds, columns=pred_columns)
            df_trues = pd.DataFrame(trues, columns=true_columns)

        df_preds.to_csv(result_folder_path + 'predictions.csv', index=False)
        df_trues.to_csv(result_folder_path + 'true_values.csv', index=False)

        print("CSV files saved successfully.")

        return

    def predict(self, setting, load=False):
        # 获取预测数据和预测数据加载器
        pred_data, pred_loader = self._get_data(flag='pred')

        # 加载模型
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            # 假设输入的形状是 (batch_size, seq_len, sample_enc_in)
            sample_batch_size = 1  # 假设的批次大小，通常设为1就可以
            sample_seq_len = self.args.seq_len  # 根据模型定义的序列长度
            sample_enc_in = self.args.enc_in  # 根据输入的特征维度

            # 创建虚拟输入以初始化模型变量
            dummy_input_x = tf.zeros((sample_batch_size, sample_seq_len, sample_enc_in), dtype=tf.float32)
            dummy_input_mark = tf.zeros((sample_batch_size, sample_seq_len, sample_enc_in), dtype=tf.float32)
            dummy_output_x = tf.zeros((sample_batch_size, sample_seq_len, sample_enc_in), dtype=tf.float32)
            dummy_output_mark = tf.zeros((sample_batch_size, sample_seq_len, sample_enc_in), dtype=tf.float32)
            # 调用模型进行一次前向传播以初始化权重
            self.model(dummy_input_x, dummy_input_mark, dummy_output_x, dummy_output_mark, training=False)

            best_model_path = os.path.join(path, 'checkpoint.h5')
            logging.info(best_model_path)
            self.model.load_weights(best_model_path)

        preds = []

        # 使用 tf.function 装饰器提高性能
        @tf.function
        def predict_step(batch_x, batch_y, batch_x_mark, batch_y_mark):
            # decoder 输入
            dec_inp = tf.zeros_like(batch_y[:, -self.args.pred_len:, :], dtype=tf.float32)
            dec_inp = tf.concat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1)

            # 运行模型
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, training=False)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            return outputs

        # 预测循环
        for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
            batch_x = tf.cast(batch_x, tf.float32)
            batch_y = tf.cast(batch_y, tf.float32)
            batch_x_mark = tf.cast(batch_x_mark, tf.float32)
            batch_y_mark = tf.cast(batch_y_mark, tf.float32)

            # 预测
            outputs = predict_step(batch_x, batch_y, batch_x_mark, batch_y_mark)
            if pred_data.scale and self.args.inverse:
                shape = outputs.shape
                outputs = pred_data.inverse_transform(outputs.numpy().reshape(shape[0] * shape[1], -1)).reshape(shape)

            # 转换为 NumPy 数组
            preds.append(outputs)

        # 合并所有批次的预测结果和时间标记
        preds = np.concatenate(preds, axis=0)

        print('Prediction shape:', preds.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        print('Prediction shape after reshape:', preds.shape)

        # 保存结果
        result_folder_path = './results/' + setting + '/'
        if not os.path.exists(result_folder_path):
            os.makedirs(result_folder_path)

        # 保存预测结果为 Numpy 文件
        np.save(result_folder_path + 'real_prediction.npy', preds)
        preds = np.maximum(preds, 0)

        # 将预测结果展平
        preds_flattened = preds.reshape(preds.shape[0] * preds.shape[1], preds.shape[-1])

        # 读取现有的时间序列 CSV 文件
        history_csv_path = './dataset/Overall/Overall_jintang_30min.csv'  # 你的历史数据文件路径
        df_history = pd.read_csv(history_csv_path)

        # 假设历史数据的时间戳在一列中，并且未来时间间隔为 30 分钟
        last_timestamp = pd.to_datetime(df_history['date'].iloc[-1])  # 获取最后一个时间戳
        time_interval = 30  # 这里假设预测是每 30 分钟进行的

        # 根据预测步长生成未来的时间戳（包括小时和分钟）
        future_timestamps = [
            last_timestamp + pd.Timedelta(minutes=time_interval * (i + 1)) for i in range(preds.shape[1])
        ]

        # 将未来的时间戳格式化为日期+小时+分钟
        future_timestamps = [ts.strftime('%Y-%m-%d %H:%M') for ts in future_timestamps]

        # 将预测结果放入 DataFrame 并附加未来的时间戳
        df_preds = pd.DataFrame(preds_flattened, columns=[f'Prediction_{i}' for i in range(preds_flattened.shape[1])])

        df_preds_30min = df_preds.copy()

        df_preds_30min['start_time'] = np.tile(future_timestamps, preds.shape[0])  # 重复时间戳
        df_preds_30min['start_time'] = pd.to_datetime(df_preds_30min['start_time'])

        # 从时间戳中提取年份
        df_preds_30min['year'] = df_preds_30min['start_time'].apply(lambda x: x.year)
        df_preds_30min['year'] = df_preds_30min['start_time'].apply(lambda x: x.year)

        # 计算 end_time 列
        df_preds_30min['end_time'] = pd.to_datetime(df_preds_30min['start_time']) + pd.Timedelta(
            minutes=time_interval)

        # 添加 pred_time 列（时间间隔）
        df_preds_30min['pred_time'] = time_interval

        # 添加 location 列
        df_preds_30min['location'] = '金塘'

        def add_date_range(df):
            def get_date_range(row):
                row['start_time'] = pd.to_datetime(row['start_time'])
                date = row['start_time'].date()
                year = row['start_time'].year
                cn_holidays = holidays.China(years=[year])

                if date in cn_holidays:
                    # 如果是节假日，返回节假日名称
                    holiday_name = cn_holidays.get(date, "节假日")
                    return pd.Series([f"{date} - {date}", holiday_name, date],
                                     index=['date_range', 'date_type', 'record_time'])
                elif row['start_time'].weekday() >= 5:  # 周末
                    next_day = row['start_time'] + pd.DateOffset(days=1)
                    return pd.Series([f"{date} - {next_day.date()}", "周末", date],
                                     index=['date_range', 'date_type', 'record_time'])
                else:  # 工作日
                    return pd.Series([f"{date}", "工作日", date], index=['date_range', 'date_type', 'record_time'])

            # 应用 get_date_range 函数，并将返回的三个值分别赋给 'date_range', 'date_type' 和 'record_time' 列
            df[['date_range', 'date_type', 'record_time']] = df.apply(lambda row: get_date_range(row), axis=1)

        # 示例使用
        # 假设df_preds_second_half和df_preds_first_half已经被定义并且包含了'start_time'列
        add_date_range(df_preds_30min)

        # 重命名预测列
        df_preds_30min.columns = ['y_pred_in', 'y_pred_out', 'start_time', 'year', 'end_time', 'pred_time',
                                       'location', 'date_range', 'date_type', 'record_time']

        new_column_order = ['date_range', 'year', 'date_type', 'location', 'start_time', 'end_time', 'pred_time',
                            'y_pred_in',
                            'y_pred_out', 'record_time']


        df_preds_30min = df_preds_30min.reindex(columns=new_column_order)


        predicted_data_b = df_preds_30min

        #############################################

        df_preds_first_half = df_preds.iloc[:, :1].copy()  # 第一列部分预测列
        df_preds_second_half = df_preds.iloc[:, 1:2].copy()  # 第二列预测列


        df_preds_first_half['time'] = np.tile(future_timestamps, preds.shape[0])  # 重复时间戳
        df_preds_second_half['time'] = np.tile(future_timestamps, preds.shape[0])


        df_preds_first_half['time'] = pd.to_datetime(df_preds_first_half['time'])
        df_preds_second_half['time'] = pd.to_datetime(df_preds_second_half['time'])

        def add_date_type(df):
            """
            添加 data_type 列，用于标识工作日、周末或节假日。

            参数:
                df (pd.DataFrame): 包含 'time' 列的 DataFrame。
            """

            # 初始化中国节假日
            year = df['time'].dt.year.unique()[0]  # 假设所有时间戳都在同一年
            cn_holidays = holidays.China(years=[year])

            def get_date_type(row):
                date = row['time'].date()
                if date in cn_holidays:
                    # 如果是节假日，返回节假日名称
                    return "节假日"
                elif row['time'].weekday() >= 5:  # 周末
                    return "周末"
                else:  # 工作日
                    return "工作日"

            # 应用 get_date_type 函数，并将返回的值赋给 'data_type' 列
            df['date_type'] = df.apply(lambda row: get_date_type(row), axis=1)

        # 调用函数
        add_date_type(df_preds_first_half)
        add_date_type(df_preds_second_half)

        # 从 time 列中提取日期，并添加为 date 列
        def extract_date(timestamp):
            """
            从 Timestamp 对象中提取日期部分。

            参数:
                timestamp (pd.Timestamp): 时间戳对象。
            返回:
                str: 日期字符串，格式为 'YYYY-MM-DD'。
            """
            try:
                # 直接从 Timestamp 中提取日期部分
                return timestamp.date().isoformat()  # 返回 'YYYY-MM-DD' 格式的字符串
            except AttributeError:
                return None  # 如果不是 Timestamp 对象，返回 None

        # 应用函数
        df_preds_first_half['date'] = df_preds_first_half['time'].apply(extract_date)
        df_preds_second_half['date'] = df_preds_second_half['time'].apply(extract_date)

        df_preds_first_half['location'] = '金塘'
        df_preds_second_half['location'] = '金塘'


        df_preds_first_half['direction'] = 'in'
        df_preds_second_half['direction'] = 'out'

        df_preds_first_half['model_type'] = '实战预测b'
        df_preds_second_half['model_type'] = '实战预测b'

        df_preds_first_half['model_name'] = 'TSMixer'
        df_preds_second_half['model_name'] = 'TSMixer'


        df_preds_first_half.columns = ['y_pred', 'time', 'date_type', 'date', 'location', 'direction', 'model_type',
                                       'model_name']
        df_preds_second_half.columns = ['y_pred', 'time', 'date_type', 'date', 'location', 'direction', 'model_type',
                                        'model_name']


        new_column_order = ['date', 'time', 'date_type', 'location', 'direction', 'model_type', 'model_name', 'y_pred']

        df_preds_first_half = df_preds_first_half.reindex(columns=new_column_order)
        df_preds_second_half = df_preds_second_half.reindex(columns=new_column_order)

        predicted_data_b_1 = pd.concat(
            [df_preds_first_half, df_preds_second_half],
            ignore_index=True)

        return predicted_data_b, predicted_data_b_1