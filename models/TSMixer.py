import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, ReLU
from tensorflow.keras.models import Model


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, configs, **kwargs):
        super(ResBlock, self).__init__(**kwargs)

        self.temporal = tf.keras.Sequential([
            tf.keras.layers.Dense(configs.d_model, activation='relu'),
            tf.keras.layers.Dense(configs.seq_len),
            tf.keras.layers.Dropout(configs.dropout)
        ])

        self.channel = tf.keras.Sequential([
            tf.keras.layers.Dense(configs.d_model, activation='relu'),
            tf.keras.layers.Dense(configs.enc_in),
            tf.keras.layers.Dropout(configs.dropout)
        ])

    def call(self, x, training=False):
        # x: [B, L, D]
        x = x + tf.transpose(self.temporal(tf.transpose(x, perm=[0, 2, 1])), perm=[0, 2, 1])
        x = x + self.channel(x, training=training)

        return x

class TSMixer(tf.keras.Model):
    def __init__(self, configs):
        super(TSMixer, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.res_blocks = [ResBlock(configs) for _ in range(configs.e_layers)]
        self.pred_len = configs.pred_len
        self.projection = Dense(configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, training=False):
        # x_enc: [B, L, D]
        for i in range(self.layer):
            x_enc = self.res_blocks[i](x_enc, training=training)
        enc_out = self.projection(tf.transpose(x_enc, perm=[0, 2, 1]))
        enc_out = tf.transpose(enc_out, perm=[0, 2, 1])
        return enc_out

    def call(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, training=False):
        if self.task_name in ['long_term_forecast_a','long_term_forecast_b', 'short_term_forecast','long_term_forecast_b_hour','long_term_forecast_jintang_b','long_term_forecast_jintang_b_hour','long_term_forecast_jintang_a']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, training=training)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
