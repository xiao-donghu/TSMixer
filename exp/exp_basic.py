import os
import tensorflow as tf
from models.TSMixer import TSMixer

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TSMixer': TSMixer,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model()

    def _acquire_device(self):
        # 用于获取设备信息，此处仅为示例
        return 'CPU'

    def _build_model(self):
        model_class = self.model_dict.get(self.args.model)
        if model_class is None:
            raise ValueError(f"Model {self.args.model} not found in model_dict.")

        if self.args.use_multi_gpu and self.args.use_gpu:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = model_class(self.args)
        else:
            model = model_class(self.args)

        return model
    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                devices = [f'/device:GPU:{i}' for i in range(len(self.args.devices.split(',')))]
                print(f'Using GPUs: {devices}')
                strategy = tf.distribute.MirroredStrategy(devices)
                return strategy.scope()
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                print(f'Using GPU: /device:GPU:{self.args.gpu}')
                return '/device:GPU:{}'.format(self.args.gpu)
        else:
            print('Using CPU')
            return '/device:CPU:0'

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
