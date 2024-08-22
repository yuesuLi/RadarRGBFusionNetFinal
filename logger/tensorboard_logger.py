import os
from torch.utils.tensorboard import SummaryWriter

class tensorboard_logger():
    def __init__(self, logdir, isTrain):
        if isTrain:
            os.mkdir(os.path.join(logdir, 'train'))
            self.writer = SummaryWriter(os.path.join(logdir, 'train'))
        else:
            os.mkdir(os.path.join(logdir, 'val'))
            self.writer = SummaryWriter(os.path.join(logdir, 'val'))


    def __del__(self):
        self.writer.close()

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def figure_writer(self, tag, val, step):
        self.writer.add_figure(tag, val, step)
        self.writer.flush()

