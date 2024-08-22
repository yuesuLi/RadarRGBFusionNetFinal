import logging
import datetime

class normal_logger():
    def __init__(self, logdir, print_on=True):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        file_handler = logging.FileHandler('{}/{}.txt'.format(logdir, str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.print_on = print_on

    def change_print_on(self, print_on):
        self.print_on = print_on

    def log(self, text, print_on=False):
        self.logger.info(text)
        if self.print_on or print_on:
            print(text)

