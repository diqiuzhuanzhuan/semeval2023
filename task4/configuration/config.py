# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
from cmath import log
import logging
import os
import pathlib
import colorlog
from colorlog import ColoredFormatter


root_path = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = root_path/'data'
log_path = root_path/'logs'
output_path = root_path/'output'

train_data_path = data_path/'training_data'
validate_data_path = data_path/'validate_data'
test_data_path = data_path/'test_data'


train_file = {
    'arguments': train_data_path/'arguments-training.tsv',
    'labels': train_data_path/'labels-training.tsv',
    'level1-labels': train_data_path/'level1-labels-training.tsv'
}


validate_file = {
    'arguments': validate_data_path/'arguments-validate.tsv',
    'labels': validate_data_path/'labels-validate.tsv',
    'level1-labels': validate_data_path/'level1-labels-validate.tsv'
}

test_file = {
    'arguments': test_data_path/'arguments-training.tsv',
    'labels': test_data_path/'labels-training.tsv',
    'level1-labels': test_data_path/'level1-labels-training.tsv'
}

performance_log = log_path/'performance.csv'


###### for log ######
formatter = ColoredFormatter(
	"%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
	datefmt=None,
	reset=True,
	log_colors={
		'DEBUG':    'cyan',
		'INFO':     'green',
		'WARNING':  'yellow',
		'ERROR':    'red',
		'CRITICAL': 'red,bg_white',
	},
	secondary_log_colors={},
	style='%'
)

handler = colorlog.StreamHandler()
handler.setFormatter(formatter)
logging = colorlog.getLogger('example')
logging.addHandler(handler)
logging.setLevel(colorlog.DEBUG)