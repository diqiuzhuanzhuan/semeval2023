# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
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
kfold_data_path = data_path/'kfold'
validate_data_path = data_path/'validate_data'
test_data_path = data_path/'test_data'
badcases_data_path = data_path/'badcases_data'

value_category_file = train_data_path/'value-categories.json'

train_file = {
    'arguments': train_data_path/'arguments-training.tsv',
    'labels': train_data_path/'labels-training.tsv',
    'level1-labels': train_data_path/'level1-labels-training.tsv'
}

validate_file = {
    'arguments': validate_data_path/'arguments-validation.tsv',
    'labels': validate_data_path/'labels-validation.tsv',
    'level1-labels': validate_data_path/'level1-labels-validation.tsv'
}

test_file = {
    'arguments': test_data_path/'arguments-test.tsv',
}

performance_log = log_path/'performance.csv'

LABEL_NAME = ['Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance', 'Power: resources', 'Face', 'Security: personal',
    'Security: societal', 'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring', 'Benevolence: dependability', 'Universalism: concern',
    'Universalism: nature', 'Universalism: tolerance', 'Universalism: objectivity']
#LABEL_NAME = ['Self-direction: thought', 'Self-direction: action', ]


# label freq
import pandas as pd
data = pd.read_csv(train_file['labels'], header=0, delimiter='\t', names=LABEL_NAME)
train_num = len(data)
label_freq = [data[column].sum() for column in LABEL_NAME]
label_ratio = [freq/train_num for freq in label_freq]
del data	


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