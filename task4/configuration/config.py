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

LEVEL1_LABEL_NAME = ['Be creative', 'Be curious', 'Have freedom of thought', 'Be choosing own goals',
                     'Be independent', 'Have freedom of action', 'Have privacy', 'Have an exciting life', 'Have a varied life',
                     'Be daring', 'Have pleasure', 'Be ambitious', 'Have success', 'Be capable', 'Be intellectual', 'Be courageous',
                     'Have influence', 'Have the right to command', 'Have wealth', 'Have social recognition', 'Have a good reputation',
                     'Have a sense of belonging', 'Have good health', 'Have no debts', 'Be neat and tidy', 'Have a comfortable life',
                     'Have a safe country', 'Have a stable society', 'Be respecting traditions', 'Be holding religious faith', 'Be compliant',
                     'Be self-disciplined', 'Be behaving properly', 'Be polite', 'Be honoring elders', 'Be humble',
                     'Have life accepted as is', 'Be helpful', 'Be honest', 'Be forgiving', 'Have the own family secured',
                     'Be loving', 'Be responsible', 'Have loyalty towards friends', 'Have equality', 'Be just', 'Have a world at peace',
                     'Be protecting the environment', 'Have harmony with nature', 'Have a world of beauty', 'Be broadminded',
                     'Have the wisdom to accept others', 'Be logical', 'Have an objective view'
                     ]


# label freq
import pandas as pd
data = pd.read_csv(train_file['labels'], header=0, delimiter='\t', names=LABEL_NAME)
train_num = len(data)
label_freq = [data[column].sum() for column in LABEL_NAME]
label_ratio = [freq/train_num for freq in label_freq]
del data	

data = pd.read_csv(train_file['level1-labels'], header=0, delimiter='\t', names=LEVEL1_LABEL_NAME)
train_num = len(data)
level1_label_freq = [data[column].sum() for column in LEVEL1_LABEL_NAME]
level1_label_ratio = [freq/train_num for freq in level1_label_freq]
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