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
test_data_path = validate_data_path # wait for the time when testdata is released

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

# label freq
import pandas as pd
label_freq = [734, 1047, 241,
163,
 1130,
 366,
 467,
 311,
 1555,
 1290,
 480,
 971,
 173,
 360,
 1184,
 633,
 1592,
 290,
 572,
 728
 ]

train_num = 4176
	
label_ratio = [
    0.17576628352490423, 0.2507183908045977, 0.05771072796934866, 0.03903256704980843, 0.2705938697318008, 0.08764367816091954,
    0.11182950191570881, 0.07447318007662836, 0.37236590038314177, 0.3089080459770115, 0.11494252873563218, 0.2325191570881226,
	0.0414272030651341, 0.08620689655172414, 0.2835249042145594, 0.15158045977011494, 0.38122605363984674, 0.06944444444444445,
 	0.13697318007662834, 0.1743295019157088
  ]


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