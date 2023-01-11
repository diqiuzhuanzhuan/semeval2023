# semeval2023


### Environment
python=3.8
All the depended packages are listed in requirements.txt

### The structure of directory
```
└── task4
    ├── README.md
    ├── apps
    ├── bin
    ├── configuration
    ├── data
    ├── data_man
    ├── logs
    ├── metric
    ├── modeling
    └── papers
```

### some experiments
At first stage, we used all the fields to encode information with transformers-based models. But we found just precise is vital for the model performance.

The best encoder model is roberta-large, compared with bert-base-uncased, bert-large-uncased, albert, etc.


### submission records
```
└── task4
    ├── apps
    ├── bin
    ├── configuration
    ├── data
    │   ├── kfold
    │   ├── submission            -------- 提交文件所在的目录
    │   │   └── labels_1.tsv        
    │   ├── test_data
    │   ├── training_data
    │   └── validate_data
    ├── data_man
    ├── logs
    ├── metric
    ├── modeling
    └── papers
```
1. 文件名：
   labels_1.tsv
2. 实验参数：
baseline_argument_data_module,rewrite_argument_dataset,class_balanced_loss_argument_model,roberta-large,baseline_argument_data_module,rewrite_argument_dataset,class_balanced_loss_argument_model,roberta-large,16,35,val_f1
进行8折交叉验证的结果。