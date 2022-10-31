"""Semeval2023 datasets."""

import csv
import json
import os

import datasets
from task4.configuration import config as CONFIG

import numpy as np
import pandas as pd

_CITATION = """
"""

# You can copy an official description
_DESCRIPTION = """
semeval2023 dataset.
"""

_HOMEPAGE = ""

_LICENSE = ""

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
    "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
}


class SemEvalDataset(datasets.GeneratorBasedBuilder):
    """Semeval 2023."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="original", version=VERSION,
                               description="This part of my dataset covers a first domain"),
        datasets.BuilderConfig(name="enhanced", version=VERSION,
                               description="This part of my dataset covers a second domain"),
    ]

    DEFAULT_CONFIG_NAME = "original"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        # if self.config.name == "first_domain":  # This is the name of the configuration selected in BUILDER_CONFIGS above
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option1": datasets.Value("string"),
        #             "answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        # else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option2": datasets.Value("string"),
        #             "second_domain_answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        features = datasets.Features({
            'id': datasets.Value('string'),
            'text': datasets.Value("string"),
            'label': datasets.Sequence(datasets.ClassLabel(names=['0', '1']))
        })
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": CONFIG.train_file,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": CONFIG.validate_file,
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": CONFIG.validate_file,
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        from task4.configuration.config import logging
        logging.info(filepath)
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        arguments_df = pd.read_csv(filepath['arguments'], sep='\t')
        labels_df = pd.read_csv(filepath['labels'], sep='\t')
        text_list = arguments_df.Premise.tolist()
        labels_list = labels_df.drop(columns=['Argument ID']).values.tolist()
        if split == 'train' and self.config.name == 'enhanced':
            augmented_texts = []
            augmented_labels = []
            for i, j in np.random.choice(np.arange(len(labels_list)), size=2000, replace=True).reshape(-1, 2).tolist():
                augmented_texts.append(text_list[i] + '\n' + text_list[j])
                augmented_labels.append(((np.array(labels_list[i]) + np.array(labels_list[j])
                                          ) > 0).astype(int).tolist()
                                        )
            text_list.extend(augmented_texts)
            labels_list.extend(augmented_labels)

        for idx, (text, label) in enumerate(zip(text_list, labels_list)):
            yield str(idx), {
                'id': str(idx),
                'text': text,
                'label': [str(int(l)) for l in label]
            }
