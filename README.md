# Audio_mfcc_gram_transformers

This repository contains the code for the paper 'Audio MFCC-gram Transformers for respiratory insufficiency detection in COVID-19'. For questions regarding the implementation contact: marcelomatheusgauy@gmail.com

The file transformer_encoder.py contains the implementation of the transformer encoder modules. As mentioned in the paper it is based on the implementation found at: http://nlp.seas.harvard.edu/2018/04/03/attention.html. Details can be found in the paper

The file train_utils.py contains various training utility functions necessary for running the model. These functions will process the data in batches (load and apply transformations such as mfcc and noise insertion) and run the batches through the model for training.

The files run_training.py, run_pretraining.py and run_finetuning.py are examples of functions that can be created using the train_utils.py functions to train the model on the respiratory insufficiency training dataset (run_training.py), to pretrain the model on Brazilian Portuguese speech (run_pretraining.py), and to finetune a pretrained model on the respiratory insufficiency training dataset (run_finetuning.py)

The files run_main.py and run_pretrain_finetuning_main.py are examples of scripts that use run_training.py (run_main.py) and run_pretraining.py and run_finetuning.py (run_pretrain_finetuning_main.py) to do the whole training loop with repetitions and save the models as well as the logs with the results.

The data used in the paper can be found in the repositories: https://zenodo.org/record/6672451#.Y1f30r7MJkg and https://zenodo.org/record/6794924#.Y1f3577MJkg

For questions regarding how to use the code please contact: marcelomatheusgauy@gmail.com
