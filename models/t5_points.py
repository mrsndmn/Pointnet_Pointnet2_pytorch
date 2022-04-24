
import torch
import torch.nn as nn

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from transformers.models.t5.modeling_t5 import T5ForConditionalGenerationExtraEmbeddings, T5EncoderWithExtraEmbeddings

model_checkpoint = "t5-small"
print("model_checkpoint", model_checkpoint)

model = T5ForConditionalGenerationExtraEmbeddings.from_pretrained(model_checkpoint)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


from datasets import load_metric
metric = load_metric("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"accuracy": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
parser.add_argument('--log_dir', type=str, default='pointnet2_msg_normals', help='Experiment root')
parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
parser.add_argument('--split', type=str, default='test', help='train/test dataset split to evaluate')
parser.add_argument('--save_embeddings', action='store_true', default=False, help='save points embeddings for each batch')
argparse_args = parser.parse_args([])

test_dataset = ModelNetDataLoader(root='data/modelnet40_normal_resampled', args=argparse_args, split='test', process_data=False)

from torch.utils.data import Dataset
import torch

from glob import glob
import pickle

class PointNet2EmbeddingsDataset(Dataset):

    def __init__(self, modelnet_dataset, embedding_level=-1, vote_idx=-1):

        self.modelnet_dataset = modelnet_dataset

        points_embeddings_files = glob(f'{self.modelnet_dataset.root}/points_embeddings/{self.modelnet_dataset.split}/*.pkl')

        self.embeddings_files = sorted(points_embeddings_files, key=lambda x: int(x.split('.')[0].split('/')[-1]))

        embedding_file = self.load_embedding_file(0)
        self.votes_num = len(embedding_file)

        self.modelnet_batch_size = embedding_file[-1][-1]['points'].shape[0]

        self.embedding_level = embedding_level
        self.vote_idx = vote_idx

    def __len__(self):
        return len(self.modelnet_dataset)

    def load_embedding_file(self, file_idx):
        filename = self.embeddings_files[file_idx]


        with open(filename, 'rb') as f:
            batch_embeddings = pickle.load(f)

        return batch_embeddings

    def _get_modelnet_label(self, idx):
        return self.modelnet_dataset.classes[self.modelnet_dataset.datapath[ idx ][0]]


    def __getitem__(self, idx):


        label = self._get_modelnet_label(idx)
        all_batch_embedding_from_file = self.load_embedding_file(idx // self.modelnet_batch_size)


        local_batch_idx = idx % self.modelnet_batch_size

        vote = all_batch_embedding_from_file[self.vote_idx]
        vote_embeddings = vote[self.embedding_level]['points']

        model_embeddings = vote_embeddings[ local_batch_idx, :, : ]

        return preprocess_function({
            "label": label,
            "label_name": self.modelnet_dataset.cat[label],
            "embeddings": model_embeddings
        })

prefix = 'what is it?'

max_input_length = 128
max_target_length = 128
source_field = ""
target_field = "label_name"

embeddings_3d_length = 1

def preprocess_function(example):
    inputs = prefix
    targets = example['label_name']
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    # extra_embeddings_positions
    model_inputs["extra_embeddings"] = example["embeddings"].squeeze().detach().cpu().numpy()
    model_inputs['extra_embeddings_positions'] = len(model_inputs['attention_mask'])

    model_inputs['input_ids'].extend([ tokenizer.pad_token_id ] * embeddings_3d_length)

    return model_inputs


embedds = PointNet2EmbeddingsDataset(test_dataset)

batch_size = 16
model_name = model_checkpoint.split("/")[-1]
hf_args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-3d-classification",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model,
    hf_args,
    train_dataset=embedds,
    eval_dataset=embedds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()