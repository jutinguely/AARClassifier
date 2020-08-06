"""
RoBERTa Model Implementation
============================

Structure:
==========
- *Configuration*: instantiate the config.json file containing the hyper-parameters.
- *class* RoBERTa: init, train, and predict new sentences.

"""
import json
import time
import random
import numpy as np
import pandas as pd
import sys
import torch
import os
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


class Configuration(object):
    """
    Instantiation of Hyper-parameters for ease of changes between experiments
    """

    def __init__(self, config):
        os.chdir(os.getcwd())
        file = json.load(open(str(config), "r"))
        self.model = file["model"]
        self.epochs = file["epochs"]
        self.batch_size = file["batch_size"]
        self.max_len = file["max_len"]
        self.learning_rate = file["learning_rate"]
        self.seed = 42
        self.train_set_rate = file["train_set_rate"]
        self.num_reviews_per_class = file["num_reviews_per_class"]
        self.early_stopping = file["early_stopping"]


class RoBERTa(object):
    """
    Inspired from the paper and the official repo example.
    """

    def __init__(self, configuration):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.config = configuration
        # initialize all seeds
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # use of default tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.model, do_lower_case=False)
        # default classifier (need to specify num_labels!)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.config.model, num_labels=5, output_attentions=False, output_hidden_states=False)
        self.model.to(self.device)
        # specify an optimizer with a customizable learning rate lr
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, eps=1e-8)

    def init_sets(self):
        """
        :return: the encoded train and valid embeddings.
        """
        # Read data and do optional preprocessing
        data_file = open("Apps_for_Android_5.json", "r")
        reviews = []
        # keep track of balance of classes
        classes_len = [0, 0, 0, 0, 0] # goal every length should be 44k (max with balancing)
        for i, line in enumerate(data_file.readlines()):
            review = json.loads(line.replace("\n", ""))
            # for ease, downgrade all scores from 1 to be [0..4]
            review["overall"] = int(review["overall"]) - 1
            if classes_len[review["overall"]] < self.config.num_reviews_per_class:
                reviews.append(review)
                classes_len[review["overall"]] += 1
        # create pandas dataframe
        random.shuffle(reviews)
        train = pd.DataFrame(reviews)
        # split reviews and scores
        scores = train["overall"]
        reviews = train["reviewText"]
        # Tokenize the text reviews
        input_ids = []
        attention_masks = []
        # iterate over all training reviews for encoding with the roberta default tokenizer
        for review in reviews:
            # encoder found on main repo, specify max length of review (in words)
            encoded_dict = self.tokenizer.encode_plus(
                review,
                add_special_tokens=True,
                max_length=self.config.max_len,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            # update DS
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])
        # save the dataset vocabulary
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(scores)

        # make data loader for the train and validation sets
        dataset = TensorDataset(input_ids, attention_masks, labels)
        train_size = int(self.config.train_set_rate * len(dataset))
        valid_size = len(dataset) - train_size
        # split randomly in function of sizes
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        # DS to pop sets as batches
        train_dataloader = DataLoader(
            train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.config.batch_size
        )
        valid_dataloader = DataLoader(
            valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=self.config.batch_size
        )
        return train_dataloader, valid_dataloader

    def train(self, train_loader, valid_loader):
        """
        0.9 of DS size is for training rest for validation
        :param train_loader: DS for train set
        :param valid_loader: DS for validation set
        :return: stats for each epoch
        """
        # total training is size(trainset)/size(batch) * num_epochs
        total_steps = len(train_loader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_training_steps=total_steps, num_warmup_steps=0
        )
        train_stats = []
        last_valid_acc = 0
        for epoch in range(0, self.config.epochs):
            print("epoch={}".format(epoch))
            device = self.device
            epoch_training_loss = 0
            self.model.train()
            for step, batch in enumerate(tqdm(train_loader, position=0, file=sys.stdout, leave=True)):
                # pop relevant data
                batch_input_ids = batch[0].to(device)
                batch_attention_masks = batch[1].to(device)
                batch_labels = batch[2].to(device)
                self.model.zero_grad()
                # get the logits from the model, the model forward the batch into the augmented model
                loss, logits = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_masks,
                    labels=batch_labels
                )
                # get the training loss of run
                epoch_training_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
            # at the end of epoch average the training loss
            avg_train_loss = epoch_training_loss / len(self.train_loader)
            avg_train_loss /= self.config.epochs
            # 2. start validation (same than training)
            # call to bring model in evaluation mode (no training anymore)
            self.model.eval()
            epoch_eval_acc = 0
            epochs_eval_loss = 0
            eval_steps = 0
            print("starting validation phase...")
            for batch in tqdm(valid_loader, position=0, file=sys.stdout, leave=True):
                # basically same as before
                batch_input_ids = batch[0].to(device)
                batch_attention_masks = batch[1].to(device)
                batch_labels = batch[2].to(device)
                with torch.no_grad():
                    loss, logits = self.model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_masks,
                        labels=batch_labels,
                    )
                epochs_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                labels_ids = batch_labels.to("cpu").numpy()
                # accuracy calculation
                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = labels_ids.flatten()
                epoch_eval_acc += np.sum(pred_flat == labels_flat) / len(labels_flat)
            avg_valid_acc = epoch_eval_acc / len(valid_loader)
            avg_valid_loss = epochs_eval_loss / len(valid_loader)
            print("valid acc: {}".format(avg_valid_acc))
            print("valid loss: {}".format(avg_valid_loss))
            train_stats.append({
                "epoch": epoch,
                "train loss": avg_train_loss,
                "valid loss": avg_valid_loss,
                "valid acc": avg_valid_acc,
            })
            # early stopping to prevent overfitting
            if self.config.early_stopping == "True":
                if last_valid_acc > avg_valid_acc:
                    return train_stats
                else:
                    last_valid_acc = avg_valid_acc
        return train_stats

    def predict(self, sentences):
        """
        custom sentences
        :param sentences: list of sentences
        :return: list of predictions
        """
        # identically bring in eval mode
        input_ids = []
        attention_masks = []
        for sentence in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.config.max_len,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        batch_size = self.config.batch_size
        prediction_data = TensorDataset(input_ids, attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(
            prediction_data, sampler=prediction_sampler, batch_size=batch_size
        )
        # Predict test data
        self.model.eval()
        predictions, true_labels = [], []
        for batch in tqdm(prediction_dataloader, position=0, file=sys.stdout, leave=True):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                outputs = self.model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)
        pred = []
        for i in predictions:
            pred += list(np.argmax(i, axis=1))
        return pred


if __name__ == '__main__':
    if len(sys.argv) < 1:
        raise Exception("specify the name of your configuration; e.g. python main.py roberta_android.json")
    configuration = Configuration(sys.argv[1])
    roberta = RoBERTa(configuration)
    # init sets
    train_loader, valid_loader = roberta.init_sets()
    # call for training and validation
    stats = roberta.train(train_loader, valid_loader)
    # displays stats (accuracies and losses of each phases of each epoch)
    for stat in stats:
        print(json.dumps(stat, indent=4))
