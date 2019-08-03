import os
from argparse import Namespace
import collections
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer(object):
    def __init__(self, dataset, model, model_state_file, save_dir, device,
                 shuffle, num_epochs, batch_size, learning_rate,
                 early_stopping_criteria):
        self.dataset = dataset
        self.class_weights = dataset.class_weights.to(device)
        self.device = device
        self.model = model.to(device)
        self.save_dir = save_dir
        self.device = device
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss_func = nn.CrossEntropyLoss(self.class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode='min', factor=0.5, patience=1)
        self.train_state = {
            'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'early_stopping_criteria': early_stopping_criteria,
            'learning_rate': learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': model_state_file}

    def update_train_state(self):

        # Verbose
        print(
            "[EPOCH]: {0:02d} | [LR]: {1} | [TRAIN LOSS]: {2:.2f} | [TRAIN ACC]: {3:.1f}% | [VAL LOSS]: {4:.2f} | [VAL ACC]: {5:.1f}%".format(
                self.train_state['epoch_index'], self.train_state['learning_rate'],
                self.train_state['train_loss'][-1], self.train_state['train_acc'][-1],
                self.train_state['val_loss'][-1], self.train_state['val_acc'][-1]))

        # Save one model at least
        if self.train_state['epoch_index'] == 0:
            torch.save(self.model.state_dict(), self.train_state['model_filename'])
            self.train_state['stop_early'] = False

        # Save model if performance improved
        elif self.train_state['epoch_index'] >= 1:
            loss_tm1, loss_t = self.train_state['val_loss'][-2:]

            # If loss worsened
            if loss_t >= self.train_state['early_stopping_best_val']:
                # Update step
                self.train_state['early_stopping_step'] += 1

            # Loss decreased
            else:
                # Save the best model
                if loss_t < self.train_state['early_stopping_best_val']:
                    torch.save(self.model.state_dict(), self.train_state['model_filename'])

                # Reset early stopping step
                self.train_state['early_stopping_step'] = 0

            # Stop early ?
            self.train_state['stop_early'] = self.train_state['early_stopping_step'] \
                                             >= self.train_state['early_stopping_criteria']
        return self.train_state

    def compute_accuracy(self, y_pred, y_target):
        _, y_pred_indices = y_pred.max(dim=1)
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        return n_correct / len(y_pred_indices) * 100

    def pad_word_seq(self, seq, length):
        vector = np.zeros(length, dtype=np.int64)
        vector[:len(seq)] = seq
        vector[len(seq):] = self.dataset.vectorizer.title_word_vocab.mask_index
        return vector

    def pad_char_seq(self, seq, seq_length, word_length):
        vector = np.zeros((seq_length, word_length), dtype=np.int64)
        vector.fill(self.dataset.vectorizer.title_char_vocab.mask_index)
        for i in range(len(seq)):
            char_padding = np.zeros(word_length - len(seq[i]), dtype=np.int64)
            vector[i] = np.concatenate((seq[i], char_padding), axis=None)
        return vector

    def collate_fn(self, batch):

        # Make a deep copy
        batch_copy = copy.deepcopy(batch)
        processed_batch = {"title_word_vector": [], "title_char_vector": [],
                           "title_length": [], "category": []}

        # Max lengths
        get_seq_length = lambda sample: len(sample["title_word_vector"])
        get_word_length = lambda sample: len(sample["title_char_vector"][0])
        max_seq_length = max(map(get_seq_length, batch))
        max_word_length = max(map(get_word_length, batch))

        # Pad
        for i, sample in enumerate(batch_copy):
            padded_word_seq = self.pad_word_seq(
                sample["title_word_vector"], max_seq_length)
            padded_char_seq = self.pad_char_seq(
                sample["title_char_vector"], max_seq_length, max_word_length)
            processed_batch["title_word_vector"].append(padded_word_seq)
            processed_batch["title_char_vector"].append(padded_char_seq)
            processed_batch["title_length"].append(sample["title_length"])
            processed_batch["category"].append(sample["category"])

        # Convert to appropriate tensor types
        processed_batch["title_word_vector"] = torch.LongTensor(
            processed_batch["title_word_vector"])
        processed_batch["title_char_vector"] = torch.LongTensor(
            processed_batch["title_char_vector"])
        processed_batch["title_length"] = torch.LongTensor(
            processed_batch["title_length"])
        processed_batch["category"] = torch.LongTensor(
            processed_batch["category"])

        return processed_batch

    def run_train_loop(self):
        for epoch_index in range(self.num_epochs):
            self.train_state['epoch_index'] = epoch_index

            # Iterate over train dataset

            # initialize batch generator, set loss and acc to 0, set train mode on
            self.dataset.set_split('train')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size, collate_fn=self.collate_fn,
                shuffle=self.shuffle, device=self.device)
            running_loss = 0.0
            running_acc = 0.0
            self.model.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # zero the gradients
                self.optimizer.zero_grad()

                # compute the output
                _, y_pred = self.model(x_word=batch_dict['title_word_vector'],
                                       x_char=batch_dict['title_char_vector'],
                                       x_lengths=batch_dict['title_length'],
                                       device=self.device)

                # compute the loss
                loss = self.loss_func(y_pred, batch_dict['category'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute gradients using loss
                loss.backward()

                # use optimizer to take a gradient step
                self.optimizer.step()

                # compute the accuracy
                acc_t = self.compute_accuracy(y_pred, batch_dict['category'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            self.train_state['train_loss'].append(running_loss)
            self.train_state['train_acc'].append(running_acc)

            # Iterate over val dataset

            # initialize batch generator, set loss and acc to 0, set eval mode on
            self.dataset.set_split('val')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size, collate_fn=self.collate_fn,
                shuffle=self.shuffle, device=self.device)
            running_loss = 0.
            running_acc = 0.
            self.model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                # compute the output
                _, y_pred = self.model(x_word=batch_dict['title_word_vector'],
                                       x_char=batch_dict['title_char_vector'],
                                       x_lengths=batch_dict['title_length'],
                                       device=self.device)

                # compute the loss
                loss = self.loss_func(y_pred, batch_dict['category'])
                loss_t = loss.to("cpu").item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = self.compute_accuracy(y_pred, batch_dict['category'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            self.train_state['val_loss'].append(running_loss)
            self.train_state['val_acc'].append(running_acc)

            self.train_state = self.update_train_state()
            self.scheduler.step(self.train_state['val_loss'][-1])
            if self.train_state['stop_early']:
                break

    def run_test_loop(self):
        # initialize batch generator, set loss and acc to 0, set eval mode on
        self.dataset.set_split('test')
        batch_generator = self.dataset.generate_batches(
            batch_size=self.batch_size, collate_fn=self.collate_fn,
            shuffle=self.shuffle, device=self.device)
        running_loss = 0.0
        running_acc = 0.0
        self.model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            _, y_pred = self.model(x_word=batch_dict['title_word_vector'],
                                   x_char=batch_dict['title_char_vector'],
                                   x_lengths=batch_dict['title_length'],
                                   device=self.device)

            # compute the loss
            loss = self.loss_func(y_pred, batch_dict['category'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = self.compute_accuracy(y_pred, batch_dict['category'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        self.train_state['test_loss'] = running_loss
        self.train_state['test_acc'] = running_acc

    def plot_performance(self):
        # Figure size
        plt.figure(figsize=(15, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(trainer.train_state["train_loss"], label="train")
        plt.plot(trainer.train_state["val_loss"], label="val")
        plt.legend(loc='upper right')

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.plot(trainer.train_state["train_acc"], label="train")
        plt.plot(trainer.train_state["val_acc"], label="val")
        plt.legend(loc='lower right')

        # Save figure
        plt.savefig(os.path.join(self.save_dir, "performance.png"))

        # Show plots
        plt.show()

    def save_train_state(self):
        with open(os.path.join(self.save_dir, "train_state.json"), "w") as fp:
            json.dump(self.train_state, fp)