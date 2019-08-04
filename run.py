from train import *
from argparse import Namespace
import collections
import os
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
import urllib
from collections import Counter
import string
from torch.utils.data import Dataset, DataLoader
import modules.RNN as NewsModel

def set_seeds(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


# Creating directories
def create_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


args = Namespace(
    seed=1234,
    cuda=True,
    shuffle=True,
    data_file="./data/news.csv",
    split_data_file="./data/split_news.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="models",
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    pretrained_embeddings=None,
    cutoff=25,
    num_epochs=2,
    early_stopping_criteria=5,
    learning_rate=1e-3,
    batch_size=128,
    embedding_dim=100,
    kernels=[3,5],
    num_filters=100,
    rnn_hidden_dim=128,
    hidden_dim=200,
    num_layers=1,
    bidirectional=False,
    dropout_p=0.25,
)



url = "https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/data/news.csv"
response = urllib.request.urlopen(url)
html = response.read()
with open(args.data_file, 'wb') as fp:
    fp.write(html)


df = pd.read_csv(args.data_file, header=0)
df.head()

by_category = collections.defaultdict(list)
for _, row in df.iterrows():
    by_category[row.category].append(row.to_dict())
for category in by_category:
    print ("{0}: {1}".format(category, len(by_category[category])))


final_list = []
for _, item_list in sorted(by_category.items()):
    if args.shuffle:
        np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(args.train_size*n)
    n_val = int(args.val_size*n)
    n_test = int(args.test_size*n)

  # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'
    for item in item_list[n_train+n_val:]:
        item['split'] = 'test'

    # Add to final list
    final_list.extend(item_list)


split_df = pd.DataFrame(final_list)
split_df["split"].value_counts()


def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    text = text.strip()
    return text


split_df.title = split_df.title.apply(preprocess_text)

split_df.to_csv(args.split_data_file, index=False)
split_df.head()

class Vocabulary(object):
    def __init__(self, token_to_idx=None):

        # Token to index
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx

        # Index to token
        self.idx_to_token = {idx: token \
                             for token, idx in self.token_to_idx.items()}

    def to_serializable(self):
        return {'token_to_idx': self.token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def add_tokens(self, tokens):
        return [self.add_token[token] for token in tokens]

    def lookup_token(self, token):
        return self.token_to_idx[token]

    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self.idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self.token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):
        super(SequenceVocabulary, self).__init__(token_to_idx)

        self.mask_token = mask_token
        self.unk_token = unk_token
        self.begin_seq_token = begin_seq_token
        self.end_seq_token = end_seq_token

        self.mask_index = self.add_token(self.mask_token)
        self.unk_index = self.add_token(self.unk_token)
        self.begin_seq_index = self.add_token(self.begin_seq_token)
        self.end_seq_index = self.add_token(self.end_seq_token)

        # Index to token
        self.idx_to_token = {idx: token \
                             for token, idx in self.token_to_idx.items()}

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self.unk_token,
                         'mask_token': self.mask_token,
                         'begin_seq_token': self.begin_seq_token,
                         'end_seq_token': self.end_seq_token})
        return contents

    def lookup_token(self, token):
        return self.token_to_idx.get(token, self.unk_index)

    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError("the index (%d) is not in the SequenceVocabulary" % index)
        return self.idx_to_token[index]

    def __str__(self):
        return "<SequenceVocabulary(size=%d)>" % len(self.token_to_idx)

    def __len__(self):
        return len(self.token_to_idx)


# Get word counts
word_counts = Counter()
for title in split_df.title:
    for token in title.split(" "):
        if token not in string.punctuation:
            word_counts[token] += 1

# Create SequenceVocabulary instance
title_word_vocab = SequenceVocabulary()
for word, word_count in word_counts.items():
    if word_count >= args.cutoff:
        title_word_vocab.add_token(word)
print (title_word_vocab) # __str__
print (len(title_word_vocab)) # __len__
index = title_word_vocab.lookup_token("general")
print (index)
print (title_word_vocab.lookup_index(index))


# Create SequenceVocabulary instance
title_char_vocab = SequenceVocabulary()
for title in split_df.title:
    for token in title:
        title_char_vocab.add_token(token)
print (title_char_vocab) # __str__
print (len(title_char_vocab)) # __len__
index = title_char_vocab.lookup_token("g")
print (index)
print (title_char_vocab.lookup_index(index))


class NewsVectorizer(object):
    def __init__(self, title_word_vocab, title_char_vocab, category_vocab):
        self.title_word_vocab = title_word_vocab
        self.title_char_vocab = title_char_vocab
        self.category_vocab = category_vocab

    def vectorize(self, title):

        # Word-level vectorization
        word_indices = [self.title_word_vocab.lookup_token(token) for token in title.split(" ")]
        word_indices = [self.title_word_vocab.begin_seq_index] + word_indices + \
                       [self.title_word_vocab.end_seq_index]
        title_length = len(word_indices)
        word_vector = np.zeros(title_length, dtype=np.int64)
        word_vector[:len(word_indices)] = word_indices

        # Char-level vectorization
        word_length = max([len(word) for word in title.split(" ")])
        char_vector = np.zeros((len(word_vector), word_length), dtype=np.int64)
        char_vector[0, :] = self.title_word_vocab.mask_index  # <BEGIN>
        char_vector[-1, :] = self.title_word_vocab.mask_index  # <END>
        for i, word in enumerate(title.split(" ")):
            char_vector[i + 1, :len(word)] = [title_char_vocab.lookup_token(char) \
                                              for char in word]  # i+1 b/c of <BEGIN> token

        return word_vector, char_vector, len(word_indices)

    def unvectorize_word_vector(self, word_vector):
        tokens = [self.title_word_vocab.lookup_index(index) for index in word_vector]
        title = " ".join(token for token in tokens)
        return title

    def unvectorize_char_vector(self, char_vector):
        title = ""
        for word_vector in char_vector:
            for index in word_vector:
                if index == self.title_char_vocab.mask_index:
                    break
                title += self.title_char_vocab.lookup_index(index)
            title += " "
        return title

    @classmethod
    def from_dataframe(cls, df, cutoff):

        # Create class vocab
        category_vocab = Vocabulary()
        for category in sorted(set(df.category)):
            category_vocab.add_token(category)

        # Get word counts
        word_counts = Counter()
        for title in df.title:
            for token in title.split(" "):
                word_counts[token] += 1

        # Create title vocab (word level)
        title_word_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                title_word_vocab.add_token(word)

        # Create title vocab (char level)
        title_char_vocab = SequenceVocabulary()
        for title in df.title:
            for token in title:
                title_char_vocab.add_token(token)

        return cls(title_word_vocab, title_char_vocab, category_vocab)

    @classmethod
    def from_serializable(cls, contents):
        title_word_vocab = SequenceVocabulary.from_serializable(contents['title_word_vocab'])
        title_char_vocab = SequenceVocabulary.from_serializable(contents['title_char_vocab'])
        category_vocab = Vocabulary.from_serializable(contents['category_vocab'])
        return cls(title_word_vocab=title_word_vocab,
                   title_char_vocab=title_char_vocab,
                   category_vocab=category_vocab)

    def to_serializable(self):
        return {'title_word_vocab': self.title_word_vocab.to_serializable(),
                'title_char_vocab': self.title_char_vocab.to_serializable(),
                'category_vocab': self.category_vocab.to_serializable()}

# Vectorizer instance
vectorizer = NewsVectorizer.from_dataframe(split_df, cutoff=args.cutoff)
print (vectorizer.title_word_vocab)
print (vectorizer.title_char_vocab)
print (vectorizer.category_vocab)
word_vector, char_vector, title_length = vectorizer.vectorize(preprocess_text(
    "Roger Federer wins the Wimbledon tennis tournament."))
print ("word_vector:", np.shape(word_vector))
print ("char_vector:", np.shape(char_vector))
print ("title_length:", title_length)
print (word_vector)
print (char_vector)
print (vectorizer.unvectorize_word_vector(word_vector))
print (vectorizer.unvectorize_char_vector(char_vector))

# Vocabulary instance
category_vocab = Vocabulary()
for index, row in df.iterrows():
    category_vocab.add_token(row.category)
print (category_vocab) # __str__
print (len(category_vocab)) # __len__
index = category_vocab.lookup_token("Business")
print (index)
print (category_vocab.lookup_index(index))

class NewsDataset(Dataset):
    def __init__(self, df, vectorizer):
        self.df = df
        self.vectorizer = vectorizer

        # Data splits
        self.train_df = self.df[self.df.split=='train']
        self.train_size = len(self.train_df)
        self.val_df = self.df[self.df.split=='val']
        self.val_size = len(self.val_df)
        self.test_df = self.df[self.df.split=='test']
        self.test_size = len(self.test_df)
        self.lookup_dict = {'train': (self.train_df, self.train_size),
                            'val': (self.val_df, self.val_size),
                            'test': (self.test_df, self.test_size)}
        self.set_split('train')

        # Class weights (for imbalances)
        class_counts = df.category.value_counts().to_dict()
        def sort_key(item):
            return self.vectorizer.category_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, split_data_file, cutoff):
        df = pd.read_csv(split_data_file, header=0)
        train_df = df[df.split=='train']
        return cls(df, NewsVectorizer.from_dataframe(train_df, cutoff))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, split_data_file, vectorizer_filepath):
        df = pd.read_csv(split_data_file, header=0)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(df, vectorizer)

    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return NewsVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self.vectorizer.to_serializable(), fp)

    def set_split(self, split="train"):
        self.target_split = split
        self.target_df, self.target_size = self.lookup_dict[split]

    def __str__(self):
        return "<Dataset(split={0}, size={1})".format(
            self.target_split, self.target_size)

    def __len__(self):
        return self.target_size

    def __getitem__(self, index):
        row = self.target_df.iloc[index]
        title_word_vector, title_char_vector, title_length = \
            self.vectorizer.vectorize(row.title)
        category_index = self.vectorizer.category_vocab.lookup_token(row.category)
        return {'title_word_vector': title_word_vector,
                'title_char_vector': title_char_vector,
                'title_length': title_length,
                'category': category_index}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def generate_batches(self, batch_size, collate_fn, shuffle=True,
                         drop_last=False, device="cpu"):
        dataloader = DataLoader(dataset=self, batch_size=batch_size,
                                collate_fn=collate_fn, shuffle=shuffle,
                                drop_last=drop_last)
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict

# Set seeds
set_seeds(seed=args.seed, cuda=args.cuda)

# Create save dir
create_dirs(args.save_dir)

# Expand filepaths
args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")
print("Using CUDA: {}".format(args.cuda))


# Initialization
dataset = NewsDataset.load_dataset_and_make_vectorizer(args.split_data_file, args.cutoff)

dataset.save_vectorizer(args.vectorizer_file)
vectorizer = dataset.vectorizer
model = NewsModel.NewsModel(embedding_dim=args.embedding_dim,
                  num_word_embeddings=len(vectorizer.title_word_vocab),
                  num_char_embeddings=len(vectorizer.title_char_vocab),
                  kernels=args.kernels,
                  num_input_channels=args.embedding_dim,
                  num_output_channels=args.num_filters,
                  rnn_hidden_dim=args.rnn_hidden_dim,
                  hidden_dim=args.hidden_dim,
                  output_dim=len(vectorizer.category_vocab),
                  num_layers=args.num_layers,
                  bidirectional=args.bidirectional,
                  dropout_p=args.dropout_p,
                  word_padding_idx=vectorizer.title_word_vocab.mask_index,
                  char_padding_idx=vectorizer.title_char_vocab.mask_index)
print (model.named_modules)


# Train
trainer = Trainer(dataset=dataset, model=model,
                  model_state_file=args.model_state_file,
                  save_dir=args.save_dir, device=args.device,
                  shuffle=args.shuffle, num_epochs=args.num_epochs,
                  batch_size=args.batch_size, learning_rate=args.learning_rate,
                  early_stopping_criteria=args.early_stopping_criteria)
trainer.run_train_loop()


# Test performance
trainer.run_test_loop()
print("Test loss: {0:.2f}".format(trainer.train_state['test_loss']))
print("Test Accuracy: {0:.1f}%".format(trainer.train_state['test_acc']))

trainer.save_train_state()