# -*- coding: utf-8 -*-
# file: data_utils.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.


import os

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, sentence_label=None, aspect_label=None, polarity=None, sentence_id=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.sentence_label = sentence_label
        self.aspect_label = aspect_label
        self.polarity = polarity
        if sentence_id != None:
            self.sentence_id = sentence_id

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_spc, input_mask, segment_ids, label_id, polarities=None, valid_ids=None, label_mask=None, sentence_id = None):
        self.input_ids_spc = input_ids_spc
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.polarities = polarities
        if sentence_id != None:
            self.sentence_id = sentence_id

def readfile(filename):
    '''
    read file
    '''
    f = open(filename, encoding='utf8')
    data = []
    sentence = []
    tag= []
    polarity = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0 ]=="\n":
            if len(sentence) > 0:
                data.append((sentence, tag, polarity))
                sentence = []
                tag = []
                polarity = []
            continue
        splits = line.split(' ')
        if len(splits) != 3:
            print('warning! detected error line(s) in input file:{}'.format(line))
        sentence.append(splits[0])
        tag.append(splits[-2])
        polarity.append(int(splits[-1][:-1]))

    if len(sentence) > 0:
        data.append((sentence, tag, polarity))
    return data

def readfile_arts(filename):
    '''
    read arts file
    '''
    f = open(filename, encoding='utf8')
    data = []
    sentence = []
    tag= []
    polarity = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0 ]=="\n":
            if len(sentence) > 0:
                data.append((sentence, tag, polarity, id))
                sentence = []
                tag = []
                polarity = []
            continue

        if " " in line:        
            splits = line.split(' ')
            if len(splits) != 3:
                print('warning! detected error line(s) in input file:{}'.format(line))
            sentence.append(splits[0])
            tag.append(splits[-2])
            polarity.append(int(splits[-1][:-1]))
        else:
            id = float(line)

    if len(sentence) > 0:
        data.append((sentence, tag, polarity, id))
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        if "ARTS" in input_file:
            return readfile_arts(input_file)
        else:
            return readfile(input_file)


class ATEPCProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir, train_seed):
        """See base class."""
        if 'LAP' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train_"+str(train_seed)+".dat")), "train")
        elif 'REST' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train_"+str(train_seed)+".dat")), "train")
        elif 'MAMS' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.dat")), "train")
        
    def get_dev_examples(self, data_dir, train_seed):
        """See base class."""
        if 'LAP' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "val_"+str(train_seed)+".dat")), "val")
        elif 'REST' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "val_"+str(train_seed)+".dat")), "val")
        elif 'MAMS' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "val.dat")), "val")

    def get_test_examples(self, data_dir):
        """See base class."""
        if 'ARTS-LAP' in data_dir:
            return self._create_examples_arts(
                self._read_tsv(os.path.join(data_dir, "test.dat")), "test")
        elif 'ARTS-REST' in data_dir:
            return self._create_examples_arts(
                self._read_tsv(os.path.join(data_dir, "test.dat")), "test")
        elif 'LAP' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.dat")), "test")
        elif 'REST' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.dat")), "test")
        elif 'MAMS' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.dat")), "test")


    def get_labels(self):
        return ["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, tag, polarity) in enumerate(lines):
            # aspect = ['[SEP]']
            # aspect_tag = ['O']
            aspect = []
            aspect_tag = []
            aspect_polarity = [-1]
            # for all non-aspect words, add aspects, tags, -1 to empty lists
            for w, t, p in zip(sentence, tag, polarity):
                if p != -1:
                    aspect.append(w)
                    aspect_tag.append(t)
                    aspect_polarity.append(-1)
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = aspect

            # polarity + len(aspect polarities)*[-1]
            polarity.extend(aspect_polarity)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, sentence_label=tag,
                                         aspect_label=aspect_tag, polarity=polarity))
        return examples


    def _create_examples_arts(self, lines, set_type):
        examples = []
        for i, (sentence, tag, polarity, id) in enumerate(lines):
            # aspect = ['[SEP]']
            # aspect_tag = ['O']
            aspect = []
            aspect_tag = []
            aspect_polarity = [-1]
            # for all non-aspect words, add aspects, tags, -1 to empty lists
            for w, t, p in zip(sentence, tag, polarity):
                if p != -1:
                    aspect.append(w)
                    aspect_tag.append(t)
                    aspect_polarity.append(-1)
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = aspect
            sentence_id = id

            # polarity + len(aspect polarities)*[-1]
            polarity.extend(aspect_polarity)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, sentence_label=tag,
                                         aspect_label=aspect_tag, polarity=polarity, sentence_id=sentence_id))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        # list of tokens
        text_spc_tokens = example.text_a
        # all tokens marked as aspects
        aspect_tokens = example.text_b
        # list of tags for one sentence
        sentence_label = example.sentence_label
        # list of tags for aspects only
        aspect_label = example.aspect_label
        # list of polarities for one sentence (-1) plus aspect polarities
        polaritiylist = example.polarity

        if hasattr(example, "sentence_id"):
            sentence_id = example.sentence_id
        else:
            sentence_id = None

        tokens = []
        labels = []
        polarities = []
        valid = []
        label_mask = []

        # add [SEP] and aspect tokens to token list
        text_spc_tokens.extend(['[SEP]'])
        text_spc_tokens.extend(aspect_tokens)
        # call this enum_tokens
        enum_tokens = text_spc_tokens

        # add [SEP] and aspect labels to tags list
        sentence_label.extend(['[SEP]'])
        # sentence_label.extend(['O'])
        sentence_label.extend(aspect_label)
        # call this label_list
        label_lists = sentence_label
        # if len(enum_tokens) != len(label_lists):
        #     print(enum_tokens)
        #     print(label_lists)

        for i, word in enumerate(enum_tokens):
            # tokenize each word into subtokens
            token = tokenizer.tokenize(word)
            # build a list out of them
            tokens.extend(token)
            # get label and polarity for each word
            label_1 = label_lists[i]
            polarity_1 = polaritiylist[i]
            # add them to label and polarity list
            # for the first subtoken, only
            # "valid" indicates whether a token is the first of a word
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    polarities.append(polarity_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)

        # truncate too long sentences
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            polarities = polarities[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []

        # add [CLS] at the beginning of each sentence
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        # label_ids.append(label_map["O"])

        # add each token
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            # ??
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        
        # add [SEP] at the end of each sentence
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        # label_ids.append(label_map["O"])

        # convert tokens to ids
        input_ids_spc = tokenizer.convert_tokens_to_ids(ntokens)

        # create masks
        input_mask = [1] * len(input_ids_spc)
        label_mask = [1] * len(label_ids)
        # pad until max_seq_length
        while len(input_ids_spc) < max_seq_length:
            input_ids_spc.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        while len(polarities) < max_seq_length:
            polarities.append(-1)

        assert len(input_ids_spc) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        #if ex_index < 5:
        #    print("*** Example ***")
        #    print("guid: %s" % (example.guid))
        #    # [CLS] tokens including subtokens [SEP] aspect tokens [SEP]
        #    print("tokens: %s" % " ".join([str(x) for x in ntokens]))
        #    # word embeddings of ntokens, i.e. [CLS] token embs [SEP] aspect tok embs [SEP]
        #    print("input_ids: %s" % " ".join([str(x) for x in input_ids_spc]))
        #    # 1 for actual tokens, 0 ow
        #    print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #    # only 0s
        #    print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #    # one label per token followed by [SEP] asp token label [SEP]
        #    print("sentence label: %s" % (example.sentence_label))
        #    # aspect token labels only
        #    print("aspect label: %s" % (example.aspect_label))
        #    # sentence labels encoded in numbers, followed by 0s for non-tokens
        #    print("label ids: %s" % (label_ids))
        #    # polarity per token, for those without actual polarities -1
        #    print("polarities: %s" % " ".join([str(x) for x in polarities]))
        #    # 1 for single-word tokens, for not-first subtokens 0
        #    print("valid ids: %s" % " ".join([str(x) for x in valid]))
        #    # 1 for actual tokens, 0 ow
        #    print("label mask: %s" % " ".join([str(x) for x in label_mask]))
        
        # input_ids_spc = np.array(input_ids_spc)
        # label_ids = np.array(label_ids)
        # labels = np.array(labels)
        # valid = np.array(valid)

        features.append(
            InputFeatures(input_ids_spc=input_ids_spc,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          polarities=polarities,
                          valid_ids=valid,
                          label_mask=label_mask,
                          sentence_id=sentence_id))
    return features
