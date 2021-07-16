from __future__ import absolute_import, division, print_function

def get_labels(label_tp_list):
    # create lists of appearing labels

    at_labels = []
    as_labels = []
    for l in label_tp_list:
        if l[0] not in at_labels:
            at_labels.append(l[0])
        if l[1] not in as_labels:
            as_labels.append(l[1])
    if "O" in at_labels:
        at_labels.remove("O")
    if "O" in as_labels:
        as_labels.remove("O")
    at_labels.insert(0, "O")
    as_labels.insert(0, "O")
    at_labels = [l.replace("_", "-") for l in at_labels]
    return at_labels, as_labels

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label_tp=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label_tp = label_tp

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, at_label_id, as_label_id,
                 label_mask, label_mask_X, sentence_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.at_label_id = at_label_id
        self.as_label_id = as_label_id
        self.label_mask = label_mask
        self.label_mask_X = label_mask_X
        if sentence_ids != None:
            self.sentence_ids = sentence_ids

class ATEASCProcessor():
    def __init__(self, file_path, set_type):
        corpus_tp, label_tp_list = self._readfile(file_path)
        examples = self._create_examples(corpus_tp, set_type)
        self.examples = examples
        self.label_tp_list = label_tp_list

    def _readfile(self, filename):
        f = open(filename)
        data = []
        labels = []

        sentence = []
        label = []
        # for each line in the data document
        for line in f:
            line = line.strip()
            line = line.replace("\t", " ")
            # if a line is empty or the first line
            # i.e. after a review has been read completely
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                # if the sentence list is not empty
                # i.e. a review has been stored there
                if len(sentence) > 0:
                    # add sentence, label  as tuple to data
                    data.append((sentence, label))
                    # add label to labels
                    labels += label
                    # clear sentence and label lists
                    sentence = []
                    label = []
                continue
            splits = line.split(' ')
            # add the word to the sentence list
            sentence.append(splits[0])
            # add aspect and polarity to label list
            label.append((splits[-3], splits[-2]))

        # add the last review to the data 
        if len(sentence) > 0:
            data.append((sentence, label))
            labels += label
            sentence = []
            label = []
        return data, labels

    def _create_examples(self, lines, set_type):
        # lines: contains word list and label list as tuple per review
        # set_type: "train", "val", "test"

        examples = []
        #
        for i, (sentence, label_tp) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # create sentences with whitespaces
            text_a = ' '.join(sentence)
            text_b = None
            # create aspect label list
            at_label = [l[0].replace("_", "-") for l in label_tp]
            # create polarity label lsit
            as_label = [l[1].replace("_", "-") for l in label_tp]
            # label is tuple of aspect and polarity tags
            label_tp = (at_label, as_label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label_tp=label_tp))
        return examples

class ATEASCProcessor_arts():
    def __init__(self, file_path, set_type):
        corpus_tp, label_tp_list = self._readfile(file_path)
        examples = self._create_examples(corpus_tp, set_type)
        self.examples = examples
        self.label_tp_list = label_tp_list

    def _readfile(self, filename):
        f = open(filename)
        data = []
        labels = []

        sentence = []
        label = []
        # for each line in the data document
        for line in f:
            line = line.strip()
            line = line.replace("\t", " ")
            # if a line is empty or the first line
            # i.e. after a review has been read completely
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                # if the sentence list is not empty
                # i.e. a review has been stored there
                if len(sentence) > 0:
                    # add sentence, label, id as tuple to data
                    data.append((sentence, label, id))
                    # add label to labels
                    labels += label
                    # clear sentence and label lists
                    sentence = []
                    label = []
                continue
            if " " in line:
                splits = line.split(' ')
                # add the word to the sentence list
                sentence.append(splits[0])
                # add aspect and polarity to label list
                label.append((splits[-3], splits[-2]))
            else:
                id = float(line)

        # add the last review to the data 
        if len(sentence) > 0:
            data.append((sentence, label, id))
            labels += label
            sentence = []
            label = []
        return data, labels

    def _create_examples(self, lines, set_type):
        # lines: contains word list and label list as tuple per review
        # set_type: "train", "val", "test"

        examples = []
        #
        for i, (sentence, label_tp, id) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # create sentences with whitespaces
            text_a = ' '.join(sentence)
            text_b = float(id)
            # create aspect label list
            at_label = [l[0].replace("_", "-") for l in label_tp]
            # create polarity label lsit
            as_label = [l[1].replace("_", "-") for l in label_tp]
            # label is tuple of aspect and polarity tags
            label_tp = (at_label, as_label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label_tp=label_tp))
        return examples

def convert_examples_to_features(examples, label_tp_list, max_seq_length, tokenizer, verbose_logging=False):
    at_label_list, as_label_list = label_tp_list
    at_label_map = {label: i for i, label in enumerate(at_label_list)}
    as_label_map = {label: i for i, label in enumerate(as_label_list)}

    # Note: below contains hard code on "B-AP", "I-AP", and "O"
    assert all(c_str in at_label_map.keys() for c_str in ["B-AP", "I-AP", "O"])

    features = []
    # for each review
    for (ex_index, example) in enumerate(examples):
        # word list
        textlist = example.text_a.split(' ')
        # label list
        label_tp_list = example.label_tp

        if example.text_b != None:
            id = example.text_b
        else:
            id = None

        tokens = []
        at_labels = []
        as_labels = []

        # for each word in a review
        for i, word in enumerate(textlist):
            # create subtokens
            token = tokenizer.tokenize(word)
            # create token list for each word
            tokens.extend(token)

            # aspect label for each word
            label_1 = label_tp_list[0][i]
            # polarity label for each word
            label_2 = label_tp_list[1][i]

            for m in range(len(token)):
                # for first tokens, add the word aspect and polarity label
                if m == 0:
                    at_labels.append(label_1)
                    as_labels.append(label_2)
                # for all other subtokens, add "X" to aspect label and polarity label
                else:
                    at_labels.append("X")
                    as_labels.append(label_2)

        # truncate lists per review
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            at_labels = at_labels[0:(max_seq_length - 2)]
            as_labels = as_labels[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        at_label_ids = []
        as_label_ids = []
        label_mask = []
        label_mask_X = []

        # add [CLS] and corresponding labels at the beginning of each review
        ntokens.append("[CLS]")
        segment_ids.append(0)
        at_label_ids.append(-1)
        as_label_ids.append(-1)
        label_mask.append(-1)
        label_mask_X.append(0)
        
        # each token in a review
        for i, token in enumerate(tokens):
            # add token and 0 as segment id
            ntokens.append(token)
            segment_ids.append(0)

            # for each not-first-token token, add -1 as aspect tag
            if at_labels[i] == "X":
                at_label_ids.append(-1)
                label_mask.append(-1)
                label_mask_X.append(1)
            # for first tokens, convert aspect tag to number
            # label mask contains second to last position of tokens in this review
            else:
                at_label_ids.append(at_label_map[at_labels[i]])
                label_mask.append(len(ntokens)-1)
                label_mask_X.append(0)

            # convert polarity tag to number or -1 if not existing
            if as_labels[i] == "X":
                as_label_ids.append(-1)
            else:
                as_label_ids.append(as_label_map[as_labels[i]])

        # add [SEP] and corresponding labels at the end of the review
        ntokens.append("[SEP]")
        segment_ids.append(0)
        at_label_ids.append(-1)
        as_label_ids.append(-1)
        label_mask.append(-1)
        label_mask_X.append(0)

        # tokens as ids
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        # pad all lists
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            at_label_ids.append(-1)
            as_label_ids.append(-1)
            label_mask.append(-1)
            label_mask_X.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(at_label_ids) == max_seq_length
        assert len(as_label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length
        assert len(label_mask_X) == max_seq_length

        if verbose_logging and ex_index < 5:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("at_label: %s (id = %s)" % (example.label_tp, " ".join([str(x) for x in at_label_ids])))
            print("as_label: %s (id = %s)" % (example.label_tp, " ".join([str(x) for x in as_label_ids])))
            print("label_mask: %s" % (" ".join([str(x) for x in label_mask])))
            print("label_mask_X: %s" % (" ".join([str(x) for x in label_mask_X])))

        # input_ids = tokens2ids
        # input_mask = 1 for existing tokens
        # segment_ids = list of zeros with length max_len
        # at_label_ids = aspecttags2ids
        # as_label_ids = poltags2ids
        # label_mask = len(ntokens)-1 for first tokens, ow -1
        # label_mask_X = 0 for first tokens, ow 1

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                          at_label_id=at_label_ids, as_label_id=as_label_ids,
                          label_mask=label_mask, label_mask_X=label_mask_X, sentence_ids=id))

    return features