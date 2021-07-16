import os
import numpy as np
import pickle
import yaml
from data_process.utils import *

def data_process(config):
    mode = config['mode']
    assert mode in ('term', 'category')
    base_path = config['base_path']
    lowercase = config['lowercase']
    train_set = config['train_set']
    rand_state = config['rand_state']

    if train_set:
        if rand_state == 0:
            raw_train_path = os.path.join(base_path, 'train.xml')
            raw_val_path = os.path.join(base_path, 'val.xml')
        else:
            raw_train_path = os.path.join(base_path, 'train_'+str(rand_state)+'.xml')
            raw_val_path = os.path.join(base_path, 'val_'+str(rand_state)+'.xml')
    raw_test_path = os.path.join(base_path, 'test.xml')

    if mode == 'term':
        if train_set:
            train_data = parse_sentence_term(raw_train_path, lowercase=lowercase)
            val_data = parse_sentence_term(raw_val_path, lowercase=lowercase)
        test_data = parse_sentence_term(raw_test_path, lowercase=lowercase)
    else:
        if train_set:
            train_data = parse_sentence_category(raw_train_path, lowercase=lowercase)
            val_data = parse_sentence_category(raw_val_path, lowercase=lowercase)
        test_data = parse_sentence_category(raw_test_path, lowercase=lowercase)

    remove_list = ['conflict']
    if train_set:
        train_data = category_filter(train_data, remove_list)
        val_data = category_filter(val_data, remove_list)
    test_data = category_filter(test_data, remove_list)

    if train_set:
        word2index, index2word = build_vocab(train_data, max_size=config['max_vocab_size'], min_freq=config['min_vocab_freq'])
    else:
        word2index, index2word = build_vocab(test_data, max_size=config['max_vocab_size'], min_freq=config['min_vocab_freq'])

    if not os.path.exists(os.path.join(base_path, 'processed')):
        os.makedirs(os.path.join(base_path, 'processed'))

    if mode == 'term':
        save_term_data(test_data, word2index, os.path.join(base_path, 'processed/test.npz'))
        if train_set:
            if rand_state == 0:
                save_term_data(train_data, word2index, os.path.join(base_path, 'processed/train.npz'))
                save_term_data(val_data, word2index, os.path.join(base_path, 'processed/val.npz'))
            else:
                save_term_data(train_data, word2index, os.path.join(base_path, 'processed/train_'+str(rand_state)+'.npz'))
                save_term_data(val_data, word2index, os.path.join(base_path, 'processed/val_'+str(rand_state)+'.npz'))
    else:
        save_category_data(test_data, word2index, os.path.join(base_path, 'processed/test.npz'))
        if train_set:
            save_category_data(train_data, word2index, os.path.join(base_path, 'processed/train_'+str(rand_state)+'.npz'))
            save_category_data(val_data, word2index, os.path.join(base_path, 'processed/val_'+str(rand_state)+'.npz'))
            

    glove = load_glove(config['glove_path'], len(index2word), word2index)
    sentiment_matrix = load_sentiment_matrix(config['glove_path'], config['sentiment_path'])
    np.save(os.path.join(base_path, 'processed/glove.npy'), glove)
    np.save(os.path.join(base_path, 'processed/sentiment_matrix.npy'), sentiment_matrix)
    with open(os.path.join(base_path, 'processed/word2index.pickle'), 'wb') as handle:
        pickle.dump(word2index, handle)
    with open(os.path.join(base_path, 'processed/index2word.pickle'), 'wb') as handle:
        pickle.dump(index2word, handle)

    analyze = analyze_term if mode == 'term' else analyze_category

    if train_set:
        log = {
        'vocab_size': len(index2word),
        'oov_size': len(word2index) - len(index2word),
        'train_data': analyze(train_data),
        'val_data': analyze(val_data),
        'test_data': analyze(test_data),
        'num_categories': 3
    }
    else:
        log = {
        'vocab_size': len(index2word),
        'oov_size': len(word2index) - len(index2word),
        'test_data': analyze(test_data),
        'num_categories': 3
    }

    if not os.path.exists(os.path.join(base_path, 'log')):
        os.makedirs(os.path.join(base_path, 'log'))
    with open(os.path.join(base_path, 'log/log.yml'), 'w') as handle:
        yaml.safe_dump(log, handle, encoding='utf-8', allow_unicode=True, default_flow_style=False)


def data_process_arts(config):
    mode = config['mode']
    assert mode in ('term', 'category')
    base_path = config['base_path']
    lowercase = config['lowercase']
    train_set = config['train_set']
    rand_state = config['rand_state']

    raw_test_path = os.path.join(base_path, 'test.xml')

    if mode == 'term':
        test_data = parse_sentence_term_arts(raw_test_path, lowercase=lowercase)

    remove_list = ['conflict']
    test_data = category_filter(test_data, remove_list)

    word2index, index2word = build_vocab(test_data, max_size=config['max_vocab_size'], min_freq=config['min_vocab_freq'])

    if not os.path.exists(os.path.join(base_path, 'processed')):
        os.makedirs(os.path.join(base_path, 'processed'))

    if mode == 'term':
        save_term_data_arts(test_data, word2index, os.path.join(base_path, 'processed/test.npz'))

    glove = load_glove(config['glove_path'], len(index2word), word2index)
    sentiment_matrix = load_sentiment_matrix(config['glove_path'], config['sentiment_path'])
    np.save(os.path.join(base_path, 'processed/glove.npy'), glove)
    np.save(os.path.join(base_path, 'processed/sentiment_matrix.npy'), sentiment_matrix)
    with open(os.path.join(base_path, 'processed/word2index.pickle'), 'wb') as handle:
        pickle.dump(word2index, handle)
    with open(os.path.join(base_path, 'processed/index2word.pickle'), 'wb') as handle:
        pickle.dump(index2word, handle)

    analyze = analyze_term_arts if mode == 'term' else analyze_category

    log = {
        'vocab_size': len(index2word),
        'oov_size': len(word2index) - len(index2word),
        'test_data': analyze(test_data),
        'num_categories': 3
    }

    if not os.path.exists(os.path.join(base_path, 'log')):
        os.makedirs(os.path.join(base_path, 'log'))
    with open(os.path.join(base_path, 'log/log.yml'), 'w') as handle:
        yaml.safe_dump(log, handle, encoding='utf-8', allow_unicode=True, default_flow_style=False)
