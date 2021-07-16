import argparse
import os
import torch
import numpy as np
from yaml.tokens import Token

from glue_utils import convert_examples_to_seq_features, compute_metrics_absa, compute_metrics_absa_arts, ABSAProcessor, processors, output_modes
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer, WEIGHTS_NAME
from absa_layer import BertABSATagger
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from seq_utils import ot2bieos_ts, bio2ot_ts, tag2ts

import mlflow

ALL_MODELS = (
     'bert-base-uncased',
 'bert-large-uncased',
 'bert-base-cased',
 'bert-large-cased',
 'bert-base-multilingual-uncased',
 'bert-base-multilingual-cased',
 'bert-base-chinese',
 'bert-base-german-cased',
 'bert-large-uncased-whole-word-masking',
 'bert-large-cased-whole-word-masking',
 'bert-large-uncased-whole-word-masking-finetuned-squad',
 'bert-large-cased-whole-word-masking-finetuned-squad',
 'bert-base-cased-finetuned-mrpc',
 'bert-base-german-dbmdz-cased',
 'bert-base-german-dbmdz-uncased',
 'xlnet-base-cased',
 'xlnet-large-cased'
)


MODEL_CLASSES = {
    'bert': (BertConfig, BertABSATagger, BertTokenizer),
}

def load_and_cache_examples(args, task, tokenizer, mode='test'):
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        print("cached_features_file:", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels(args.tagging_schema)
        if mode == 'test':
            examples = processor.get_test_examples(args.data_dir, args.tagging_schema)
        else:
            raise Exception("Invalid data mode %s..." % mode)
        features = convert_examples_to_seq_features(examples=examples, label_list=label_list, tokenizer=tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_evaluate_label_ids = [f.evaluate_label_ids for f in features]

    if hasattr(features[0], "sentence_ids"):
        sentence_ids = torch.tensor([f.sentence_ids for f in features], dtype=torch.double)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, sentence_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset, all_evaluate_label_ids

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--absa_home", type=str, required=False, help="Home directory of the trained ABSA model")
    parser.add_argument("--ckpt", type=str, required=False, help="Directory of model checkpoint for evaluation")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The incoming data dir. Should contain the files of test/unseen data")
    parser.add_argument("--task_name", type=str, required=True, help="task name")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--tagging_schema', type=str, default='BIEOS', help="Tagging schema, should be kept same with "
                                                                            "that of ckpt")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--absa_type", default=None, type=str, required=True,
                        help="Downstream absa layer type selected in the list: [linear, gru, san, tfm, crf]")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    
    parser.add_argument('--logged_model', type=str, help="model logged with mlflow")
    parser.add_argument('--run_name', type=str, required=True, help="run name for mlflow")

    args = parser.parse_args()

    return args


def evaluate(args, model, tokenizer, mode, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = args.task_name
    results = {}

    eval_dataset, eval_evaluate_label_ids = load_and_cache_examples(args, eval_task, tokenizer, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    crf_logits, crf_mask = [], []
    ids = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                        'labels':         batch[3],}
            outputs = model(**inputs)
            # logits: (bsz, seq_len, label_size)
            # here the loss is the masked loss
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

            crf_logits.append(logits)
            crf_mask.append(batch[1])
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        # for arts data
        if len(batch) > 4:
            [ids.append(id) for id in batch[4].detach().cpu().numpy()]

    eval_loss = eval_loss / nb_eval_steps
    # argmax operation over the last dimension
    if model.tagger_config.absa_type != 'crf':
        # greedy decoding
        preds = np.argmax(preds, axis=-1)
    else:
        # viterbi decoding for CRF-based model
        crf_logits = torch.cat(crf_logits, dim=0)
        crf_mask = torch.cat(crf_mask, dim=0)
        preds = model.tagger.viterbi_tags(logits=crf_logits, mask=crf_mask)
    # compute macro_f1, precision, recall, micro_f1
    if len(ids) == 0:
        result = compute_metrics_absa(preds, out_label_ids, eval_evaluate_label_ids, args.tagging_schema)
    else:
        result = compute_metrics_absa_arts(preds, out_label_ids, eval_evaluate_label_ids, args.tagging_schema, ids)
    result['eval_loss'] = eval_loss
    results.update(result)

    return results


def main():
    # perform evaluation on single GPU
    args = init_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count()

    mlflow.set_tracking_uri("../mlruns")
    mlflow.set_experiment("bert+"+args.absa_type)
    mlflow.start_run(run_name=args.run_name)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    args.output_mode = output_modes[args.task_name]

    for key in sorted(args.__dict__):
        print("  {}: {}".format(key, args.__dict__[key]))
        mlflow.log_param(key, args.__dict__[key])

    # initialize the pre-trained model
    args.model_type = args.model_type.lower()
    _,_, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    model = mlflow.pytorch.load_model(args.logged_model)
    model.to(args.device)

    test_results = evaluate(args, model, tokenizer, mode='test')
    print(">> test results: ")
    print(test_results)

    mlflow.log_metrics({"f1_macro": test_results['macro-f1'], "f1_micro": test_results['micro-f1'], 
                        "precision": test_results['precision'], "recall": test_results['recall']})
    if "acc_ars" in test_results.keys():
        mlflow.log_metric("acc_ars", test_results["acc_ars"])

    mlflow.end_run()


if __name__ == "__main__":
    main()