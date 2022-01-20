import logging

from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, id, text, label=None):
        """Constructs a InputExample.
        Args:
            id: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.id = id
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        _, input_ids, input_mask, segment_ids = choices_features[0]
        self.choices_features = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
        }
        self.label = label


def read_samples(df):
    examples = []
    for val in df[["id", "text", "target"]].values:
        examples.append(InputExample(id=val[0], text=val[1], label=val[2]))
    return examples, df


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_features(examples, tokenizer, max_seq_length, is_training):
    features = []
    for example_index, example in enumerate(examples):

        text = tokenizer.tokenize(example.text)
        MAX_TEXT_LEN = max_seq_length - 2
        text = text[:MAX_TEXT_LEN]

        choices_features = []

        tokens = ["[CLS]"] + text + ["[SEP]"]
        segment_ids = [0] * (len(text) + 2)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length
        choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index < 1 and is_training:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example_index))
            logger.info("id: {}".format(example.id))
            logger.info("tokens: {}".format(" ".join(tokens).replace("\u2581", "_")))
            logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
            logger.info("input_mask: {}".format(len(input_mask)))
            logger.info("segment_ids: {}".format(len(segment_ids)))
            logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.id, choices_features=choices_features, label=label
            )
        )
    return features


def select_field(features, field):
    return [feature.choices_features[field] for feature in features]


def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1
