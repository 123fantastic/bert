# coding=utf-8
"""Fine-tune BERT for dynasty classification on 古诗.csv."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import os
import random
from collections import OrderedDict, defaultdict

import run_classifier
import tokenization
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_file", None,
    "Path to the poetry CSV file. Expected columns: title, author, dynasty, content.")
flags.DEFINE_float(
    "eval_ratio", 0.2,
    "Evaluation split ratio used when do_eval=True. Samples with unique labels stay in train.")
flags.DEFINE_integer(
    "split_seed", 42,
    "Random seed for train/dev split.")
flags.DEFINE_string(
    "text_col", "content",
    "CSV column used as the main text input.")
flags.DEFINE_bool(
    "use_title", True,
    "Whether to prepend the poem title to the input text.")
flags.DEFINE_bool(
    "use_author", True,
    "Whether to prepend the author name to the input text.")
flags.DEFINE_string(
    "predict_text", None,
    "A single poem text for inference when do_predict=True.")
flags.DEFINE_string(
    "predict_title", "",
    "Optional title used together with predict_text.")
flags.DEFINE_string(
    "predict_author", "",
    "Optional author used together with predict_text.")


class PoetryProcessor(run_classifier.DataProcessor):
  """Processor for poem dynasty classification."""

  def __init__(self, data_file, text_col, use_title, use_author, eval_ratio, split_seed):
    self.data_file = data_file
    self.text_col = text_col
    self.use_title = use_title
    self.use_author = use_author
    self.eval_ratio = eval_ratio
    self.split_seed = split_seed
    self.rows = self._load_rows()
    self.train_rows, self.dev_rows = self._split_rows(self.rows)

  def get_train_examples(self, _):
    return self._create_examples(self.train_rows, "train")

  def get_dev_examples(self, _):
    return self._create_examples(self.dev_rows, "dev")

  def get_test_examples(self, _):
    return self._create_examples(self.dev_rows, "test")

  def get_labels(self):
    labels = sorted({row["dynasty"] for row in self.rows if row.get("dynasty")})
    return labels

  def build_predict_example(self, title, author, text):
    row = {
        "title": title or "",
        "author": author or "",
        self.text_col: text or "",
        "dynasty": self.get_labels()[0],
    }
    return self._create_examples([row], "predict")[0]

  def _load_rows(self):
    rows = []
    with tf.gfile.Open(self.data_file, "r") as file_obj:
      reader = csv.DictReader(file_obj)
      required_fields = {"dynasty", self.text_col}
      missing_fields = required_fields - set(reader.fieldnames or [])
      if missing_fields:
        raise ValueError("CSV missing fields: %s" % ", ".join(sorted(missing_fields)))
      for row in reader:
        text = (row.get(self.text_col) or "").strip()
        label = (row.get("dynasty") or "").strip()
        if text and label:
          rows.append(row)
    if not rows:
      raise ValueError("No valid rows found in CSV file.")
    return rows

  def _split_rows(self, rows):
    grouped = defaultdict(list)
    for row in rows:
      grouped[row["dynasty"]].append(row)

    rng = random.Random(self.split_seed)
    train_rows = []
    dev_rows = []

    for label_rows in grouped.values():
      items = list(label_rows)
      rng.shuffle(items)
      if len(items) < 2 or self.eval_ratio <= 0:
        train_rows.extend(items)
        continue
      dev_count = max(1, int(round(len(items) * self.eval_ratio)))
      dev_count = min(dev_count, len(items) - 1)
      dev_rows.extend(items[:dev_count])
      train_rows.extend(items[dev_count:])

    if not dev_rows:
      dev_rows = train_rows[:]
    return train_rows, dev_rows

  def _compose_text(self, row):
    parts = []
    if self.use_title and row.get("title"):
      parts.append(tokenization.convert_to_unicode(row["title"]))
    if self.use_author and row.get("author"):
      parts.append(tokenization.convert_to_unicode(row["author"]))
    parts.append(tokenization.convert_to_unicode(row[self.text_col]))
    return " [SEP] ".join([part for part in parts if part])

  def _create_examples(self, rows, set_type):
    examples = []
    for index, row in enumerate(rows):
      guid = "%s-%d" % (set_type, index)
      text_a = self._compose_text(row)
      label = tokenization.convert_to_unicode(row["dynasty"])
      examples.append(run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def convert_single_prediction(example, label_list, max_seq_length, tokenizer):
  feature = run_classifier.convert_single_example(0, example, label_list, max_seq_length, tokenizer)
  return {
      "input_ids": [feature.input_ids],
      "input_mask": [feature.input_mask],
      "segment_ids": [feature.segment_ids],
      "label_ids": [feature.label_id],
      "is_real_example": [int(feature.is_real_example)],
  }


def prediction_input_fn_builder(feature_dict, seq_length):
  def input_fn(params):
    batch_size = params["batch_size"]
    dataset = tf.data.Dataset.from_tensor_slices({
        "input_ids": tf.constant(feature_dict["input_ids"], shape=[1, seq_length], dtype=tf.int32),
        "input_mask": tf.constant(feature_dict["input_mask"], shape=[1, seq_length], dtype=tf.int32),
        "segment_ids": tf.constant(feature_dict["segment_ids"], shape=[1, seq_length], dtype=tf.int32),
        "label_ids": tf.constant(feature_dict["label_ids"], shape=[1], dtype=tf.int32),
        "is_real_example": tf.constant(feature_dict["is_real_example"], shape=[1], dtype=tf.int32),
    })
    return dataset.batch(batch_size=batch_size, drop_remainder=False)
  return input_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.data_file:
    raise ValueError("`data_file` must be specified.")

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train`, `do_eval` or `do_predict` must be True.")

  bert_config = run_classifier.modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  processor = PoetryProcessor(
      data_file=FLAGS.data_file,
      text_col=FLAGS.text_col,
      use_title=FLAGS.use_title,
      use_author=FLAGS.use_author,
      eval_ratio=FLAGS.eval_ratio,
      split_seed=FLAGS.split_seed)

  label_list = processor.get_labels()
  tokenizer_obj = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file,
      do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(None)
    num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = run_classifier.model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    run_classifier.file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer_obj, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Num labels = %d", len(label_list))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = run_classifier.file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(None)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(run_classifier.PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    run_classifier.file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer_obj, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual)", len(eval_examples), num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_steps = None
    if FLAGS.use_tpu:
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_input_fn = run_classifier.file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=FLAGS.use_tpu)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    output_eval_file = os.path.join(FLAGS.output_dir, "poetry_eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      for key in sorted(result.keys()):
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    if not FLAGS.predict_text:
      raise ValueError("`predict_text` must be provided when do_predict=True.")

    predict_example = processor.build_predict_example(
        title=FLAGS.predict_title,
        author=FLAGS.predict_author,
        text=FLAGS.predict_text)
    predict_feature = convert_single_prediction(
        predict_example, label_list, FLAGS.max_seq_length, tokenizer_obj)
    predict_input_fn = prediction_input_fn_builder(predict_feature, FLAGS.max_seq_length)
    predictions = list(estimator.predict(input_fn=predict_input_fn))
    probabilities = predictions[0]["probabilities"]

    ranked = sorted(
        [{"label": label_list[index], "probability": float(score)} for index, score in enumerate(probabilities)],
        key=lambda item: item["probability"], reverse=True)
    output = OrderedDict()
    output["predict_title"] = FLAGS.predict_title
    output["predict_author"] = FLAGS.predict_author
    output["predict_text"] = FLAGS.predict_text
    output["top_prediction"] = ranked[0]["label"]
    output["probabilities"] = ranked

    output_predict_file = os.path.join(FLAGS.output_dir, "poetry_prediction.json")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      writer.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")

    tf.logging.info("Prediction written to %s", output_predict_file)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
