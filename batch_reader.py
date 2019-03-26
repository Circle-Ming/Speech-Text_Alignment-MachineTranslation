# -*- coding: utf-8 -*-

"""
Batch generator with bucketing support.
"""
import queue
import time

from collections import namedtuple
from random import shuffle
from threading import Thread

import numpy as np
import tensorflow as tf
import data_utils_pre_h

from data_utils_pre_h import tf_Examples

ModelInput = namedtuple('ModelInput',
                        ['input_wav_data', 'input_label_data', 'input_label_data_raw'])

BUCKET_CACHE_BATCH = 3
QUEUE_NUM_BATCH = 3


class Generator:
  """Data class for batch generator."""
  def __init__(self, file_path, batch_size,
               wav_data_key, label_data_key, label_data_raw_key, wav_length, words_length):
    self._file_path = file_path
    self._batch_size = batch_size
    self._wav_length = wav_length
    self._words_length = words_length
    self._wav_data_key = wav_data_key
    self._label_data_key = label_data_key
    self._label_data_raw_key = label_data_raw_key
    self._input_queue = queue.Queue(QUEUE_NUM_BATCH * self._batch_size)
    self._bucket_input_queue = queue.Queue(QUEUE_NUM_BATCH)
    self._input_threads = []

    for _ in range(2):
      self._input_threads.append(Thread(target=self._enqueue))
      self._input_threads[-1].daemon = True
      self._input_threads[-1].start()

    self._bucketing_threads = []
    for _ in range(1):
      self._bucketing_threads.append(Thread(target=self._fill_bucket))
      self._bucketing_threads[-1].daemon = True
      self._bucketing_threads[-1].start()

    self._watch_thread = Thread(target=self._monitor)
    self._watch_thread.daemon = True
    self._watch_thread.start()

  def next(self):
    """Returns next batch of inputs for model.
    Returns:
      batch_context: A batch of encoder inputs [c_timesteps, batch_size].
      batch_question: A batch of encoder inputs [q_timesteps, batch_size].
      batch_answer: A batch of one-hot encoded answers [2, batch_size].
      origin_context: original context words.
      origin_question: original question words.
      origin_answer: original answer words.
    """
    # batch_wav = []
    # batch_label = []
    # batch_label_raw = []
    #
    #
    # buckets = self._bucket_input_queue.get()
    # for i in range(self._batch_size):
    #   (wav, label, label_raw) = buckets[i]
    #
    #   batch_wav.append(wav)
    #   batch_label.append(label)
    #   batch_label_raw.append(label_raw)

    batch_wav = np.zeros(
        (self._batch_size, self._wav_length, 20), dtype=np.float32)
    batch_label = np.zeros(
        (self._batch_size, self._words_length), dtype=np.int32)
    batch_label_raw = []

    buckets = self._bucket_input_queue.get()
    for i in range(self._batch_size):
      (wav, label, label_raw) = buckets[i]
      batch_wav[i] = wav
      batch_label[i, :] = label[:]
      batch_label_raw.append(label_raw)

    return (batch_wav, batch_label, batch_label_raw)

  def _enqueue(self):
    """Fill input queue with ModelInput."""
    input_gen = self._textGenerator(tf_Examples(self._file_path))

    while True:
      (wav, label, label_raw) = next(input_gen)
      element = ModelInput(wav, label, label_raw)
      self._input_queue.put(element)


  def _fill_bucket(self):
    """Fill bucketed batches into the bucket_input_queue."""
    while True:
      inputs = []
      for _ in range(self._batch_size * BUCKET_CACHE_BATCH):
        inputs.append(self._input_queue.get())

      batches = []
      for i in range(0, len(inputs), self._batch_size):
        batches.append(inputs[i:i+self._batch_size])
      # shuffle(batches)

      for b in batches:
        self._bucket_input_queue.put(b)

  def _monitor(self):
    """Watch the daemon input threads and restart if dead."""
    while True:
      time.sleep(60)
      input_threads = []
      for t in self._input_threads:
        if t.is_alive():
          input_threads.append(t)
        else:
          tf.logging.error('Found input thread dead.')
          new_t = Thread(target=self._enqueue)
          input_threads.append(new_t)
          input_threads[-1].daemon = True
          input_threads[-1].start()

      self._input_threads = input_threads

      bucketing_threads = []
      for t in self._bucketing_threads:
        if t.is_alive():
          bucketing_threads.append(t)
        else:
          tf.logging.error('Found bucketing thread dead.')
          new_t = Thread(target=self._fill_bucket)
          bucketing_threads.append(new_t)
          bucketing_threads[-1].daemon = True
          bucketing_threads[-1].start()

      self._bucketing_threads = bucketing_threads

  def _getExFeatureText(self, ex, key):
    """Extract text for a feature from td.Example.
    Args:
      ex: tf.Example.
      key: key of the feature to be extracted.
    Returns:
      feature: a feature text extracted.
    """
    return ex.features.feature[key].bytes_list.value

  def _get_int64_feature(self, ex, key):
    return ex.features.feature[key].int64_list.value

  def _get_float_feature(self, ex, key):
    return ex.features.feature[key].float_list.value



  def _textGenerator(self, example_gen):
    """Yields original (context, question, answer) tuple."""
    while True:
      e = next(example_gen)
      try:
        wav = self._get_float_feature(e, self._wav_data_key)
        wav = data_utils_pre_h.pat2two_dim(wav, 20)
        label = self._get_int64_feature(e, self._label_data_key)
        label_raw = self._getExFeatureText(e, self._label_data_raw_key)[0].decode('utf-8')
      except ValueError:
        tf.logging.error('Failed to get data from example')
        continue

      yield (wav, label, label_raw)
