# -*- coding: utf-8 -*-
import numpy as np
import os
import librosa  # https://github.com/librosa/librosa
import tensorflow as tf
import config


def get_wav_files(wav_path, label_file_suffix):
    wav_files = []
    labels = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                # tf.logging.info("handle_wav:{}".format(filename))
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:  # 剔除掉一些小文件
                    continue
                label_file_path = os.sep.join([dirpath, filename+label_file_suffix])
                with open(label_file_path, 'r') as label_file:
                    for line in label_file:
                        words = line.strip('\n').split()
                        labels.append(words)
                        # tf.logging.info("line info:{}".format(line))
                        break
                wav_files.append(filename_path)
    return wav_files, labels

label_max_len = 0


def load_vocab(vocab_path):
    vocab = {}
    vocab_id2word = {}
    with open(vocab_path, 'r') as words_file:
        for line in words_file:
            line = line.strip('\n')
            (word, index) = line.split()
            vocab[word] = index
            vocab_id2word[index] = word
    return vocab, vocab_id2word


# special tokens
UNKNOWN_INDEX = 2
SILENCE_INDEX = 0


def words2id(words, vocab):
    indices = []
    for word in words:
        if word in vocab:
            indices.append(int(vocab.get(str(word), UNKNOWN_INDEX)))
        else:
            if len(word) > 1:
                for char in word:
                    indices.append(int(vocab.get(str(char), UNKNOWN_INDEX)))
        indices.append(SILENCE_INDEX)
    return indices


def load_all_data(wav_path, label_file_suffix, vocab):
    wav_files, labels = get_wav_files(wav_path, label_file_suffix)
    tf.logging.info("样本数:{}".format(len(wav_files)))  # 8911

    # tf.logging.info(wav_files[0], labels[0])
    # wav/train/A11/A11_0.WAV -> 绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 秀媚 诗意 盎然
    words_size = len(vocab)
    tf.logging.info('词汇表大小:{}'.format(words_size))
    # to_num = lambda word: vocab.get(word, 2)  # 词典中索引2表示unknown
    # labels_vector = [list(map(to_num, label)) for label in labels]
    labels_vector = [list(words2id(label, vocab)) for label in labels]
    # tf.logging.info(wavs_file[0], labels_vector[0])
    # wav/train/A11/A11_0.WAV -> [479, 0, 7, 0, 138, 268, 0, 222, 0, 714, 0, 23, 261, 0, 28, 1191, 0, 1, 0, 442, 199, 0, 72, 38, 0, 1, 0, 463, 0, 1184, 0, 269, 7, 0, 479, 0, 70, 0, 816, 254, 0, 675, 1707, 0, 1255, 136, 0, 2020, 91]
    # tf.logging.info(words[479]) #绿
    global label_max_len
    label_max_len = np.max([len(label) for label in labels_vector])
    tf.logging.info('最长句子的字数:{}'.format(label_max_len))

    # wav_max_len = 0  # 673
    wav_max_len = 673  # 673
    # for wav_file in wav_files:
    #     # tf.logging.info("trying to open wav:{}".format(wav_file))
    #     wav, sr = librosa.load(wav_file, mono=True)
    #     mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0])
    #     if len(mfcc) > wav_max_len:
    #         wav_max_len = len(mfcc)
    tf.logging.info("最长的语音:{}".format(wav_max_len))

    return wav_files, labels, words_size, wav_max_len, labels_vector


def get_next_batches(batch_size, step, wav_files, labels_vector, wav_max_len):
    # tf.logging.info("----------------------load next batch data---------------------------")
    pointer = batch_size * (step - 1)
    batches_wavs = []  # batch_size * n * 20, 这里的n是每个声道的长度,是声音的长度,最终：batch_size * wav_max_len * 20，20bit采样
    batches_labels = []
    len_wav_files = len(wav_files)
    for i in range(batch_size):
        pointer = pointer % len_wav_files
        wav, sr = librosa.load(wav_files[pointer], sr=16000, mono=True)
        # tf.logging.info("wav file:{}, sample rate:{}".format(wav_files[pointer], sr))
        mfcc = np.transpose(librosa.feature.mfcc(wav, sr, n_mfcc=config.n_mfcc), [1, 0])
        wav_data = mfcc.tolist()
        label_data = labels_vector[pointer]
        # 补零对齐
        while len(wav_data) < wav_max_len:
            wav_data.append([0] * config.n_mfcc)
        if len(wav_data) > wav_max_len:
            wav_data = wav_data[0:wav_max_len]
        while len(label_data) < label_max_len:
            label_data.append(0)
        if len(label_data) > label_max_len:
            label_data = label_data[0:label_max_len]
        batches_wavs.append(wav_data)
        batches_labels.append(label_data)
        pointer += 1
    # tf.logging.info("len of wav files:{}, len of label list:{}".format(len(batches_wavs), len(batches_labels)))
    # tf.logging.info("len of wav:{}, len of label:{}".format(len(batches_wavs[0][0]), len(batches_labels[0])))
    # tf.logging.info("label:{}\n".format(list(len(x) for x in batches_labels)))
    # tf.logging.info("-------------------------------------------------")
    # return batches_wavs, batches_labels
    return batches_wavs, np.array(batches_labels)
