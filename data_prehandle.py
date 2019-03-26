# -*- coding: utf-8 -*-
import os
from collections import Counter
from tqdm import tqdm

this_util_config = {
    "vocab_saved_path": "./data/vocab.txt",
    "data_path": "/home/yyz/resources/caldi/data_thchs30/data",
    "label_suffix": ".trn"
}


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


def construct_vocab(wav_path, label_file_suffix):
    save_path = this_util_config["vocab_saved_path"]
    wav_files, lables = get_wav_files(wav_path, label_file_suffix)
    all_words = []
    print("construct labels")
    for label in tqdm(lables):
        all_words += [word for word in label]
    counter = Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    print("construct vocabulary")
    vocab = {"PADDING": 0}
    count = 1
    for line in tqdm(words):
        for word in line:
            if word in vocab:
                continue
            else:
                vocab[word] = count
                count = count + 1
    print("saving vocabulary")
    with open(save_path, 'wt', encoding="utf-8") as f:
        for token in tqdm(vocab):
            f.write(token + ' ' + str(vocab[token]) + '\n')

if __name__ == '__main__':
    construct_vocab(this_util_config["data_path"], this_util_config["label_suffix"])
