# -*- coding: utf-8 -*-
"""
python main.py --mode train

Tensorflow implementation of https://arxiv.org/abs/1611.01604.
"""
import time
import tensorflow as tf
import config
import voice_model
import data_utils
import random

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_root', 'log', 'Directory for model root.')
tf.app.flags.DEFINE_string('train_dir', 'log/train', 'Directory for train.')
tf.app.flags.DEFINE_string('eval_dir', 'log/eval', 'Directory for eval.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_run_steps', 1000000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 180, 'How often to checkpoint.')

tf.app.flags.DEFINE_integer('random_seed', 123, 'A seed value for randomness.')


def _train():
    vocab, vocab_id2word = data_utils.load_vocab(config.vocab_path)
    wav_files, labels, words_size, wav_max_len, labels_vector = data_utils.load_all_data(config.wav_path, config.label_suffix, vocab)
    model = voice_model.VoiceModel(config.batch_size, words_size, "train")

    model.build_graph()
    saver = tf.train.Saver()
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
    sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=FLAGS.checkpoint_secs,
                             global_step=model._global_step)
    config_sess = tf.ConfigProto(
        allow_soft_placement=True)
    config_sess.gpu_options.allow_growth = True
    sess = sv.prepare_or_wait_for_session(config=config_sess)
    step = 0
    # 暂时在每次重启的时候从第一个样例开始，第二个样例按照真正的global step进行
    init_global_step = 1

    while (not sv.should_stop()) and step < FLAGS.max_run_steps:
        batch_wav, batch_label = data_utils.get_next_batches(config.batch_size, init_global_step,
                                                             wav_files, labels_vector, wav_max_len)
        tf.logging.info("-------already fetch this batch data---------")
        start = time.time()
        # tf.logging.info("batch_label:{}".format(labels_vector))
        # (_, summaries, loss, train_step) = model.train(sess, batch_wav, batch_label)
        (summaries, train_step) = model.train(sess, batch_wav, batch_label)

        tf.logging.info("+++++++++++++++++++++++++++++++++++")
        tf.logging.info('took: %.4f sec', time.time()-start)
        tf.logging.info('global_step: %d', train_step)
        # tf.logging.info('loss: {}'.format(loss))
        # tf.logging.info('should_stop: {}'.format(sv.should_stop()))
        # tf.logging.info('batch_question: {}'.format(original_question))
        # tf.logging.info('batch_answer(answer1-answer2): {}-{}'.format(original_answer1, original_answer2))
        # tf.logging.info('labels(label1-label2): {}-{}'.format(batch_label1, batch_label2))
        tf.logging.info("+++++++++++++++++++++++++++++++++++")
        init_global_step = int(train_step)

        summary_writer.add_summary(summaries, train_step)
        # summary_writer.add_summary(summaries, loss)
        step += 1
        if step % 100 == 0:
            summary_writer.flush()

    sv.Stop()
        # return running_avg_loss


def id2word(id_array, vocab_id2word):
    words = []
    for word_id in id_array:
        words.append(vocab_id2word.get(str(word_id), "unknown"))
    return words


def _eval():
    import Levenshtein
    epoch = 10
    eval_batch_size = 1
    vocab, vocab_id2word = data_utils.load_vocab(config.vocab_path)
    wav_files, labels, words_size, wav_max_len, labels_vector = data_utils.load_all_data(config.wav_path,
                                                                                         config.label_suffix, vocab)
    model = voice_model.VoiceModel(eval_batch_size, words_size, "decode")
    model.build_graph()
    saver = tf.train.Saver()
    len_source = len(wav_files)
    flag = True
    while flag:
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue

        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.train_dir)
            continue

        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        config_sess = tf.ConfigProto(
            allow_soft_placement=True)
        config_sess.gpu_options.allow_growth = True

        # to_word = lambda word_id: vocab_id2word.get(word_id, "unknown")  # 词典中索引2表示unknown

        with tf.Session(config=config_sess) as sess:
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            for i in range(epoch):
                index = random.randint(1, len_source)
                batch_wav, batch_label = data_utils.get_next_batches(eval_batch_size, index,
                                                                     wav_files, labels_vector, wav_max_len)
                # (decoded, predict) = model.infer(sess, batch_wav)
                (decoded_1, predict_1, decoded_2, predict_2, sequence_len, log_probabilities_1, log_probabilities_2) = model.infer(sess, batch_wav)
                words_vector_predict = [list(id2word(indeces, vocab_id2word)) for indeces in predict_1]
                ground_truth = ''.join(labels[index-1])
                predict_result = ''.join(list([str(x) for x in words_vector_predict]))
                error_distance = Levenshtein.distance(ground_truth.encode('utf-8'), predict_result.encode('utf-8'))

                tf.logging.info("+++++++++++++++++++++++++++++++++++")
                tf.logging.info('decoded_1:{}'.format(decoded_1))
                tf.logging.info('decoded_2:{}'.format(decoded_2))
                tf.logging.info('predict:{}'.format(predict_1))
                tf.logging.info('predict_2:{}'.format(predict_2))
                tf.logging.info('log_probabilities_1:{}'.format(log_probabilities_1))
                tf.logging.info('log_probabilities_2:{}'.format(log_probabilities_2))
                tf.logging.info('predict result:{}'.format(predict_result))
                tf.logging.info('sequence_len:{}'.format(sequence_len))
                tf.logging.info('labels:{}'.format(ground_truth))
                tf.logging.info('labels vector:{}'.format(labels_vector[index-1]))
                tf.logging.info('error_distance:{}'.format(error_distance))
                tf.logging.info('error_rate:{}'.format(error_distance/len(ground_truth)))
                tf.logging.info("+++++++++++++++++++++++++++++++++++")

        flag = False


def main(unused_argv):
    mode = FLAGS.mode
    tf.set_random_seed(FLAGS.random_seed)

    if mode == 'train':
        _train()
    elif mode == 'eval':
        _eval()


def log_train_correct_count(correct_count, total_count, file_name):
    with open("data/"+file_name+".log", 'a') as log_file:
        log_file.write(str(correct_count) + ' ' + str(total_count) + '\n')


if __name__ == '__main__':
    tf.app.run()
