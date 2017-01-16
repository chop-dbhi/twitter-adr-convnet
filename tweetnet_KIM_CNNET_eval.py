import tensorflow as tf
import numpy as np
import tweetnet_KIM_CNNET
import tweetnet_KIM_CNNET_train
from utils import tweetnet_wrangle as wrangle
import os
import itertools
from sklearn_x.metrics import PerformanceMetrics
from sklearn_x import printers
from sklearn.metrics import confusion_matrix

ROOT_DATA_PATH = os.path.join('.','data')
ROOT_INPUT_DATA_PATH = os.path.join(ROOT_DATA_PATH, 'input')
EVAL_DATA_POSITIVE_PATH = os.path.join(ROOT_INPUT_DATA_PATH, 'ds1_revised_test_pos.txt')
EVAL_DATA_NEGATIVE_PATH = os.path.join(ROOT_INPUT_DATA_PATH, 'ds1_revised_test_neg.txt')
ROOT_OUTPUT_DATA_PATH = os.path.join(ROOT_DATA_PATH,'output')
EVAL_OUTPUT_PATH = os.path.join(ROOT_OUTPUT_DATA_PATH, 'eval')
TRAINING_EVENTS_OUTPUT_PATH = os.path.join(ROOT_OUTPUT_DATA_PATH, 'training')
CHECKPOINTS_PATH = os.path.join(TRAINING_EVENTS_OUTPUT_PATH, 'checkpoints')
WORD_EMBEDDING_DIR = os.path.join(ROOT_DATA_PATH, 'resources', 'embeddings')
_WORD_EMBEDDING_FILES = ['word2vec_twitter_model.bin']


_WORD_EMBEDDING_LENGTH = 16
_USE_PRETRAINED_WORD_EMBEDDING = False
NUM_CLASSES = 2

# NO DROPOUT DURING EVAL
_DROPOUT_KEEP_PROB = 1.0

L2_CONSTRAINT = 0.004
_FILTER_WIDTHS= [3,4,5]
_NUM_MAPS_PER_FILTER_WIDTH = 16
_BATCH_SIZE = 1
INITIAL_LEARN_RATE = 0.3
MOVING_AVERAGE_DECAY = 1.0

def eval_once(saver,predictions_op, correct_for_batch_op,  feed_dict, ckpt_file = None):
    with tf.Session() as sess:
        if ckpt_file is None:
            ckpt_file = tf.train.latest_checkpoint(CHECKPOINTS_PATH)

        if ckpt_file:
            saver.restore(sess, ckpt_file)
        else:
            print('No checkpoint file found')
            return

        predictions, correct_for_batch = sess.run([predictions_op, correct_for_batch_op], feed_dict=feed_dict)
        #print('predictions = {0}'.format(predictions))
        true_count = np.sum(correct_for_batch)
        return true_count, predictions

def build_logits(vocab, vocab_inv,
                 tf_tweet_place_holder, tweet_length,
                 parameter_dict):

    vocab_size = len(vocab_inv)

    # model parameters
    filter_widths = parameter_dict.get('FILTER_WIDTHS', _FILTER_WIDTHS)
    num_maps_per_filter_width = parameter_dict.get('NUM_MAPS_PER_FILTER_WIDTH', _NUM_MAPS_PER_FILTER_WIDTH)
    word_embedding_length = parameter_dict.get('WORD_EMBEDDING_LENGTH', _WORD_EMBEDDING_LENGTH)
    use_pretrained_word_embeddings = parameter_dict.get('USE_PRETRAINED_WORD_EMBEDDING',_USE_PRETRAINED_WORD_EMBEDDING)
    word_embedding_files = parameter_dict.get('WORD_EMBEDDING_FILES', _WORD_EMBEDDING_FILES)
    use_multiple_embeddings = len(word_embedding_files) > 1

    if use_multiple_embeddings and use_pretrained_word_embeddings:
        embedding_matrices = []
        for f in word_embedding_files:
            p = os.path.join(WORD_EMBEDDING_DIR, f)
            em, es = wrangle.extractWord2VecEmbeddings(p, vocab_inv)
            pre_trained_embedding_constant = tf.constant(em, tf.float32, shape=es)
            embedding_matrices.append(pre_trained_embedding_constant)

        # build graph to compute logits
        logits = tweetnet_KIM_CNNET.inference_avg_multi_pretrained(tf_tweet_place_holder, NUM_CLASSES, tweet_length,
                                                                       filter_widths, num_maps_per_filter_width,
                                                                       _DROPOUT_KEEP_PROB, L2_CONSTRAINT,
                                                                       embedding_matrices)
    else:
        if use_pretrained_word_embeddings:
            p = os.path.join(WORD_EMBEDDING_DIR, word_embedding_files[0])
            pre_trained_embeddings, embedding_shape = \
                wrangle.extractWord2VecEmbeddings(p, vocab_inv)
            word_embedding_length = embedding_shape[1]
            pre_trained_embedding_constant = tf.constant(pre_trained_embeddings, tf.float32, shape=embedding_shape)
        else:
            pre_trained_embedding_constant = None
            # set vocab size += 1 to maintain an untrained (random) value for words in eval sets that are NOT seen in training
            vocab_size += 1

        # build graph to compute logits
        logits = tweetnet_KIM_CNNET.inference(tf_tweet_place_holder, NUM_CLASSES, vocab_size, tweet_length, word_embedding_length,
                                              filter_widths, num_maps_per_filter_width,
                                              _DROPOUT_KEEP_PROB, L2_CONSTRAINT,
                                              pre_trained_embedding_constant)
    return logits



def eval(eval_tweets_as_words, eval_labels, tweet_length,
         vocab = None, vocab_inv = None,
         saved_model_file=None,
         parameter_dict={},
         print_results=True,
         batch_size = _BATCH_SIZE):
    '''

    :param eval_tweets_as_words:
    :param eval_labels: NOTE: these are 1-D labels used to compare predictions, NOT the 2-D labels used in training
    :param tweet_length:
    :param saved_model_file:
    :param parameter_dict:
    :param print_results:
    :param batch_size:
    :return:
    '''

    if vocab is None and vocab_inv is None:
        vocab, vocab_inv = wrangle.build_vocab(eval_tweets_as_words)

    vocab_size = len(vocab_inv)
    eval_tweets_indices = wrangle.tweetsToIndices(eval_tweets_as_words, tweet_length, vocab)

    batches = batch_iter(zip(eval_tweets_indices, eval_labels), batch_size, 1, shuffle=False)

    # # model parameters
    # filter_widths = parameter_dict.get('FILTER_WIDTHS', _FILTER_WIDTHS)
    # num_maps_per_filter_width = parameter_dict.get('NUM_MAPS_PER_FILTER_WIDTH', _NUM_MAPS_PER_FILTER_WIDTH)
    # word_embedding_length = parameter_dict.get('WORD_EMBEDDING_LENGTH', _WORD_EMBEDDING_LENGTH)
    # use_pretrained_word_embeddings = parameter_dict.get('USE_PRETRAINED_WORD_EMBEDDING',_USE_PRETRAINED_WORD_EMBEDDING)
    # word_embedding_files = parameter_dict.get('WORD_EMBEDDING_FILES', _WORD_EMBEDDING_FILES)
    # use_multiple_embeddings = len(word_embedding_files) > 1

    # pass in vocab, vocab inv
    # pass vocab size to inference
    # eval_tweets_indices needs to be constructed from this vocab (so skip words in a tweet that are not in the vocab)
    with tf.Graph().as_default() as g:
        #create place holders
        tweets = tf.placeholder(tf.int32, [batch_size, tweet_length])
        logits = build_logits(vocab, vocab_inv, tweets, tweet_length, parameter_dict)

        # calculate predictions
        labels = tf.placeholder(tf.float32, [batch_size])
        labels_ints = tf.cast(labels, tf.int64)
        #top_k_op = tf.nn.in_top_k(logits, labels_ints, 1)
        predictions_op = tweetnet_KIM_CNNET.predictions(logits)
        correct_predictions_op = tf.equal(predictions_op, labels_ints)

        #restore moving average version of learned variables
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # build summary operation
        # summary_op = tf.merge_all_summaries()

        # summary_writer = tf.train.SummaryWriter(EVAL_OUTPUT_PATH, g)

        total_samples = 0
        correct_count = 0
        sample_positives = 0
        target_labels = []
        output_labels = []
        for idx, batch in enumerate(batches):
            #print("processing batch {0}".format(idx))
            tweet_batch, label_batch = zip(*batch)
            sample_positives += np.sum(label_batch)
            feed_dict = {tweets: tweet_batch,
                         labels: label_batch,
                         }
            #print('batch labels = {0}'.format(label_batch))
            target_labels.append(label_batch)
            total_samples += len(label_batch)
            #correct_count += eval_once(saver, summary_writer, top_k_op, summary_op, feed_dict)
            batch_correct_count, batch_predictions \
                = eval_once(saver, predictions_op, correct_predictions_op, feed_dict,
                            ckpt_file = saved_model_file)
            correct_count += batch_correct_count
            output_labels.append(batch_predictions)


        output_labels = list(itertools.chain(*output_labels))
        target_labels = list(itertools.chain(*target_labels))

        if print_results:
            pm = PerformanceMetrics(target_labels, output_labels)
            output_file = os.path.join(EVAL_OUTPUT_PATH,'eval_stats.txt')
            printers.printsfPerformanceMetrics(pm, output_file)
            cm = confusion_matrix(target_labels, output_labels)
            printers.printTwoClassConfusion(cm, output_file)

        return (target_labels, output_labels)



def load_eval_inputs():
    pos_tweets = wrangle.load_tweets(EVAL_DATA_POSITIVE_PATH)
    #need 1D labels for eval
    pos_labels = np.ones(len(pos_tweets), dtype=np.float32)
    neg_tweets = wrangle.load_tweets(EVAL_DATA_NEGATIVE_PATH)
    neg_labels = np.zeros(len(neg_tweets), dtype=np.float32)
    tweets = pos_tweets + neg_tweets
    labels = np.concatenate((pos_labels,neg_labels))
    padded_tweets, tweet_length = wrangle.pad_tweets(tweets)
    return padded_tweets, labels, tweet_length

def batch_iter(data, batch_size, num_epochs, shuffle = True):
    data = np.array(data)
    data_size = len(data)
    num_examples = len(data)
    num_batches_per_epoch = int(np.floor(num_examples/batch_size))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def main(argv=None):
    eval_tweets_as_words, eval_labels, tweet_length = load_eval_inputs()
    eval(eval_tweets_as_words, eval_labels, tweet_length)

if __name__ == '__main__':
    tf.app.run()
