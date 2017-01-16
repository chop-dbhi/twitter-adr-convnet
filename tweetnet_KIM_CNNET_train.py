import tensorflow as tf
from utils import tweetnet_wrangle as wrangle
from utils import config_helper as confhelp
import tweetnet_KIM_CNNET
import tweetnet_KIM_CNNET_eval
import numpy as np
import time
from datetime import datetime
from itertools import product, izip
from sklearn_x.metrics import PerformanceMetrics, KFoldPerformanceMetrics
import os

ROOT_DATA_PATH = os.path.join('.', 'data')
ROOT_INPUT_DATA_PATH = os.path.join(ROOT_DATA_PATH, 'input')
TRAINING_DATA_POSITIVE_PATH = os.path.join(ROOT_INPUT_DATA_PATH, 'ds1_revised_train_pos.txt')
TRAINING_DATA_NEGATIVE_PATH = os.path.join(ROOT_INPUT_DATA_PATH, 'ds1_revised_train_neg.txt')
ROOT_OUTPUT_DATA_PATH = os.path.join(ROOT_DATA_PATH,'output')
TRAINING_EVENTS_OUTPUT_PATH = os.path.join(ROOT_OUTPUT_DATA_PATH, 'training')
CHECKPOINTS_PATH = os.path.join(TRAINING_EVENTS_OUTPUT_PATH, 'checkpoints')
LOG_FILE = os.path.join(ROOT_OUTPUT_DATA_PATH,'logs','tweetnet.log')
WORD_EMBEDDING_DIR = os.path.join(ROOT_DATA_PATH, 'resources', 'embeddings')
_WORD_EMBEDDING_FILES = ['word2vec_twitter_model.bin']

# Global constants for training
_WORD_EMBEDDING_LENGTH = 16
_USE_PRETRAINED_WORD_EMBEDDING = False
NUM_CLASSES = 2

_NUM_EPOCHS = 10
_DROPOUT_KEEP_PROB = 0.5
L2_CONSTRAINT = 0.004
_FILTER_WIDTHS= [3,4,5]
# Y. KIM USED 100 CHANNELS PER FILTER WIDTH, THOUGH WITH A MUCH LARGER DATA SET
_NUM_MAPS_PER_FILTER_WIDTH = 16
BATCH_SIZE = 64
INITIAL_LEARN_RATE = 0.3
LEARN_RATE_DECAY_FACTOR = 0.0
MOVING_AVERAGE_DECAY = 1.0
_NFOLDS = 5

min_fraction_of_examples_in_queue = 0.3

SAVE_CHECKPOINTS = True
VERBOSE_MODE = False

#PARAMETER MODE
# 0 READ CONFIGURATION FILE AS SINGLE PARAMETER SET
# 1 READ CONFIGURATION FILE AS RANGE OF PARAMETERS
SINGLE_PARAMETER_MODE = 0
MULTI_PARAMETER_MODE = 1


def log(line):
    with open(LOG_FILE, 'a') as f:
        f.write('{0}\n'.format(line))


def dictionaryXproduct(d):
    return (dict(izip(d, x)) for x in product(*d.itervalues()))


def gridSearch(parameter_ranges, training_tweets_as_words, training_labels, tweet_length,
               cross_validate=True, train_and_store_best=False):

    # generate X product of parameter ranges
    parameter_XP = dictionaryXproduct(parameter_ranges)
    cnt = 1
    for v in parameter_ranges.values():
	cnt *= len(v)
    log('Starting grid search for {0} parameter combinations'.format(cnt))
    param_keys = parameter_ranges.keys()
    param_string = param_keys[0]
    for k in param_keys[1:]:
	param_string = '{0},{1}'.format(param_string,k)
    f = open(os.path.join(ROOT_OUTPUT_DATA_PATH, 'cross_validation_results.txt'), 'a')
    f.write('{0},accuracy_mean, accuracy_std, f1_mean, f1_std, npv_mean, npv_std, ppv_mean, ppv_std, '.format(param_string) +
            'precision_mean, precision_std, recall_mean, recall_std, sensitivity_mean, sensitivity_std, ' +
            'specificity_mean, specificity_std\n')
    f.close()
    max_f1_score = 0
    best_params = None
    
    icnt = 0
    for parameters in parameter_XP:
        if cross_validate:
            kfpm = crossValidate(parameters, training_tweets_as_words, training_labels, tweet_length)
            param_str = str(parameters[param_keys[0]]).replace(",","_")
            for k in param_keys[1:]:
                v = parameters[k]
                param_str = '{0},{1}'.format(param_str,str(v).replace(",","_"))
            line = '{0},{1},{2}'.format(param_str,kfpm.accuracy_mean, kfpm.accuracy_std)
            line = '{0},{1},{2}'.format(line, kfpm.f1_mean, kfpm.f1_std)
            line = '{0},{1},{2}'.format(line, kfpm.npv_mean, kfpm.npv_std)
            line = '{0},{1},{2}'.format(line,kfpm.ppv_mean, kfpm.ppv_std)
            line = '{0},{1},{2}'.format(line,kfpm.precision_mean, kfpm.precision_std)
            line = '{0},{1},{2}'.format(line,kfpm.recall_mean, kfpm.recall_std)
            line = '{0},{1},{2}'.format(line,kfpm.sensitivity_mean, kfpm.sensitivity_std)
            line = '{0},{1},{2}\n'.format(line,kfpm.specificity_mean, kfpm.specificity_std)
            f = open(os.path.join(ROOT_OUTPUT_DATA_PATH, 'cross_validation_results.txt'), 'a')
            f.write(line)
            f.close()
            if kfpm.f1_mean>max_f1_score:
                max_f1_score = kfpm.f1_mean
                best_params = parameters
            icnt += 1
            log('Completed {0} of {1} parameter combinations in grid search'.format(icnt, cnt)) 
        else:
            # WHY ARE YOU NOT CROSS VALIDATING IN A GRID SEARCH?
            pass

    if train_and_store_best:
        train(training_tweets_as_words, training_labels, tweet_length,
              best_params, save_file_base_name='tweetnet_best_{0}'.format(file_name_from_params(best_params)))


def file_name_from_params(params):
    ps=''
    for k,v in params.items():
        ps = '{0}_{1}_{2}'.format(ps, k, str(v).replace(',', '_').replace(' ', '_').replace('[','').replace(']','').replace('\'',''))
    ps = hash(ps)
    today = time.strftime("%d_%m_%Y_%H_%M_%S")
    fn = 'tweetnet_{0}_{1}'.format(today,ps)
    return fn

def crossValidate(parameters, training_tweets_as_words, training_labels, tweet_length, nfolds = _NFOLDS):
    cond = training_labels[:,0] == 1
    r = 1/float(nfolds)
    ratios = [r for _ in range(nfolds)]
    (pids, nids) = wrangle.partion(cond, ratios)
    bins = [np.concatenate(z) for z in zip(pids, nids)]
    pm_bins = []
    use_pretrained_word_embeddings = parameters.get('USE_PRETRAINED_WORD_EMBEDDING',_USE_PRETRAINED_WORD_EMBEDDING)
    if use_pretrained_word_embeddings:
        vocab, vocab_inv = wrangle.build_vocab(training_tweets_as_words)
    for idx, b1 in enumerate(bins):
        cv_tweets = training_tweets_as_words[b1]

        # convert the 2D labels to the 1D target class for use in eval call
        cv_labels = np.argmax(training_labels[b1],axis=1)
        tr_tweets = None
        tr_labels = None
        for jdx, b2 in enumerate(bins):
            if not jdx==idx:
                if tr_tweets is not None:
                    tr_tweets = np.concatenate((tr_tweets, training_tweets_as_words[b2]))
                    tr_labels = np.concatenate((tr_labels, training_labels[b2]))
                else:
                    tr_tweets = training_tweets_as_words[b2]
                    tr_labels = training_labels[b2]

        z = np.array(zip(tr_tweets, tr_labels))
        np.random.shuffle(z)
        tr_tweets = z[:,0]
        tr_labels = np.array([_.tolist() for _ in z[:,1]])
        base_file_name = '{0}_{1}'.format(file_name_from_params(parameters), idx)

        saved_file_path = train(tr_tweets, tr_labels, tweet_length,
                                parameters, base_file_name)


        if not use_pretrained_word_embeddings:
            vocab, vocab_inv = wrangle.build_vocab(tr_tweets)
        _, pred_labels = tweetnet_KIM_CNNET_eval.eval(cv_tweets, cv_labels, tweet_length,
                                                      vocab, vocab_inv,
                                                      saved_file_path, parameters, False, batch_size=len(cv_labels))
        #print('pred_labels = \n{0}'.format(pred_labels))
        pm_bins.append(PerformanceMetrics(cv_labels, pred_labels))
	#remove checkpoint file
	os.remove(saved_file_path)
	os.remove('{0}.meta'.format(saved_file_path))

    return KFoldPerformanceMetrics(pm_bins)

def train(training_tweets_as_words, training_labels, tweet_length,
          parameter_dict = {},
          save_file_base_name = 'tweetnet',
          save_checkpoints = True,
          checkpoints_save_path = CHECKPOINTS_PATH, loss_listener = None,
          cv_tweets_as_words=None, cv_labels=None, cv_tweet_length=None):
    ''' train the tweetnet model'''

    perform_cv_each_batch = (cv_tweets_as_words is not None) and (cv_labels is not None) and (cv_tweet_length is not None)

    # set parameters
    num_epochs = parameter_dict.get('NUM_EPOCHS', _NUM_EPOCHS)
    dropout_keep_prob = parameter_dict.get('DROPOUT_KEEP_PROB', _DROPOUT_KEEP_PROB)
    filter_widths = parameter_dict.get('FILTER_WIDTHS', _FILTER_WIDTHS)
    num_maps_per_filter_width = parameter_dict.get('NUM_MAPS_PER_FILTER_WIDTH', _NUM_MAPS_PER_FILTER_WIDTH)
    use_stratification = parameter_dict.get('USE_STRATIFICATION', False)
    negative_ratio = parameter_dict.get('NEGATIVE_RATIO', 0.5)
    word_embedding_length = parameter_dict.get('WORD_EMBEDDING_LENGTH', _WORD_EMBEDDING_LENGTH)
    use_pretrained_word_embeddings = parameter_dict.get('USE_PRETRAINED_WORD_EMBEDDING',_USE_PRETRAINED_WORD_EMBEDDING)
    word_embedding_files = parameter_dict.get('WORD_EMBEDDING_FILES', _WORD_EMBEDDING_FILES)
    use_multiple_embeddings = len(word_embedding_files) > 1

    # preload input data
    #training_tweets_as_words, training_labels, tweet_length = load_train_inputs()

    if use_pretrained_word_embeddings and perform_cv_each_batch:
        vocab, vocab_inv = wrangle.build_vocab(training_tweets_as_words + cv_tweets_as_words)
    else:
        vocab, vocab_inv = wrangle.build_vocab(training_tweets_as_words)
    vocab_size = len(vocab_inv)
    training_tweets_indices = wrangle.tweetsToIndices(training_tweets_as_words, tweet_length, vocab)

    if perform_cv_each_batch:
        cv_tweet_indices = wrangle.tweetsToIndices(cv_tweets_as_words, cv_tweet_length, vocab)

    num_training_examples = len(training_labels)
    print("{0} training examples".format(num_training_examples))
    num_batches_per_epoch = int(np.floor(num_training_examples/BATCH_SIZE))
    NUM_EXAMPLES_PER_EPOCH_TRAIN = num_batches_per_epoch * BATCH_SIZE
    NUM_EPOCHS_PER_DECAY = 100

    #min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_TRAIN * min_fraction_of_examples_in_queue)
    TOTAL_BATCH_COUNT = num_epochs * num_batches_per_epoch

    if use_stratification:
        batches = batch_iter_stratify(training_tweets_indices, training_labels, BATCH_SIZE, num_epochs, negative_ratio=negative_ratio)
    else:
        batches = batch_iter(zip(training_tweets_indices, training_labels), BATCH_SIZE, num_epochs)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # create placeholders for inputs
        tweets = tf.placeholder(tf.int32, [BATCH_SIZE, tweet_length])
        labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])

        if use_multiple_embeddings and use_pretrained_word_embeddings:
            embedding_matrices = []
            for f in word_embedding_files:
                p = os.path.join(WORD_EMBEDDING_DIR, f)
                em, es = wrangle.extractWord2VecEmbeddings(p, vocab_inv)
                pre_trained_embedding_constant = tf.constant(em, tf.float32, shape=es)
                embedding_matrices.append(pre_trained_embedding_constant)

            # build graph to compute logits
            if VERBOSE_MODE:
                print("logits based on mulitple word embedding files {0}".format(word_embedding_files))
            if perform_cv_each_batch:
                with tf.variable_scope("cv_loss") as scope:
                    logits = tweetnet_KIM_CNNET.inference_avg_multi_pretrained(tweets, NUM_CLASSES, tweet_length,
                                                                            filter_widths, num_maps_per_filter_width,
                                                                            dropout_keep_prob, L2_CONSTRAINT,
                                                                            embedding_matrices)
                    tweets_cv = tf.constant(cv_tweet_indices, tf.int32, shape=[len(cv_tweets_as_words), cv_tweet_length])
                    labels_cv = tf.constant(cv_labels, tf.float32, shape=[len(cv_tweets_as_words), NUM_CLASSES])
                    scope.reuse_variables()
                    logits_cv = tweetnet_KIM_CNNET.inference_avg_multi_pretrained(tweets_cv, NUM_CLASSES, cv_tweet_length,
                                                                            filter_widths, num_maps_per_filter_width,
                                                                            1.0, L2_CONSTRAINT,
                                                                            embedding_matrices)
            else:
                logits = tweetnet_KIM_CNNET.inference_avg_multi_pretrained(tweets, NUM_CLASSES, tweet_length,
                                                                        filter_widths, num_maps_per_filter_width,
                                                                        dropout_keep_prob, L2_CONSTRAINT,
                                                                        embedding_matrices)
        else:
            if use_pretrained_word_embeddings:
                    if VERBOSE_MODE:
                        print("logits based on single embedding file {0}".format(word_embedding_files[0]))
                    p = os.path.join(WORD_EMBEDDING_DIR, word_embedding_files[0])
                    pre_trained_embeddings, embedding_shape = \
                        wrangle.extractWord2VecEmbeddings(p, vocab_inv)
                    word_embedding_length = embedding_shape[1]
                    pre_trained_embedding_constant = tf.constant(pre_trained_embeddings, tf.float32, shape=embedding_shape)
            else:
                if VERBOSE_MODE:
                    print("logits based on learned embeddings")
                pre_trained_embedding_constant = None
                # set vocab size += 1 to maintain an untrained (random) value for words in eval sets that are NOT seen in training
                vocab_size += 1

            # build graph to compute logits
            if perform_cv_each_batch:
                with tf.variable_scope("cv_loss") as scope:
                    logits = tweetnet_KIM_CNNET.inference(tweets, NUM_CLASSES, vocab_size, tweet_length, word_embedding_length,
                                                  filter_widths, num_maps_per_filter_width,
                                                  dropout_keep_prob, L2_CONSTRAINT,
                                                  pre_trained_embedding_constant)
                    tweets_cv = tf.constant(cv_tweet_indices, tf.int32, shape=[len(cv_tweets_as_words), cv_tweet_length])
                    labels_cv = tf.constant(cv_labels, tf.float32, shape=[len(cv_tweets_as_words), NUM_CLASSES])
                    scope.reuse_variables()
                    logits_cv = tweetnet_KIM_CNNET.inference(tweets_cv, NUM_CLASSES, vocab_size, cv_tweet_length, word_embedding_length,
                                                         filter_widths, num_maps_per_filter_width, 1.0, L2_CONSTRAINT,
                                                         pre_trained_embedding_constant)
            else:
                logits = tweetnet_KIM_CNNET.inference(tweets, NUM_CLASSES, vocab_size, tweet_length, word_embedding_length,
                                                  filter_widths, num_maps_per_filter_width,
                                                  dropout_keep_prob, L2_CONSTRAINT,
                                                  pre_trained_embedding_constant)

        # calculate loss
        loss = tweetnet_KIM_CNNET.loss(logits, labels)
        if perform_cv_each_batch:
            loss_cv = tweetnet_KIM_CNNET.loss(logits_cv, labels_cv)
            predictions_op = tweetnet_KIM_CNNET.predictions(logits_cv)

        # create train op
        train_op = tweetnet_KIM_CNNET.train(loss, global_step, BATCH_SIZE,
                                            NUM_EXAMPLES_PER_EPOCH_TRAIN,
                                            NUM_EPOCHS_PER_DECAY,
                                            INITIAL_LEARN_RATE,
                                            LEARN_RATE_DECAY_FACTOR,
                                            MOVING_AVERAGE_DECAY)

        # save stuff
        saver = tf.train.Saver(tf.all_variables())

        # build summary op
        if VERBOSE_MODE:
            summary_op = tf.merge_all_summaries()

        # initialize
        init = tf.initialize_all_variables()

        # start the graph ops
        sess = tf.Session()
        sess.run(init)

        # start queue runners
        tf.train.start_queue_runners(sess=sess)

        if VERBOSE_MODE:
            summary_writer = tf.train.SummaryWriter(TRAINING_EVENTS_OUTPUT_PATH, sess.graph)

        # the latest saved version of the model, can be passed directly to tf.train.Saver.restore()
        latest_saved_file = ''
        for step in xrange(TOTAL_BATCH_COUNT):
            start_time = time.time()
            batch = batches.next()
            tweet_batch, label_batch = zip(*batch)
            feed_dict = {tweets: tweet_batch,
                         labels: label_batch,
                         }


            if perform_cv_each_batch:
                _, loss_value, loss_value_cv, predictions = sess.run([train_op, loss, loss_cv, predictions_op], feed_dict=feed_dict)
            else:
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time
            if loss_listener is not None:
                if perform_cv_each_batch:
                    loss_listener(step, loss_value, loss_value_cv, predictions, np.argmax(cv_labels,axis=1), step+1==TOTAL_BATCH_COUNT)
                else:
                    loss_listener(step, loss_value)

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if VERBOSE_MODE and step % 10 == 0:
                num_examples_per_step = BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                log (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

            if VERBOSE_MODE and step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # save model checkpoint
            if save_checkpoints and ((VERBOSE_MODE and step % 100 == 0) or (step+1) == TOTAL_BATCH_COUNT):
                latest_saved_file = saver.save(sess, os.path.join(CHECKPOINTS_PATH,save_file_base_name), global_step=step)

        return latest_saved_file


def batch_iter_stratify(tweets, labels, batch_size, num_epochs, negative_ratio=0.5):
    num_neg = int(negative_ratio * batch_size)
    num_pos = batch_size - num_neg

    pos_tweet_indices = np.where(labels[:, 0] == 0)[0]

    neg_tweet_indices = np.where(labels[:, 0] == 1)[0]

    # print('pos_indices: {0}'.format(pos_tweet_indices))
    # print('neg_incides: {0}'.format(neg_tweet_indices))

    data_size = len(labels)
    num_examples = len(labels)
    num_batches_per_epoch = int(np.floor(num_examples/batch_size))
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            rand_neg_tweet_indices = neg_tweet_indices[np.random.permutation(len(neg_tweet_indices))][0:num_neg]
            rand_pos_tweet_indices = pos_tweet_indices[np.random.permutation(len(pos_tweet_indices))][0:num_pos]
            tweet_indices = np.concatenate((rand_pos_tweet_indices, rand_neg_tweet_indices))
            np.random.shuffle(tweet_indices)
            batch_tweets = tweets[tweet_indices]
            batch_labels = labels[tweet_indices]
            yield zip(batch_tweets, batch_labels)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
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


def load_train_inputs():
    pos_tweets = wrangle.load_tweets(TRAINING_DATA_POSITIVE_PATH)
    #need 2D labels for training (prob of neg , pos class) because tensorflow calculates on both
    pos_labels = np.array([[0,1] for _ in pos_tweets], dtype=np.float32)
    neg_tweets = wrangle.load_tweets(TRAINING_DATA_NEGATIVE_PATH)
    neg_labels = np.array([[1,0] for _ in neg_tweets], dtype=np.float32)
    tweets = pos_tweets + neg_tweets
    labels = np.concatenate((pos_labels,neg_labels))
    padded_tweets, tweet_length = wrangle.pad_tweets(tweets)
    return padded_tweets, labels, tweet_length

def load_config(config_file, parameter_mode = SINGLE_PARAMETER_MODE):
    conf = confhelp.loadConfig(config_file)
    if parameter_mode==SINGLE_PARAMETER_MODE:
        section = 'SingleParameterMode'
        params = {}
        if conf.has_option(section, 'NUM_EPOCHS'):
            params['NUM_EPOCHS'] = conf.getint(section, 'NUM_EPOCHS')
        if conf.has_option(section, 'DROPOUT_KEEP_PROB'):
            params['DROPOUT_KEEP_PROB'] = conf.getfloat(section, 'DROPOUT_KEEP_PROB')
        if conf.has_option(section, 'FILTER_WIDTHS'):
            params['FILTER_WIDTHS'] = confhelp.getListInt(conf, section, 'FILTER_WIDTHS')
        if conf.has_option(section, 'NUM_MAPS_PER_FILTER_WIDTH'):
            params['NUM_MAPS_PER_FILTER_WIDTH'] = conf.getint(section, 'NUM_MAPS_PER_FILTER_WIDTH')
        if conf.has_option(section, 'USE_STRATIFICATION'):
            params['USE_STRATIFICATION'] = conf.getboolean(section, 'USE_STRATIFICATION')
        if conf.has_option(section, 'NEGATIVE_RATIO'):
            params['NEGATIVE_RATIO'] = conf.getfloat(section, 'NEGATIVE_RATIO')
        if conf.has_option(section, 'WORD_EMBEDDING_LENGTH'):
            params['WORD_EMBEDDING_LENGTH'] = conf.getint(section, 'WORD_EMBEDDING_LENGTH')
        if conf.has_option(section, 'USE_PRETRAINED_WORD_EMBEDDING'):
            params['USE_PRETRAINED_WORD_EMBEDDING'] = conf.getboolean(section, 'USE_PRETRAINED_WORD_EMBEDDING')
        if conf.has_option(section, 'WORD_EMBEDDING_FILES'):
            params['WORD_EMBEDDING_FILES'] = [x.strip() for x in conf.get(section, 'WORD_EMBEDDING_FILES').split(',')]
        return params
    else:
        section = 'MultiParameterMode'
        parameter_ranges = {}
        if conf.has_option(section, 'NUM_EPOCHS'):
            parameter_ranges['NUM_EPOCHS'] = confhelp.getListInt(conf, section, 'NUM_EPOCHS')

        if conf.has_option(section, 'DROPOUT_KEEP_PROB'):
            parameter_ranges['DROPOUT_KEEP_PROB'] = confhelp.getListFloat(conf, section, 'DROPOUT_KEEP_PROB')

        if conf.has_section('MultiParameterMode_FilterWidths'):
            opt_dict = confhelp.ConfigSectionMap(conf, 'MultiParameterMode_FilterWidths')
            fws = []
            for v in opt_dict.values():
                fws.append([int(x) for x in v.split(',')])
            if fws:
                parameter_ranges['FILTER_WIDTHS'] = fws

        if conf.has_option(section, 'NUM_MAPS_PER_FILTER_WIDTH'):
            parameter_ranges['NUM_MAPS_PER_FILTER_WIDTH'] = confhelp.getListInt(conf, section, 'NUM_MAPS_PER_FILTER_WIDTH')

        if conf.has_option(section, 'USE_STRATIFICATION'):
            parameter_ranges['USE_STRATIFICATION'] = confhelp.getListBool(conf, section, 'USE_STRATIFICATION')

        if conf.has_option(section, 'NEGATIVE_RATIO'):
            parameter_ranges['NEGATIVE_RATIO'] = confhelp.getListFloat(conf, section, 'NEGATIVE_RATIO')

        if conf.has_option(section, 'WORD_EMBEDDING_LENGTH'):
            parameter_ranges['WORD_EMBEDDING_LENGTH'] = confhelp.getListInt(conf, section, 'WORD_EMBEDDING_LENGTH')

        if conf.has_option(section, 'USE_PRETRAINED_WORD_EMBEDDING'):
            parameter_ranges['USE_PRETRAINED_WORD_EMBEDDING'] = confhelp.getListBool(conf, section, 'USE_PRETRAINED_WORD_EMBEDDING')

        if conf.has_section('MultiParameterMode_WordEmbeddingFiles'):
            opt_dict = confhelp.ConfigSectionMap(conf, 'MultiParameterMode_WordEmbeddingFiles')
            wes = []
            for v in opt_dict.values():
                wes.append([s.strip() for s in v.split(',')])
            if wes:
                parameter_ranges['WORD_EMBEDDING_FILES'] = wes
        return parameter_ranges


def main(argv=None):
    if argv is not None and len(argv)==3:
        config_file = argv[1]
        parameter_mode = argv[2]
        conf = confhelp.loadConfig(config_file)
    else:
        print("Application requires 2 arguments:\nconfiguration file path\nparameter mode")
        exit(0)

    training_tweets_as_words, training_labels, tweet_length = load_train_inputs()

    param_config = load_config(config_file, parameter_mode)

    if parameter_mode == SINGLE_PARAMETER_MODE:
        print(param_config)
        print("len training tweets as words: {0}".format(len(training_tweets_as_words)))
        print("tweet length = {0}".format(tweet_length))
        model_file = train(training_tweets_as_words, training_labels, tweet_length, param_config)
        eval_labels = np.argmax(training_labels[0:100],axis=1)

        vocab, vocab_inv = wrangle.build_vocab(training_tweets_as_words)
        tweetnet_KIM_CNNET_eval.eval(training_tweets_as_words[0:100], eval_labels, tweet_length,
                                     vocab, vocab_inv,
                                     saved_model_file=model_file, parameter_dict=param_config)
    else:
        log(param_config)

        gridSearch(param_config, np.array(training_tweets_as_words), np.array(training_labels),
                   tweet_length, True, True)

if __name__ == '__main__':
    tf.app.run()
