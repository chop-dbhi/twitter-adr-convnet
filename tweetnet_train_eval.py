__author__ = 'Aaron J. Masino'

from utils import config_helper as confhelp
import tweetnet_KIM_CNNET_train
import tweetnet_KIM_CNNET_eval
import tweetnet_multilayer_train
import tweetnet_multilayer_eval
import tweetnet_KIM_CNNET
import tweetnet_multilayer
from utils import tweetnet_wrangle as wrangle
from sklearn_x.metrics import PerformanceMetrics, KFoldPerformanceMetrics
import tensorflow as tf
from functools import reduce
import os
import numpy as np
import pandas as pd

USE_KIM_MODEL = 0
USE_MULT_LAYER_MODEL = 1

ROOT_DATA_PATH = os.path.join('.','data')
ROOT_DATA_OUTPUT_PATH = os.path.join(ROOT_DATA_PATH, 'output')
ROOT_DATA_EVAL_OUTPUT_PATH = os.path.join(ROOT_DATA_OUTPUT_PATH, 'eval')
NPARRAY_STORAGE_PATH = os.path.join(ROOT_DATA_EVAL_OUTPUT_PATH, 'nparrays')
LOSS_STORAGE_PATH = os.path.join(ROOT_DATA_EVAL_OUTPUT_PATH, 'losses')
BOOST_STORAGE_PATH = os.path.join(ROOT_DATA_EVAL_OUTPUT_PATH, 'boosting')
LOG_FILE = os.path.join(ROOT_DATA_OUTPUT_PATH,'logs','tweetnet.log')
BOOST_DIR = os.path.join(ROOT_DATA_PATH, 'resources', 'boost_tweets')

def log(line):
    with open(LOG_FILE, 'a') as f:
        f.write('{0}\n'.format(line))

def store_filters(saved_model_file, parameters,
                  vocab, vocab_inv, tweet_length,
                  batch_size, evaler, model,
                  output_dir, base_output_file_name):
    with tf.Graph().as_default():
        tweets = tf.placeholder(tf.int32, [batch_size, tweet_length])
        MOVING_AVERAGE_DECAY = 1.0
        logits = evaler.build_logits(vocab, vocab_inv,
                                tweets, tweet_length,
                                parameters)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, saved_model_file)
            if model==USE_KIM_MODEL:
                filter_widths = parameters['FILTER_WIDTHS']
                num_channels = len(parameters.get('WORD_EMBEDDING_FILES',[1]))
                variables = tf.trainable_variables()
                for fw in filter_widths:
                    for c in range(num_channels):
                        name = 'conv-mp-{0}'.format(fw)
                        if num_channels>1:
                            name = '{0}/{1}'.format(name, 'we-channel-{0}'.format(c))
                        name = '{0}/weights:0'.format(name)
                        for v in variables:
                            if v.name == name:
                                x=sess.run(v)
                                n = '{0}_{1}'.format(base_output_file_name,name.replace('/','_').replace(':','_'))
                                fn = os.path.join(output_dir,n)
                                np.save(fn, x)
            elif USE_MULT_LAYER_MODEL:
                pass

def store_performances(performances, out_file):
    kp = KFoldPerformanceMetrics(performances)
    with open(out_file, 'a') as of:
        header='run,accuracy,f1,precision,recall,ppv,npv,specificity'
        of.write(header)
        for idx,p in enumerate(performances):
            values = [p.accuracy, p.f1, p.precision,
                      p.recall, p.ppv, p.npv,
                      p.specificity]
            line = reduce(lambda x,y: '{0},{1}'.format(x,y), values, '\n{0}'.format(str(idx)))
            of.write(line)
        values = [kp.accuracy_mean, kp.f1_mean, kp.precision_mean,
                  kp.recall_mean, kp.ppv_mean, kp.npv_mean, kp.specificity_mean]
        line = reduce(lambda x,y: '{0},{1}'.format(x,y), values, '\nmean')
        of.write(line)
        values = [kp.accuracy_std,kp.f1_std, kp.precision_std,
                  kp.recall_std, kp.ppv_std, kp.npv_std, kp.specificity_std]
        line = reduce(lambda x,y: '{0},{1}'.format(x,y), values, '\nstdev')
        of.write(line)


def loss_listener_creator(storage_file_path, performances=None):

    def loss_listener(batch_id, loss, cv_loss=None, predictions=None, target_labels = None, isLast=False):
        with open(storage_file_path,'a') as f:
            if predictions is not None and target_labels is not None and isLast:
                performance = PerformanceMetrics(target_labels, predictions)
                performances.append(performance)
            if cv_loss is None:
                f.write('\n{0},{1}'.format(batch_id,loss))
            else:
                f.write('\n{0},{1},{2}'.format(batch_id,loss,cv_loss))

    return loss_listener


def unpad_tweets(padded_tweets):
    unpadded_tweets = []
    for tweet in padded_tweets:
        nt = []
        for w in tweet:
            if not w=='<PAD/>':
                nt.append(w)
        unpadded_tweets.append(nt)
    return unpadded_tweets


def generate_boost_samples(parameters, model, trainer, evaler,
                           training_tweets_as_words, training_labels, train_tweet_length,
                           num_boost_runs, boost_tweets):

    #need to strip <PAD/> from training tweets so we can "grow" the set
    base_train_tweets = unpad_tweets(training_tweets_as_words)

    input_tweets_as_words = training_tweets_as_words
    input_labels = training_labels
    input_tweet_length = train_tweet_length

    # boost_tweets is expected as a pandas dataframe with a column labeled 'text' containing the tweet text
    split_boost_samples = np.array_split(boost_tweets, num_boost_runs)
    base_file_name = trainer.file_name_from_params(parameters)

    postive_boost_samples = []
    for idx in xrange(num_boost_runs):
        # train the model on current input set
        saved_file_path = trainer.train(input_tweets_as_words, input_labels, input_tweet_length,
                                        parameters, base_file_name)

        # use model to label randomly selected unlabeled inputs
        non_padded_boost_samples = split_boost_samples[idx]['text'].tolist()
        non_padded_boost_samples = wrangle.clean_strip_split(non_padded_boost_samples)
        padded_boost_samples, boost_tweet_length = wrangle.pad_tweets(non_padded_boost_samples)
        batch_size = len(padded_boost_samples)

        use_pretrained_word_embeddings = parameters.get('USE_PRETRAINED_WORD_EMBEDDING',trainer._USE_PRETRAINED_WORD_EMBEDDING)

        if use_pretrained_word_embeddings:
            vocab, vocab_inv = wrangle.build_vocab(input_tweets_as_words + padded_boost_samples)
        else:
            vocab, vocab_inv = wrangle.build_vocab(input_tweets_as_words)

        # build and run graph to label boost set
        with tf.Graph().as_default():
            tweets = tf.placeholder(tf.int32, [batch_size, boost_tweet_length])
            MOVING_AVERAGE_DECAY = 1.0
            logits = evaler.build_logits(vocab, vocab_inv,
                                         tweets, boost_tweet_length,
                                         parameters)

            #init_op = tf.initialize_all_variables()
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            if model==USE_KIM_MODEL:
                predictions_op = tweetnet_KIM_CNNET.predictions(logits)
            elif USE_MULT_LAYER_MODEL:
                predictions_op = tweetnet_multilayer.predictions(logits)

            with tf.Session() as sess:
                #sess.run(init_op)
                saver.restore(sess, saved_file_path)
                padded_boost_samples_indices = wrangle.tweetsToIndices(padded_boost_samples, boost_tweet_length, vocab)
                feed_dict = {tweets: padded_boost_samples_indices,            }
                predictions = sess.run([predictions_op], feed_dict=feed_dict)[0]
                # add tweets labeled as positive to the input set
                for jdx in xrange(len(predictions)):
                    if predictions[jdx]==1:
                        # add boot strapped example to training set
                        postive_boost_samples.append(non_padded_boost_samples[jdx])
                        base_train_tweets.append(non_padded_boost_samples[jdx])
                        input_labels = np.append(input_labels, [[0,1]],0)
                input_tweets_as_words, input_tweet_length = wrangle.pad_tweets(base_train_tweets)

    return postive_boost_samples


def performance_sampling(parameters, trainer, evaler, num_runs,
                         training_tweets_as_words, training_labels, train_tweet_length,
                         eval_tweets_as_words, eval_labels, eval_tweet_length,
                         apply_boosting=False, model = None, num_boost_runs=None, boost_tweets=None):
    # train on complete training set, test on eval set
    # conduct N train and eval runs to account for different random initializations of weights
    # log the performance metrics for each of the N runs
    performances = []
    for idx in range(num_runs):
        base_file_name = trainer.file_name_from_params(parameters)

        sf = os.path.join(LOSS_STORAGE_PATH, '{0}_losses_run_{1}'.format(base_file_name, idx))
        with open(sf, 'w') as f:
            f.write('batch_id,loss,cv_loss')
        loss_callback = loss_listener_creator(sf, performances)

        cv_labels = []
        for l in eval_labels:
            if l==0:
                cv_labels.append([1,0])
            else:
                cv_labels.append([0,1])
        cv_labels = np.array(cv_labels, dtype=np.float32)

        if apply_boosting:
            positive_boost_samples = generate_boost_samples(parameters, model, trainer, evaler,
                                                           training_tweets_as_words, training_labels, train_tweet_length,
                                                           num_boost_runs, boost_tweets)
            pbsf = os.path.join(BOOST_STORAGE_PATH, '{0}_boosted_samples_run_{1}'.format(base_file_name, idx))
            with open(pbsf, 'a') as f:
                for tweet_as_list in positive_boost_samples:
                    tweet = reduce(lambda x,y: '{0} {1}'.format(x,y), tweet_as_list[1:], tweet_as_list[0])
                    f.write('{0}\n'.format(tweet))
            unpadded_training_tweets = unpad_tweets(training_tweets_as_words)
            input_tweets_as_words, input_tweet_length = wrangle.pad_tweets(unpadded_training_tweets + positive_boost_samples)
            input_labels = np.append(training_labels, [[0,1] for _ in xrange(len(positive_boost_samples))], 0)
        else:
            input_tweets_as_words = training_tweets_as_words
            input_tweet_length = train_tweet_length
            input_labels = training_labels


        #train the model
        saved_file_path = trainer.train(input_tweets_as_words, input_labels, input_tweet_length,
                                            parameters, base_file_name, loss_listener=loss_callback,
                                            cv_tweets_as_words=eval_tweets_as_words,
                                            cv_labels=cv_labels,
                                            cv_tweet_length=eval_tweet_length)

    return performances

def load_eval_raw_tweets(evaler):
    pos_tweets = wrangle.load_raw_tweet_text(evaler.EVAL_DATA_POSITIVE_PATH)
    neg_tweets = wrangle.load_raw_tweet_text(evaler.EVAL_DATA_NEGATIVE_PATH)
    return pos_tweets + neg_tweets

def cleanup(path):
    files = os.listdir(path)
    for f in files:
        file_to_remove = os.path.join(path,f)
        if os.path.isfile(file_to_remove):
            os.remove(file_to_remove)

def main(argv=None):
    if argv is not None and len(argv)==2:
        config_file = argv[1]
        conf = confhelp.loadConfig(config_file)
        log('starting training and eval for: {0}'.format(config_file))
    else:
        print("Application requires input argument:\nconfiguration file path")
        exit(0)

    # get eval config
    section = 'EvalConfig'
    if conf.has_option(section, 'NUM_RUNS'):
        num_runs = conf.getint(section, 'NUM_RUNS')
    else:
        num_runs = 10

    if conf.has_option(section, 'MODEL'):
        model = conf.getint(section, 'MODEL')
    else:
        model = -1

    if conf.has_option(section, 'STORE_PREDICTIONS'):
        _store_predictions = conf.getboolean(section, 'STORE_PREDICTIONS')
    else:
        _store_predictions = False

    if conf.has_option(section, 'STORE_FILTERS'):
       _store_filters = conf.getboolean(section, 'STORE_FILTERS')
    else:
        _store_filters = False

    if conf.has_option(section, 'APPLY_BOOSTING') and \
            conf.has_option(section, 'NUM_BOOST_ITERATIONS') and \
            conf.has_option(section, 'BOOST_TWEET_FILE'):
        _apply_boosting = conf.getboolean(section, 'APPLY_BOOSTING')
        _num_boost_iterations = conf.getint(section, 'NUM_BOOST_ITERATIONS')
        _boost_file = conf.get(section, 'BOOST_TWEET_FILE')
    else:
        _apply_boosting = False

    if not (model==USE_KIM_MODEL or model==USE_MULT_LAYER_MODEL):
        print('Invalid model choice or model not specified in configuration file: {0}'.format(model))
        exit(0)

    if model==USE_KIM_MODEL:
        param_config = tweetnet_KIM_CNNET_train.load_config(config_file)
        trainer = tweetnet_KIM_CNNET_train
        evaler = tweetnet_KIM_CNNET_eval
    elif model==USE_MULT_LAYER_MODEL:
        param_config = tweetnet_multilayer_train.load_config(config_file)
        trainer = tweetnet_multilayer_train
        evaler = tweetnet_multilayer_eval

    training_tweets_as_words, training_labels, train_tweet_length = trainer.load_train_inputs()

    eval_tweets_as_words, eval_labels, eval_tweet_length = evaler.load_eval_inputs()

    if _apply_boosting:
        boost_tweets = pd.read_csv(os.path.join(BOOST_DIR, _boost_file))[1:5000]
        performances = performance_sampling(param_config, trainer, evaler, num_runs,
                                            training_tweets_as_words, training_labels, train_tweet_length,
                                            eval_tweets_as_words, eval_labels, eval_tweet_length,
                                            apply_boosting=_apply_boosting, model = model,
                                            num_boost_runs=_num_boost_iterations, boost_tweets=boost_tweets)
    else:
        performances = performance_sampling(param_config, trainer, evaler, num_runs,
                                            training_tweets_as_words, training_labels, train_tweet_length,
                                            eval_tweets_as_words, eval_labels, eval_tweet_length)

    out_file = '{0}_eval_results.csv'.format(trainer.file_name_from_params(param_config))
    out_file = os.path.join(ROOT_DATA_EVAL_OUTPUT_PATH, out_file)
    store_performances(performances, out_file)

    use_pretrained_word_embeddings = param_config.get('USE_PRETRAINED_WORD_EMBEDDING',trainer._USE_PRETRAINED_WORD_EMBEDDING)

    if _store_filters or _store_predictions:
        # store filters or predictions using ONLY ONE TRAINED MODEL (IE DO NOT LOOP OVER NUM_RUNS)
        base_file_name = trainer.file_name_from_params(param_config)

        # select training tweets based on boost option
        if _apply_boosting:
            positive_boost_samples = generate_boost_samples(param_config, model, trainer, evaler,
                                                           training_tweets_as_words, training_labels, train_tweet_length,
                                                           _num_boost_iterations, boost_tweets)
            unpadded_training_tweets = unpad_tweets(training_tweets_as_words)
            input_tweets_as_words, input_tweet_length = wrangle.pad_tweets(unpadded_training_tweets + positive_boost_samples)
            input_labels = np.append(training_labels, [[0,1] for _ in xrange(len(positive_boost_samples))], 0)
        else:
            input_tweets_as_words = training_tweets_as_words
            input_tweet_length = train_tweet_length
            input_labels = training_labels

        saved_file_path = trainer.train(input_tweets_as_words, input_labels, input_tweet_length,
                                        param_config, base_file_name)

        if use_pretrained_word_embeddings:
            vocab, vocab_inv = wrangle.build_vocab(input_tweets_as_words + eval_tweets_as_words)
        else:
            vocab, vocab_inv = wrangle.build_vocab(input_tweets_as_words)

    if _store_filters:
        # store filters for single run
        store_filters(saved_file_path, param_config, vocab, vocab_inv,
                      eval_tweet_length, len(eval_labels), evaler, model,
                      NPARRAY_STORAGE_PATH, trainer.file_name_from_params(param_config))

    if _store_predictions:
        # store predictions for single run
        _, pred_labels = evaler.eval(eval_tweets_as_words, eval_labels, eval_tweet_length,
                                     vocab, vocab_inv,
                                     saved_file_path, param_config, True, len(eval_labels))
        raw_tweet_text = load_eval_raw_tweets(evaler)
        out_file = '{0}_eval_predicted_labels.csv'.format(trainer.file_name_from_params(param_config))
        out_file = os.path.join(ROOT_DATA_EVAL_OUTPUT_PATH, out_file)
        with open(out_file, 'w') as of:
            lines = 'idx,tweet,actual,pred'
            for idx in xrange(len(raw_tweet_text)):
                lines='{0}\n{1},"{2}",{3},{4}'.format(lines,
                                                      idx,
                                                      raw_tweet_text[idx],
                                                      int(eval_labels[idx]),
                                                      int(pred_labels[idx]))
            of.write(lines)

    cleanup(trainer.CHECKPOINTS_PATH)

    log('training and eval complete for {0}'.format(config_file))

if __name__ == '__main__':
    tf.app.run()



