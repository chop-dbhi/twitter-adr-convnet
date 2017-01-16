# Copyright 2016 The Children's Hospital of Philadelphia. All Rights Reserved.
# Created by Dr. Aaron J. Masino
# May 6, 2016

import tensorflow as tf
from math import ceil as ceil

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: regularization constant (typically called lambda but that is a Python keyword)
        add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inference(tweets,
              num_classes,
              vocab_size, tweet_length, word_embedding_length,
              maps_per_filter_width,
              dropout_p, l2_lambda,
              pretrained_word_embeddings = None,
              filter_widths_by_conv_stack = [[1]]):
    '''
    builds the tweetnet model - a 1 layer convolutional neural net
    :param tweets:
    :param num_classes
    :param vocab_size:
    :param tweet_length: length of longest tweets, others are assumed padded to same length
    :param word_embedding_length:
    :param dropout_p: dropout regularization probability
    :param pretrained_word_embeddings: a pretrained embedding matrix
    :param filter_widths_by_conv_stack: list of lists containing filter widths for each successive convolutional layer for multi-path stacked conv nets (number stacked conv2d layers = len(list))
    :return: output layer - tensor of shape [len(tweets) X num_classes] of non-normalized predictions scores
    '''

    # Instantiate all variables using tf.Variable()
    # If using multiple gpus need to use tf.get_variable(), see
    # https://www.tensorflow.org/versions/r0.8/tutorials/deep_cnn/index.html#training-a-model-using-multiple-gpu-cards
    #
    # TODO ideally, one would not need to specify tweet_length or add padding as a CNNet w/ pooling handles
    # nonuniform sized inputs naturally. Not sure how to handle this in tensor flow

    with tf.variable_scope('embedding') as scope:
        # the embedding matrix
        initializer = tf.random_uniform_initializer(-1.0, 1.0)

        if pretrained_word_embeddings is not None:
            embedding_matrix = pretrained_word_embeddings
        else:
            embedding_matrix = _variable_on_cpu("embeddings",
                                                [vocab_size, word_embedding_length],
                                                initializer)
        #exapnd dimension from [batch, height, width] to [batch, height, width, channels] where for a
        #tweet with N words each represented by a word embedding of length W, height = W and width = N
        embedded_tokens = tf.expand_dims(tf.nn.embedding_lookup(embedding_matrix, tweets), -1, name=scope.name)

    # define the convolution layer
    conv_layer_output = []
    for conv_stack_idx,filter_widths in enumerate(filter_widths_by_conv_stack):
        with tf.variable_scope("conv-stack-{0}".format(conv_stack_idx)) as scope:
            prev_layer = embedded_tokens
            filter_height = word_embedding_length
            channels_per_sample = 1
            for layer_idx, filter_width in enumerate(filter_widths):
                with tf.variable_scope("conv-layer-{0}".format(layer_idx)) as scope:
                    kernel = _variable_with_weight_decay('weights',
                                                         shape=[filter_width,
                                                                filter_height,
                                                                channels_per_sample,
                                                                maps_per_filter_width],
                                                         stddev=0.1,
                                                         wd = l2_lambda)
                    conv = tf.nn.conv2d(prev_layer,
                                        kernel,
                                        strides=[1,1,1,1],
                                        padding='VALID',
                                        name='conv')
                    biases = _variable_on_cpu('biases', [maps_per_filter_width], tf.constant_initializer(0.0))
                    relu = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
                    prev_layer = relu
                    filter_height = 1
                    channels_per_sample = maps_per_filter_width
                    _activation_summary(relu)

            # POOL AT LAST conv layer
            # apply max pooling
            stride = prev_layer.get_shape()[1].value
            pool = tf.nn.max_pool(prev_layer,
                                  ksize=[1, stride, 1, 1],
                                  strides=[1,1,1,1],
                                  padding='VALID',
                                  name='pool')
            #append outputs
            conv_layer_output.append(pool)

            # TODO is there a more "tensor flow" way to do this? Maybe with tf.add_to_collection?
    # I am concerned there is suboptimal data flow
    #combine outputs from each filter
    total_filter_count = maps_per_filter_width * len(filter_widths_by_conv_stack)
    conv_layer_output_flat = tf.reshape(tf.concat(3, conv_layer_output), [-1, total_filter_count])

    # add dropout regularization
    with tf.variable_scope('dropout') as scope:
        conv_layer_with_dropout = tf.nn.dropout(conv_layer_output_flat, dropout_p, name=scope.name)
        _activation_summary(conv_layer_with_dropout)

    # output layer (no normalization), for normalized need to add softmax computation, also would need to change
    # loss function
    with tf.variable_scope('output') as scope:
        weights = _variable_with_weight_decay('weights',
                                              [total_filter_count, num_classes],
                                              stddev=1/float(total_filter_count),
                                              wd=l2_lambda)
        biases = _variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(tf.matmul(conv_layer_with_dropout, weights), biases, name='logits')

    return outputs

def predictions(logits):
    return tf.arg_max(logits, 1, name='predictions')

def loss(logits, target_labels):
    '''
    Add l2 loss to trainable variables
    :param output:
    :param target_labels:
    :return:
    '''


    labels = tf.cast(target_labels, tf.float32)

    # FROM TENSORFLOW DOCS:
    # WARNING: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency.
    # Do not call this op with the output of softmax, as it will produce incorrect results.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_tweet')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # total loss is defined as the average cross entropy loss per tweet plus all of the weight decay terms (L2 loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  """Add summaries for losses in tweetnet model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train(total_loss, global_step,
          batch_size,
          num_examples_per_train_epoch, num_epochs_per_decay,
          initial_learn_rate=0.1, learn_rate_decay_factor=0.1, moving_average_decay = 0.9999):
    '''
    builds the tensorflow operation for training - does NOT actually train the model
    applies exponential decay to learning rate

    create an optimizer and apply for trainable variables
    add moving average for all trainable variables (for status monitoring on tensor board)

    :param total_loss:
    :param global_step:
    :return: train_op: the tensorflow operation for training
    '''

    #setup exponential decay learning rate
    num_batches_per_epoch = num_examples_per_train_epoch / batch_size
    decay_step = int(num_batches_per_epoch * num_epochs_per_decay)
    lr = tf.train.exponential_decay(initial_learn_rate,
                                    global_step,
                                    decay_step,
                                    learn_rate_decay_factor,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # moving averages of losses and summaries
    loss_averages_op = _add_loss_summaries(total_loss)

    # note compute & apply gradients can be done in one step with opt.minimize()
    # however we wish to add gradient info to tensorboard so we calculate separately
    # compute gradients
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # add histograms for gradients
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # add histograms for gradients
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # track the moving average of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
