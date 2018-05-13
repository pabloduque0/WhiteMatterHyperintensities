import tensorflow as tf
import numpy as np
from collections import OrderedDict
import logging

class Unet(object):
    """
    A unet implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, channels=3, n_class=2, cost_kwargs={}, **kwargs):
        tf.reset_default_graph()

        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)

        self.x = tf.placeholder("float", shape=[None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, None, None, n_class])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        logits, self.variables, self.offset = self.create_conv_net(self.x, self.keep_prob, channels, n_class, **kwargs)

        self.cost = self._get_cost(logits, cost_kwargs)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        self.cross_entropy = tf.reduce_mean(self.cross_entropy(tf.reshape(self.y, [-1, n_class]),
                                                          tf.reshape(self.pixel_wise_softmax_2(logits), [-1, n_class])))

        self.predicter = self.pixel_wise_softmax_2(logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def create_conv_net(self, x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2,
                        summaries=True):
        """
        Creates a new convolutional unet for the given parametrization.

        :param x: input tensor, shape [?,nx,ny,channels]
        :param keep_prob: dropout probability tensor
        :param channels: number of channels in the input image
        :param n_class: number of output labels
        :param layers: number of layers in the net
        :param features_root: number of features in the first layer
        :param filter_size: size of the convolution filter
        :param pool_size: size of the max pooling operation
        :param summaries: Flag if summaries should be created
        """

        logging.info(
            "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(
                layers=layers,
                features=features_root,
                filter_size=filter_size,
                pool_size=pool_size))
        # Placeholder for the input image
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image
        batch_size = tf.shape(x_image)[0]

        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()

        in_size = 1000
        size = in_size
        # down layers
        for layer in range(0, layers):
            features = 2 ** layer * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            if layer == 0:
                w1 = self.weight_variable([filter_size, filter_size, channels, features], stddev)
            else:
                w1 = self.weight_variable([filter_size, filter_size, features // 2, features], stddev)

            w2 = self.weight_variable([filter_size, filter_size, features, features], stddev)
            b1 = self.bias_variable([features])
            b2 = self.bias_variable([features])

            conv1 = self.conv2d(in_node, w1, keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = self.conv2d(tmp_h_conv, w2, keep_prob)
            dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size -= 4
            if layer < layers - 1:
                pools[layer] = self.max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2

        in_node = dw_h_convs[layers - 1]

        # up layers
        for layer in range(layers - 2, -1, -1):
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))

            wd = self.weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev)
            bd = self.bias_variable([features // 2])
            h_deconv = tf.nn.relu(self.deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = self.crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            w1 = self.weight_variable([filter_size, filter_size, features, features // 2], stddev)
            w2 = self.weight_variable([filter_size, filter_size, features // 2, features // 2], stddev)
            b1 = self.bias_variable([features // 2])
            b2 = self.bias_variable([features // 2])

            conv1 = self.conv2d(h_deconv_concat, w1, keep_prob)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = self.conv2d(h_conv, w2, keep_prob)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size *= 2
            size -= 4

        # Output Map
        weight = self.weight_variable([1, 1, features_root, n_class], stddev)
        bias = self.bias_variable([n_class])
        conv = self.conv2d(in_node, weight, tf.constant(1.0))
        output_map = tf.nn.relu(conv + bias)
        up_h_convs["out"] = output_map

        if summaries:
            for i, (c1, c2) in enumerate(convs):
                tf.summary.image('summary_conv_%02d_01' % i, self.get_image_summary(c1))
                tf.summary.image('summary_conv_%02d_02' % i, self.get_image_summary(c2))

            for k in pools.keys():
                tf.summary.image('summary_pool_%02d' % k, self.get_image_summary(pools[k]))

            for k in deconv.keys():
                tf.summary.image('summary_deconv_concat_%02d' % k, self.get_image_summary(deconv[k]))

            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

        variables = []
        for w1, w2 in weights:
            variables.append(w1)
            variables.append(w2)

        for b1, b2 in biases:
            variables.append(b1)
            variables.append(b2)

        return output_map, variables, int(in_size - size)


    def weight_variable(self, shape, stddev=0.1):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)

    def weight_variable_devonc(self, shape, stddev=0.1):
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, keep_prob_):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        return tf.nn.dropout(conv_2d, keep_prob_)

    def deconv2d(self, x, W, stride):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID')

    def max_pool(self, x, n):
        return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

    def crop_and_concat(self, x1, x2):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

    def pixel_wise_softmax(self, output_map):
        exponential_map = tf.exp(output_map)
        evidence = tf.add(exponential_map, tf.reverse(exponential_map, [False, False, False, True]))
        return tf.div(exponential_map, evidence, name="pixel_wise_softmax")

    def pixel_wise_softmax_2(self, output_map):
        exponential_map = tf.exp(output_map)
        sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
        tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
        return tf.div(exponential_map, tensor_sum_exp)

    def cross_entropy(self, y_, output_map):
        return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")

    def get_image_summary(self, img, idx=0):
        """
        Make an image summary for 4d tensor image with index idx
        """

        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
        V *= 255

        img_w = tf.shape(img)[1]
        img_h = tf.shape(img)[2]
        V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
        return V


    def _get_cost(self, logits, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """

        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])

        eps = 1e-5
        prediction = self.pixel_wise_softmax_2(logits)
        intersection = tf.reduce_sum(prediction * self.y)
        union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
        loss = -(2 * intersection / (union))



        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += (regularizer * regularizers)

        return loss

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})

        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)
