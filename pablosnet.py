import tensorflow as tf

class PablosNet():

    def __init__(self):
        self.model = None

    def build_model(self, images):
        """Model function for CNN."""

        # Training Parameters
        learning_rate = 0.0001
        epochs = 10
        display_step = 10
        n_images, rows, columns, slices = images.shape
        batch_size = 5
        num_classes = 2

        # tf Graph input
        X = tf.placeholder(tf.float32, [None, n_images])
        Y = tf.placeholder(tf.float32, [None, num_classes])
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Store layers weight & bias
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, num_classes]))
        }

        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        # Input Layer
        input_layer = tf.reshape(images, [batch_size, rows, columns, slices])

        # Convolution Layer
        conv1 = self.conv3d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool3d(conv1, k=2)

        # Convolution Layer
        conv2 = self.conv3d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool3d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out


    # Create some wrappers for simplicity
    def conv3d(self, x, weights, biases, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv3d(x, weights, strides=[1, strides, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, biases)
        return tf.nn.relu(x)

    def maxpool3d(self, x, k=2, strides=2):
        return tf.nn.max_pool3d(x, ksize=[1, k, k, k, 1], strides=[1, strides, strides, strides, 1], padding='SAME')

