import json
import os
import random
import shutil
import time
import numpy as np
import tensorflow as tf
import language_helpers
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot
import pprint
# sys.path.append(os.getcwd())
import argparse

parser = argparse.ArgumentParser(description='Run wgan with improved training on some texts.')
parser.add_argument('data_dir', type=str,
                    help='Directory with text. All files from this directory are used')
parser.add_argument('output_dir', type=str,
                    help='Directory for outputs. It will override everything that is there.')
parser.add_argument('--batch_size', type=int, nargs='?', default=128,
                    help='Batch size.')
parser.add_argument('--iters', type=int, nargs='?', default=2000,
                    help='Number of iterations.')
parser.add_argument('--seq_len', type=int, nargs='?', default=32,
                    help='Length of generated sequences.')
parser.add_argument('--dim', type=int, nargs='?', default=500,
                    help='Medels dimensionality. The same dimensionality is used almost everywhere. Default is usually too much.')
parser.add_argument('--critiq_iters', type=int, nargs='?', default=20,
                    help='Number of critique iterations per one generator.')
parser.add_argument('--gradient_penalty', type=int, nargs='?', default=10,
                    help='Gradient penalty (lambda).')
parser.add_argument('--max_examples', type=int, nargs='?', default=1000000,
                    help='Maximal number of training examples.')
parser.add_argument('--ngram_data_size', type=int, nargs='?', default=1000000,
                    help='Data used for ngrams training.')
parser.add_argument('--ngram_size', type=int, nargs='?', default=4,
                    help='Ngram size used for validation.')

# DATA_DIR = '/media/vlejd/4BCEC8CA76426012/ML/data/1-billion-word-language-modeling-benchmark-r13output/small'

args = parser.parse_args()
arg_dict = vars(args)
pprint.pprint(arg_dict)  # print settings

DATA_DIR = args.data_dir
BATCH_SIZE = args.batch_size
ITERS = args.iters
SEQ_LEN = args.seq_len
DIM = args.dim
CRITIC_ITERS = args.critiq_iters
LAMBDA = args.gradient_penalty
MAX_N_EXAMPLES = args.max_examples
NGRAM_EXAMPLES = args.ngram_data_size
OUTPUT_DIR = args.output_dir
NGRAM_SIZE = args.ngram_size

if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)

json.dump(arg_dict, open(os.path.join(OUTPUT_DIR, 'args.json'), "w"))

tflib.plot.set_dir(OUTPUT_DIR)

lines, charmap, inv_charmap = language_helpers.load_dataset(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)


def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, len(charmap)])
        ),
        tf.shape(logits)
    )


def make_noise(shape):
    return tf.random_normal(shape)


def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 5, output)
    return inputs + (0.3*output)


def Generator(n_samples, prev_outputs=None):
    print("Generator")
    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, SEQ_LEN*DIM, output)
    output = tf.reshape(output, [-1, DIM, SEQ_LEN])
    output = ResBlock('Generator.1', output)
    output = ResBlock('Generator.2', output)
    output = ResBlock('Generator.3', output)
    output = ResBlock('Generator.4', output)
    output = ResBlock('Generator.5', output)
    output = lib.ops.conv1d.Conv1D('Generator.Output', DIM, len(charmap), 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output)
    return output


def Discriminator(inputs):
    output = tf.transpose(inputs, [0, 2, 1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', len(charmap), DIM, 1, output)
    output = ResBlock('Discriminator.1', output)
    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    output = tf.reshape(output, [-1, SEQ_LEN*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', SEQ_LEN*DIM, 1, output)
    return output


real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
fake_inputs = Generator(BATCH_SIZE)
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

disc_real = Discriminator(real_inputs)
disc_fake = Discriminator(fake_inputs)

disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN lipschitz-penalty
alpha = tf.random_uniform(
    shape=[BATCH_SIZE, 1, 1],
    minval=0.,
    maxval=1.
)
differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)


# Dataset iterator
def inf_train_gen():
    while True:
        random.shuffle(lines)
        for i in range(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]],
                dtype='int32'
            )


print('Validation')

# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
true_char_ngram_lms = []
for i in range(NGRAM_SIZE):
    print("Training {0:d}-grams".format(i+1))
    true_char_ngram_lms.append(
        language_helpers.NgramLanguageModel(i+1, lines[:NGRAM_EXAMPLES][10*BATCH_SIZE:]))

validation_char_ngram_lms = []
for i in range(NGRAM_SIZE):
    print("Validating {0:d}-grams".format(i+1))
    validation_char_ngram_lms.append(
        language_helpers.NgramLanguageModel(i+1, lines[:NGRAM_EXAMPLES][:10*BATCH_SIZE]))

for i in range(NGRAM_SIZE):
    print("validation set JSD for n={}: {}".format(
        i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))

# true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines) for i in range(4)]

with tf.Session() as session:
    print("Session start")
    session.run(tf.initialize_all_variables())

    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    gen = inf_train_gen()

    for iteration in range(ITERS):
        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        for i in range(CRITIC_ITERS):
            _data = gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_inputs_discrete: _data}
            )

        tflib.plot.plot('time', time.time() - start_time)
        tflib.plot.plot('train disc cost', _disc_cost)

        if not iteration % 5:
            samples = []
            for i in range(10):
                samples.extend(generate_samples())

            for i in range(NGRAM_SIZE):
                lm = language_helpers.NgramLanguageModel(i+1, samples, tokenize=False)
                tflib.plot.plot('js{}'.format(i+1), lm.js_with(true_char_ngram_lms[i]))

            sample_dir = os.path.join(OUTPUT_DIR, 'samples_{}.txt'.format(iteration))
            with open(sample_dir, 'w') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")

        if iteration % 10 == 9:
            tflib.plot.flush()

        tflib.plot.tick()
