import helper
import numpy as np
import tensorflow as tf

from collections import Counter
from tensorflow.contrib import seq2seq

def build_nn(cell, rnn_size, input_data, vocab_size):
    embed_inputs = get_embed(input_data, vocab_size, rnn_size)
    outputs, final_state = build_rnn(cell, embed_inputs)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, \
        activation_fn = None)

    return (logits, final_state)

def build_rnn(cell, inputs):
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
    final_state = tf.identity(final_state, 'final_state')

    return outputs, final_state

def create_lookup_tables(text):
    word_counts = Counter(text)
    _sorted = sorted(word_counts, key = word_counts.get, reverse = True)

    i_to_v = { i: word for i, word in enumerate(_sorted) }
    v_to_i = { word: i for i, word in i_to_v.items() }

    return (v_to_i, i_to_v)

def get_batches(int_text, batch_size, seq_length):
    num_batches = len(int_text) // (seq_length * batch_size)
    output = []

    for i in range(num_batches):
        _x = []
        _y = []

        for ii in range(batch_size):
            mult = (i * seq_length) + (ii * seq_length)
            _x.append(int_text[mult : mult + seq_length])
            _y.append(int_text[mult + 1 : mult + 1 + seq_length])
        output.append([_x, _y])

    return np.array(output)

def get_embed(input_data, vocab_size, embed_dim):
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    return tf.nn.embedding_lookup(embedding, input_data)

def get_init_cell(batch_size, rnn_size):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * 2)

    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, 'initial_state')

    return (cell, initial_state)

def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')

    return (inputs, targets, lr)

def get_tensors(loaded_graph):
    tensors = ['input:0', 'initial_state:0', 'final_state:0', 'probs:0']
    return tuple([loaded_graph.get_tensor_by_name(tensor) \
        for tensor in tensors])

def pick_word(probabilities, int_to_vocab):
    return int_to_vocab[np.random.choice(len(probabilities), p = probabilities)]

def token_lookup():
    tkn_dict = {}
    tkn_dict['.'] = '<PERIOD>'
    tkn_dict[','] = '<COMMA>'
    tkn_dict['"'] = '<QUOTATION_MARK>'
    tkn_dict[';'] = '<SEMICOLON>'
    tkn_dict['!'] = '<EXCLAMATION_MARK>'
    tkn_dict['?'] = '<QUESTION_MARK>'
    tkn_dict['('] = '<LEFT_PARENTHESIS>'
    tkn_dict[')'] = '<RIGHT_PARENTHESIS>'
    tkn_dict['\n'] = '<NEW_LINE>'
    tkn_dict['--'] = '<DASH>'
    return tkn_dict

### HyperParameters
batch_size = 512
every_n_batches = 4
learning_rate = 0.05
num_epochs = 40
rnn_size = 512
seq_length = 15

### Build Network
helper.preprocess_and_save_data('./data/simpsons/moes_tavern_lines.txt', \
    token_lookup, create_lookup_tables)
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)

    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)

    probs = tf.nn.softmax(logits, name = 'probs')
    cost = seq2seq.sequence_loss(logits, targets, \
        tf.ones([input_data_shape[0], input_data_shape[1]]))
    optimizer = tf.train.AdamOptimizer(lr)

    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for \
        grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gradients)

### Train Network
batches = get_batches(int_text, batch_size, seq_length)
with tf.Session(graph = train_graph) as s:
    s.run(tf.global_variables_initializer())

    for e in range(num_epochs):
        state = s.run(initial_state, { input_text: batches[0][0] })

        for b, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate }
            train_loss, state, _ = s.run([cost, final_state, train_op], feed)

            if (e * len(batches) + b) % every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{} train_loss = {:.3f}'.format(
                    (e + 1), b, len(batches), train_loss))

    saver = tf.train.Saver()
    saver.save(s, './save')
    print('Model Trained and Saved')

helper.save_params((seq_length, './save'))
