import numpy as np
import tensorflow as tf

from helper import load_params, load_preprocess
from nn_tv_script_creator import get_tensors, pick_word


_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()

### Create Script
len_script = 400
first_word = 'homer_simpson'

loaded_graph = tf.Graph()
with tf.Session(graph = loaded_graph) as s:
    print('here')
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(s, load_dir)

    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    gen_sentences = [first_word + ':']
    print('here2')
    prev_state = s.run(initial_state, { input_text : np.array([[1]]) })

    print('here3')
    for n in range(len_script):
        dyn_input = [[vocab_to_int[word] for word \
            in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        _p, prev_state = s.run(
            [probs, final_state],
            { input_text: dyn_input, initial_state : prev_state })

        pred_word = pick_word(_p[dyn_seq_length - 1], int_to_vocab)
        gen_sentences.append(pred_word)

    tv_script = ' '.join(gen_sentences)

    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')

    print(tv_script)
