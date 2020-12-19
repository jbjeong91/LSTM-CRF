def LSTM_CRF(embeddings, lstm_size, seq_len, num_tags, labels):
    lstm_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, activation=tf.nn.tanh)
    lstm_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, activation=tf.nn.tanh)
    output_states, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, inputs=embeddings, dtype=tf.float32)
    output = tf.concat(output_states,2)
    # CRF
    #logits = tf.layers.dense(output, num_tags)
    logits = tf.layers.dense(output, num_tags)
    seq_len = np.array([seq_len] * DEFINES.batch_size)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, seq_len)
    '''
    tf.contrib.crf.crf_log_likelihood(
    inputs,
    tag_indices,
    sequence_lengths,
    transition_params=None)
    
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials to use as input to the CRF layer.
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we compute the log-likelihood.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix, if available.
    '''
    loss = 1e-10
    if DEFINES.cmd == 'train':
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, labels, seq_len, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

    return pred_ids, logits, loss
