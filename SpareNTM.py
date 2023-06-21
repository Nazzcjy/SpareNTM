import numpy as np
import tensorflow as tf
import os
from gensim import corpora
import utils as utils
import argparse

# np.random.seed(0)
# tf.set_random_seed(0)

flags = tf.flags
flags.DEFINE_integer('n_hidden', 500, 'Size of each hidden layer.')
flags.DEFINE_boolean('test', True, 'Process test data.')
flags.DEFINE_string('non_linearity', 'relu', 'Non-linearity of the MLP.')
flags.DEFINE_string("topN_file", 'output/SpareNTM.topWords', 'output topWords')
flags.DEFINE_string("theta_file", 'output/SpareNTM.theta', 'output topWords')
FLAGS = flags.FLAGS


class SpareNTM(object):
    def __init__(self, vocab_size, n_hidden, n_topic, learning_rate,
                 batch_size, non_linearity, adam_beta1, adam_beta2, dir_prior, bern_prior):
        tf.reset_default_graph()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.bern = bern_prior

        self.x = tf.placeholder(tf.float32, [None, vocab_size], name='input')
        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings
        self.warm_up = tf.placeholder(tf.float32, (), name='warm_up')  # warm up
        self.temp = tf.placeholder(tf.float32, (), name='temp')
        self.B = tf.placeholder(tf.int32, (), name='B')
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.min_alpha = tf.placeholder(tf.float32, (), name='min_alpha')

        # encoder
        with tf.variable_scope("transform"):
            self.enc_vec = utils.mlp(self.x, [self.n_hidden],tf.nn.relu)
            self.enc_vec = tf.nn.dropout(self.enc_vec, self.keep_prob)
        with tf.variable_scope("encoder_for_bern"):
            un_q1_k = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='q1_k'))
            un_1_q1_k = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='1_q1_k'))
            gumble_logits = tf.stack([un_q1_k, un_1_q1_k], axis=-1)
            u_g = tf.random_uniform(tf.shape(gumble_logits),minval=0.01,maxval=0.99)
            y = gumble_logits - tf.log(-tf.log(u_g+1e-10)+1e-10)
            y = tf.nn.softmax(y / self.temp)
            # y_hard = tf.one_hot(tf.math.argmax(y, axis=-1), depth=y.shape[-1], dtype=y.dtype)
            # y = tf.stop_gradient(y_hard-y)+y
            y = tf.reshape(y, [batch_size, self.n_topic, 2])
            cons = tf.tile(tf.constant([[[1.], [0.]]]), [batch_size, 1, 1])
            z = tf.matmul(y, cons)
            self.lam = tf.squeeze(z, -1)
            q_y = tf.nn.softmax(tf.stack([un_q1_k, un_1_q1_k], axis=-1)) #Batch x K x 2
            log_q_y = tf.log(1e-10+q_y / tf.reshape(tf.stack([self.bern, 1. - self.bern]), [1, 1, -1])) #1x1x2
            self.kl_bernoulli = tf.reduce_sum(tf.reduce_sum(q_y * log_q_y, axis=-1), axis=-1)
            self.kl_bernoulli = self.mask * self.kl_bernoulli

        with tf.variable_scope('encoder_for_alpha'):

            self.mean = tf.contrib.layers.batch_norm(utils.linear(self.enc_vec, self.n_topic, scope='mean'))
            self.mean = tf.nn.softplus(self.mean)
            self.alpha = tf.multiply(self.mean,self.lam)
            self.alpha = tf.maximum(self.min_alpha,self.alpha)

            # Dirichlet prior alpha0
            self.prior = tf.ones((batch_size, self.n_topic), dtype=tf.float32, name='prior') * dir_prior
            self.prior = tf.multiply(self.prior,self.lam)
            self.prior = tf.maximum(self.min_alpha, self.prior)

            self.kl_dirichlet = tf.lgamma(tf.reduce_sum(self.alpha, axis=1)) - tf.lgamma(
                tf.reduce_sum(self.prior, axis=1))
            self.kl_dirichlet -= tf.reduce_sum(tf.lgamma(self.alpha+1e-10), axis=1)
            self.kl_dirichlet += tf.reduce_sum(tf.lgamma(self.prior), axis=1)
            minus = self.alpha - self.prior
            test = tf.reduce_sum(tf.multiply(minus, tf.digamma(self.alpha+1e-10) - tf.reshape(
                tf.digamma(tf.reduce_sum(self.alpha, 1)), (batch_size, 1))), 1)
            self.kl_dirichlet += test
            self.kl_dirichlet = self.mask * self.kl_dirichlet

        with tf.variable_scope('decoder'):
            # single sample
            gam = tf.squeeze(tf.random_gamma(shape=(1,), alpha=self.alpha + tf.to_float(self.B)))
            eps = tf.stop_gradient(calc_epsilon(gam, self.alpha + tf.to_float(self.B)))
            # uniform variables for shape augmentation of gamma
            u = tf.random_uniform((self.B, batch_size, self.n_topic))
            with tf.variable_scope('prob'):
                # this is the sampled gamma for this document, boosted to reduce the variance of the gradient
                self.doc_vec = gamma_h_boosted(eps, u, self.alpha, self.B)
                # normalize
                self.doc_vec = tf.div(gam, tf.reshape(tf.reduce_sum(gam, 1), (-1, 1)))
                self.doc_vec.set_shape(self.alpha.get_shape())

            # reconstruction
            logits = tf.nn.log_softmax(tf.contrib.layers.batch_norm(
                    utils.linear(self.doc_vec, self.vocab_size, scope='projection', no_bias=True)))
            self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)

        self.objective = self.recons_loss + self.warm_up * (self.kl_dirichlet + self.kl_bernoulli)
        self.true_objective = self.recons_loss + self.kl_dirichlet + self.kl_bernoulli

        fullvars = tf.trainable_variables()
        enc_trans_vars = utils.variable_parser(fullvars, 'transform')
        enc_bern_vars = utils.variable_parser(fullvars, 'encoder_for_bern')
        enc_alpha_vars = utils.variable_parser(fullvars, 'encoder_for_alpha')
        dec_vars = utils.variable_parser(fullvars, 'decoder')

        # this is the standard gradient for the reconstruction network
        dec_grads = tf.gradients(self.objective, dec_vars)

        # redefine kld and recons_loss for proper gradient back propagation
        gammas = gamma_h_boosted(eps, u, self.alpha, self.B)
        self.doc_vec = tf.div(gammas, tf.reshape(tf.reduce_sum(gammas, 1), (-1, 1)))
        self.doc_vec.set_shape(self.alpha.get_shape())
        with tf.variable_scope("decoder", reuse=True):
            logits2 = tf.nn.log_softmax(tf.contrib.layers.batch_norm(
                utils.linear(self.doc_vec, self.vocab_size, scope='projection', no_bias=True)))
            self.recons_loss2 = -tf.reduce_sum(tf.multiply(logits2, self.x), 1)

        dir_kl_grad_alpha = tf.gradients(self.kl_dirichlet, enc_alpha_vars)
        dir_kl_grad_bern = tf.gradients(self.kl_dirichlet, enc_bern_vars)
        ber_kl_grad_bern = tf.gradients(self.kl_bernoulli, enc_bern_vars)
        dir_kl_grad_trans = tf.gradients(self.kl_dirichlet, enc_trans_vars)
        ber_kl_grad_trans = tf.gradients(self.kl_bernoulli, enc_trans_vars)

        g_rep_alpha = tf.gradients(self.recons_loss2, enc_alpha_vars)
        g_rep_bern = tf.gradients(self.recons_loss2, enc_bern_vars)
        g_rep_trans = tf.gradients(self.recons_loss2, enc_trans_vars)


        enc_trans_grads = [g_r + self.warm_up * (g_d + g_b)
                           for g_r, g_d, g_b in zip(g_rep_trans, dir_kl_grad_trans, ber_kl_grad_trans)]
        enc_bern_grads = [g_r + self.warm_up * (g_d + g_b)
                          for g_r, g_d, g_b in zip(g_rep_bern, dir_kl_grad_bern,ber_kl_grad_bern)]
        enc_alpha_grads = [g_r + self.warm_up * g_d
                           for g_r, g_d in zip(g_rep_alpha, dir_kl_grad_alpha)]

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.adam_beta1,
                                           beta2=self.adam_beta2)
        self.optim_enc = optimizer.apply_gradients(list(zip(enc_trans_grads, enc_trans_vars))
                                                   + list(zip(enc_bern_grads, enc_bern_vars))
                                                   + list(zip(enc_alpha_grads, enc_alpha_vars))
                                                   )
        self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))
        self.optim_all = optimizer.apply_gradients(list(zip(enc_trans_grads, enc_trans_vars))
                                                   + list(zip(enc_bern_grads, enc_bern_vars))
                                                   + list(zip(enc_alpha_grads, enc_alpha_vars))
                                                   + list(zip(dec_grads, dec_vars)))

# Transformation and its derivative
def gamma_h(epsilon, alpha):
    """
    Reparameterization for gamma rejection sampler without shape augmentation.
    """
    b = alpha - 1. / 3.
    c = 1. / tf.sqrt(9. * b)
    v = 1. + epsilon * c

    return b * (v ** 3)

def gamma_h_boosted(epsilon, u, alpha, model_B):
    """
    Reparameterization for gamma rejection sampler with shape augmentation.
    """
    # B = u.shape.dims[0] #u has shape of alpha plus one dimension for B
    B = tf.shape(u)[0]
    K = alpha.shape[1]  # (batch_size,K)
    r = tf.range(B)
    rm = tf.to_float(tf.reshape(r, [-1, 1, 1]))  # dim Bx1x1
    alpha_vec = tf.reshape(tf.tile(alpha, (B, 1)), (model_B, -1, K)) + rm  # dim BxBSxK + dim Bx1
    u_pow = tf.pow(u, 1. / alpha_vec) + 1e-10
    gammah = gamma_h(epsilon, alpha + tf.to_float(B))
    return tf.reduce_prod(u_pow, axis=0) * gammah


def calc_epsilon(gamma, alpha):
    return tf.sqrt(9. * alpha - 3.) * (tf.pow(gamma / (alpha - 1. / 3.), 1. / 3.) - 1.)


def train(sess, model,
          batch_size,
          vocab_size,
          data_set,
          counts,
          train_test_idxes,
          alternate_epochs=10,
          B=10,
          warm_up_period=100):

    np.random.shuffle(train_test_idxes[0])
    train_set = data_set.copy()[train_test_idxes[0]]
    train_count = counts[train_test_idxes[0]]
    test_set = data_set.copy()[train_test_idxes[1]]
    test_count = counts[train_test_idxes[1]]
    train_size = len(train_set)
    validation_size = int(train_size * 0.1)

    dev_set = train_set[:validation_size]
    dev_count = train_count[:validation_size]
    train_set = train_set[validation_size:]
    train_count = train_count[validation_size:]

    dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False)
    test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)

    optimize_jointly = False
    warm_up = 0.
    tem_init = 5.
    tem = tem_init
    tem_min = 1.
    min_alpha = 0.00001
    curr_B = B

    best_loss = np.inf
    early_stopping_iters = 20
    no_improvement_iters = 0
    stopped = False
    epoch = -1

    while not stopped:
        epoch += 1

        train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)
        if warm_up < 1.:
            warm_up += 1. / warm_up_period
        else:
            warm_up = 1.

        if tem > tem_min:
            tem -= tem_init / warm_up_period
        else:
            tem = tem_min
        # -------------------------------
        # train
        for switch in range(0, 2):
            if optimize_jointly:
                optim = model.optim_all
                print_mode = 'updating encoder and decoder'
            elif switch == 0:
                optim = model.optim_dec
                print_mode = 'updating decoder'
            else:
                optim = model.optim_enc
                print_mode = 'updating encoder'

            for i in range(alternate_epochs):
                loss_sum = 0.0
                dir_kld_sum_train = 0.0
                bern_kld_sum_train=0.0
                recon_sum = 0.0
                for idx_batch in train_batches:
                    data_batch, count_batch, mask = utils.fetch_data(
                        train_set, train_count, idx_batch, vocab_size)
                    input_feed = {model.x.name: data_batch, model.mask.name: mask, model.keep_prob.name: 0.75,model.temp.name:tem,
                                  model.warm_up.name: warm_up, model.min_alpha.name: min_alpha, model.B.name: curr_B}
                    _, (loss, recon,dir_kld_train, bern_kl_train) = sess.run((optim,
                                                                                      [model.true_objective,
                                                                                       model.recons_loss,
                                                                                       model.kl_dirichlet,
                                                                                       model.kl_bernoulli]),
                                                                                     input_feed)
                    loss_sum += np.sum(loss)/np.sum(mask)
                    dir_kld_sum_train += np.sum(dir_kld_train)/np.sum(mask)
                    bern_kld_sum_train += np.sum(bern_kl_train)/np.sum(mask)
                    recon_sum += np.sum(recon)/np.sum(mask)
                print_loss = loss_sum / len(train_batches)
                print_rec_loss = recon_sum / len(train_batches)
                print_dir_kld_train = dir_kld_sum_train / len(train_batches)
                print_bern_kld_train = bern_kld_sum_train / len(train_batches)
                print('| Epoch train: {:d} |'.format(epoch + 1),
                      print_mode, '{:d}'.format(i),
                      '| Loss: {:.5}'.format(print_loss),
                      '| Recon Loss: {:.5}'.format(print_rec_loss),
                      '| KLD dir: {:.5}'.format(print_dir_kld_train),
                      '| KLD bern: {:.5f}'.format(print_bern_kld_train),
                      )

        # -------------------------------
        # dev
        loss_sum = 0.0
        dir_sum_dev = 0.0
        bern_sum_dev = 0.0
        recon_sum = 0.0
        for idx_batch in dev_batches:
            data_batch, count_batch, mask = utils.fetch_data(
                dev_set, dev_count, idx_batch, vocab_size)
            input_feed = {model.x.name: data_batch, model.mask.name: mask, model.keep_prob.name: 1.0,model.temp.name:tem,
                          model.warm_up.name: 1.0, model.min_alpha.name: min_alpha, model.B.name: B}
            loss, recon, dir_kld, bern_kld = sess.run(
                [model.objective, model.recons_loss, model.kl_dirichlet,model.kl_bernoulli],
                input_feed)
            loss_sum += np.sum(loss)/np.sum(mask)
            dir_sum_dev += np.sum(dir_kld)/np.sum(mask)
            bern_sum_dev += np.sum(bern_kld)/np.sum(mask)
            recon_sum += np.sum(recon)/np.sum(mask)
        print_dir_dev = dir_sum_dev / len(dev_batches)
        print_bern_dev = bern_sum_dev / len(dev_batches)
        print_recon = recon_sum/len(dev_batches)
        print_loss = loss_sum / len(dev_batches)
        if print_loss < best_loss:
            no_improvement_iters = 0
            best_loss = print_loss
            # check on validation set, if ppx better-> save improved model
            tf.train.Saver().save(sess, 'models/improved_model')

        else:
            no_improvement_iters += 1
            print('no_improvement_iters', no_improvement_iters, 'best loss', best_loss)
            if no_improvement_iters >= early_stopping_iters:
                # if model has not improved for 30 iterations, stop training
                ###########STOP TRAINING############
                stopped = True
                print('stop training after', epoch, 'iterations,no_improvement_iters', no_improvement_iters)
                ###########LOAD BEST MODEL##########
                print('load stored model')
                tf.train.Saver().restore(sess, 'models/improved_model')
        print('| Epoch dev: {:d} |'.format(epoch + 1),
              '| Loss: {:.5}'.format(print_loss),
              '| KLD dir: {:.5}'.format(print_dir_dev),
              '| KLD bern: {:.5}'.format(print_bern_dev),
              '| Recon Loss: {:.5}'.format(print_recon)
              )

        # -------------------------------
        # test
        if FLAGS.test:
            loss_sum = 0.0
            dir_sum_test = 0.0
            bern_sum_test = 0.0
            recon_sum = 0.0
            for idx_batch in test_batches:
                data_batch, count_batch, mask = utils.fetch_data(
                    test_set, test_count, idx_batch, vocab_size)
                input_feed = {model.x.name: data_batch, model.mask.name: mask, model.keep_prob.name: 1.0,model.temp.name:tem,
                              model.warm_up.name: 1.0, model.min_alpha.name: min_alpha, model.B.name: B}
                loss, recon, dir_kld_test, bern_kld_test = sess.run(
                    [model.objective, model.recons_loss, model.kl_dirichlet,model.kl_bernoulli],
                    input_feed)
                loss_sum += np.sum(loss)/np.sum(mask)
                dir_sum_test += np.sum(dir_kld_test)/np.sum(mask)
                bern_sum_test += np.sum(bern_kld_test) / np.sum(mask)
                recon_sum += np.sum(recon)/np.sum(mask)
            print_loss = loss_sum / len(test_batches)
            print_dir_test = dir_sum_test / len(dev_batches)
            print_bern_test = bern_sum_test / len(dev_batches)
            print_recon = recon_sum / len(dev_batches)
            print('| Epoch test: {:d} |'.format(epoch + 1),
                  '| Loss: {:.5}'.format(print_loss),
                  '| KLD dir: {:.5}'.format(print_dir_test),
                  '| KLD bern: {:.5}'.format(print_bern_test),
                  '| Recon Loss: {:.5}'.format(print_recon),
                  )
    dec_vars = utils.variable_parser(tf.trainable_variables(), 'decoder')
    phi = dec_vars[0]
    phi = sess.run(phi)

    all_batches = utils.create_batches(len(data_set), batch_size, shuffle=False)
    theta = []
    topic_selected = []
    for idx_batch in all_batches:
        data_batch, count_batch, mask = utils.fetch_data(
            data_set, counts, idx_batch, vocab_size)
        input_feed = {model.x.name: data_batch, model.mask.name: mask, model.keep_prob.name: 1.,model.temp.name:tem,
                      model.min_alpha.name: min_alpha, model.B.name: curr_B}
        doc_vec, lamb = sess.run([model.doc_vec,model.lam], input_feed)
        theta.extend(doc_vec.tolist())
        topic_selected.extend(lamb.tolist())
    n_topics_selected = np.mean(np.sum(np.round(topic_selected), axis=1))
    return phi, theta, topic_selected,n_topics_selected


def myrelu(features):
    return tf.maximum(features, 0.0)


def texts_process(texts):
    id2word = corpora.Dictionary(texts, )
    corpus = [np.array([id2word.token2id[item] for item in text]) for text in texts]  # word-id sequences
    corpus = np.array([doc.astype('int32') for doc in corpus if np.sum(doc) != 0])
    return id2word, corpus


def onehot(data, min_length):
    # turning the sequence data into bag-of-word
    return np.bincount(data, minlength=min_length)


def main(argv=None):
    if FLAGS.non_linearity == 'tanh':
        non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
        non_linearity = tf.nn.sigmoid
    else:
        non_linearity = myrelu

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--adam_beta1', default=0.9, type=float)
    argparser.add_argument('--adam_beta2', default=0.999, type=float)
    argparser.add_argument('--learning_rate', default=1e-4, type=float)
    argparser.add_argument('--bs', default=200, type=int)
    argparser.add_argument('--dir_prior', default=0.02, type=float)
    argparser.add_argument('--bern_prior', default=0.05, type=float)
    argparser.add_argument('--B', default=10, type=int)
    argparser.add_argument('--n_topic', default=50, type=int)
    argparser.add_argument('--warm_up_period', default=100, type=int)
    argparser.add_argument('--data_dir', default='./data/20ng', type=str)
    argparser.add_argument('--data_name', default="", type=str)

    args = argparser.parse_args()
    adam_beta1 = args.adam_beta1
    adam_beta2 = args.adam_beta2
    learning_rate = args.learning_rate
    dir_prior = args.dir_prior
    bern = args.bern_prior
    B = args.B
    warm_up_period = args.warm_up_period
    n_topic = args.n_topic

    train_url = args.data_dir + '/train.txt'
    test_url = os.path.join(args.data_dir, 'test.txt')

    with open(train_url, 'r') as f:
        text_tr = [line.strip('\n').split(' ') for line in f.readlines()]
    with open(test_url, 'r') as f:
        text_te = [line.strip('\n').split(' ') for line in f.readlines()]
    tr_len, te_len = len(text_tr), len(text_te)
    id2word, corpus = texts_process(text_tr + text_te)
    num_words = id2word.__len__()
    corpus = np.array(
        [onehot(doc.astype('int'), num_words) for doc in corpus])  # bag-of-word format
    counts = np.sum(corpus, 1)
    train_test_idxes = list(range(tr_len)), list(range(tr_len, tr_len + te_len))

    vocab_size = num_words
    for i in range(1,4):
        sparentm = SpareNTM(
                    vocab_size=vocab_size,
                    n_hidden=FLAGS.n_hidden,
                    n_topic=n_topic,
                    learning_rate=learning_rate,
                    batch_size=args.bs,
                    non_linearity=non_linearity,
                    adam_beta1=adam_beta1,
                    adam_beta2=adam_beta2,
                    dir_prior=dir_prior,
                   bern_prior=bern)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        phi, theta, topic_selected, n_seleted = train(sess, sparentm, args.bs, vocab_size, corpus, counts, train_test_idxes,
                           B=B, warm_up_period=warm_up_period)

        with open(FLAGS.topN_file,'w') as f:
            for x in range(n_topic):
                twords = [(n, phi[x][n]) for n in range(vocab_size)]
                twords.sort(key=lambda i: i[1], reverse=True)
                for y in range(20):
                    word = id2word[twords[y][0]]
                    f.write(word + '\t')
                f.write('\n')

        with open(FLAGS.theta_file,'w') as f:
            for x in range(len(corpus)):
                for y in range(n_topic):
                    f.write(str(theta[x][y]) + '\t')
                f.write('\n')

if __name__ == '__main__':
    tf.app.run()
