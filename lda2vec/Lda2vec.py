import tensorflow as tf
import numpy as np
import lda2vec.word_embedding as W
import lda2vec.embedding_mixture as M
import lda2vec.dirichlet_likelihood as DL
from lda2vec import utils
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Lda2vec:
    RESTORE_KEY = 'to_restore'

    def __init__(self, num_unique_documents, vocab_size, num_topics, freqs=None, 
                 save_graph_def=True, embedding_size=128, num_sampled=40,
                 learning_rate=0.001, lmbda=200.0, alpha=None, power=0.75, batch_size=500, logdir='logdir',
                 restore=False, fixed_words=False, factors_in=None, pretrained_embeddings=None):
        """Summary
        
        Args:
            num_unique_documents (int): Number of unique documents in your dataset
            vocab_size (int): Number of unique words/tokens in your dataset
            num_topics (int): The set number of topics to cluster your data into
            freqs (list, optional): Python list of length vocab_size with frequencies of each token
            save_graph_def (bool, optional): If true, we will save the graph to logdir
            embedding_size (int, optional): Dimension of the embeddings. This will be shared between docs, words, and topics.
            num_sampled (int, optional): Negative sampling number for NCE Loss.
            learning_rate (float, optional): Learning rate for optimizer
            lmbda (float, optional): Strength of dirichlet prior
            alpha (None, optional): alpha of dirichlet process (defaults to 1/n_topics)
            power (float, optional): unigram sampler distortion
            batch_size (int, optional): Batch size coming into model
            logdir (str, optional): Location for models to be saved - note, we will append on the datetime too on each run
            restore (bool, optional): When True, we will restore the model from the logdir parameter's location
            fixed_words (bool, optional): Description
            factors_in (None, optional): Pretrained Topic Embedding (shape should be [num_topics, embedding_size])
            pretrained_embeddings (None, optional): Description
        
        """
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sesh = tf.Session(config=self.config)
        self.moving_avgs = tf.train.ExponentialMovingAverage(0.9)
        self.num_unique_documents = num_unique_documents
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.freqs = freqs
        self.save_graph_def = save_graph_def
        self.logdir = logdir
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.alpha = alpha
        self.power = power
        self.batch_size = batch_size
        self.pretrained_embeddings = pretrained_embeddings
        self.factors_in = factors_in
        self.compute_normed = False
        self.fixed_words = fixed_words


        if not restore:
            self.date = datetime.now().strftime('%y%m%d_%H%M')
            self.logdir = ('{}_{}').format(self.logdir, self.date)

            # Load pretrained embeddings if provided.
            if type(pretrained_embeddings) !=None:
                W_in = tf.constant(pretrained_embeddings, name="word_embedding") if fixed_words else tf.get_variable("word_embedding", shape=[self.vocab_size,self.embedding_size], initializer=tf.constant_initializer(pretrained_embeddings))


            # Initialize the word embedding
            self.w_embed = W.Word_Embedding(self.embedding_size, self.vocab_size, self.num_sampled,
                                            W_in=W_in, freqs=self.freqs,
                                            power=self.power)
            # Initialize the Topic-Document Mixture
            self.mixture = M.EmbedMixture(self.num_unique_documents, self.num_topics, self.embedding_size)


            # Builds the graph and returns variables within it
            handles = self._build_graph()


            for handle in handles:
                tf.add_to_collection(Lda2vec.RESTORE_KEY, handle)

            # Add Word Embedding Variables to collection
            tf.add_to_collection(Lda2vec.RESTORE_KEY, self.w_embed.embedding)
            tf.add_to_collection(Lda2vec.RESTORE_KEY, self.w_embed.nce_weights)
            tf.add_to_collection(Lda2vec.RESTORE_KEY, self.w_embed.nce_biases)

            # Add Doc Mixture Variables to collection
            tf.add_to_collection(Lda2vec.RESTORE_KEY, self.mixture.doc_embedding)
            tf.add_to_collection(Lda2vec.RESTORE_KEY, self.mixture.topic_embedding)

            (self.x, self.y, self.docs, self.step, self.switch_loss,
            self.word_context, self.doc_context, self.loss_word2vec,
            self.fraction, self.loss_lda, self.loss, self.loss_avgs_op,
            self.optimizer, self.merged) = handles

        else:
            meta_graph = logdir + '/model.ckpt'
            tf.train.import_meta_graph(meta_graph + '.meta').restore(self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(Lda2vec.RESTORE_KEY)

            (self.x, self.y, self.docs, self.step, self.switch_loss,
            self.word_context, self.doc_context, self.loss_word2vec,
            self.fraction, self.loss_lda, self.loss, self.loss_avgs_op,
            self.optimizer, self.merged, embedding, nce_weights, nce_biases,
            doc_embedding, topic_embedding) = handles

            self.w_embed = W.Word_Embedding(self.embedding_size, self.vocab_size, self.num_sampled,
                                            W_in=embedding, freqs=self.freqs,
                                            power=self.power,
                                            nce_w_in=nce_weights,
                                            nce_b_in=nce_biases)

            # Initialize the Topic-Document Mixture
            self.mixture = M.EmbedMixture(self.num_unique_documents,
                                          self.num_topics,
                                          self.embedding_size,
                                          W_in=doc_embedding,
                                          factors_in=topic_embedding)

    def prior(self):
        """Computes Dirichlet Prior.
        
        Returns:
            TYPE: Dirichlet Prior Value
        """
        doc_prior = DL.dirichlet_likelihood(self.mixture.doc_embedding, alpha=self.alpha)

        return doc_prior

    def _build_graph(self):
        """Builds the Lda2vec model graph.
        """
        # Model Inputs
        # Pivot Words
        x = tf.placeholder(tf.int32, shape=[None], name='x_pivot_idxs')
        # Context/Target Words
        y = tf.placeholder(tf.int64, shape=[None], name='y_target_idxs')
        # Document ID
        docs = tf.placeholder(tf.int32, shape=[None], name='doc_ids')

        # Global Step
        step = tf.Variable(0, trainable=False, name='global_step')
        # What epoch should we switch on lda loss?
        switch_loss = tf.Variable(0, trainable=False)
        # Word embedding lookup
        word_context = tf.nn.embedding_lookup(self.w_embed.embedding, x, name='word_embed_lookup')
        # Document Context via document ID lookup
        doc_context = self.mixture(doc_ids=docs)

        # Compile word + doc context in list and add them together
        contexts_to_add=[word_context, doc_context]
        context = tf.add_n(contexts_to_add, name='context_vector')

        # Compute Word2Vec Loss
        with tf.name_scope('nce_loss'):
            loss_word2vec = self.w_embed(context, y)
            tf.summary.scalar('nce_loss', loss_word2vec)
        # Compute LDA Loss
        with tf.name_scope('lda_loss'):
            fraction = tf.Variable(1, trainable=False, dtype=tf.float32, name='fraction')
            loss_lda = self.lmbda * fraction * self.prior()
            tf.summary.scalar('lda_loss', loss_lda)

        # Determine if we should be using only word2vec loss or if we should add in LDA loss based on switch_loss Variable
        loss = tf.cond(step < switch_loss, lambda: loss_word2vec, lambda: loss_word2vec + loss_lda)
        # Add current loss to moving average of loss
        loss_avgs_op = self.moving_avgs.apply([loss_lda, loss_word2vec, loss])
        
        # Init the optimizer
        with tf.control_dependencies([loss_avgs_op]):
            optimizer = tf.contrib.layers.optimize_loss(loss,
                                                        tf.train.get_global_step(),
                                                        self.learning_rate,
                                                        'Adam',
                                                        name='Optimizer')
        
        # Initialize all variables
        self.sesh.run(tf.global_variables_initializer(), options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
        
        # Create a merged summary of variables
        merged = tf.summary.merge_all()

        to_return = [x, y, docs, step, switch_loss, word_context, doc_context,
                     loss_word2vec, fraction, loss_lda, loss, loss_avgs_op, optimizer, merged]


        return to_return


    def train(self, pivot_words, target_words, doc_ids, data_size, num_epochs, switch_loss_epoch=0,
              save_every=1, report_every=1, print_topics_every=5, idx_to_word=None):
        """Train the Lda2vec Model. pivot_words, target_words, and doc_ids should be
        the same size.
        
        Args:
            pivot_words (np.array): Array of word idxs corresponding to pivot words
            target_words (np.array): Array of word idxs corresponding to target words
            doc_ids (TYPE): Document IDs linking word idxs to their docs
            data_size (TYPE): Length of pivot_words array
            num_epochs (TYPE): Number of epochs to train model
            switch_loss_epoch (int, optional): Epoch to switch on LDA loss. LDA loss not learned
                                               until this epoch
            save_every (int, optional): Save model every "save_every" epoch
            report_every (int, optional): Report model metrics every "report_every" epoch.
            print_topics_every (int, optional): Print top 10 words in each topic every "print_topics_every" 
            idx_to_word (None, optional): IDX to word mapping - Required if you want to see word-topic membership
        """
        # Calculate fraction used in DL Loss calculation
        temp_fraction = self.batch_size * 1.0 / data_size
        # Assign the fraction placeholder variable with the value we calculated
        self.sesh.run(tf.assign(self.fraction, temp_fraction))

        # Calculate the number of iterations per epoch so we can figure out when to switch the loss
        iters_per_epoch = int(data_size / self.batch_size) + np.ceil(data_size % self.batch_size)
        # Calculate what step we would be on @ the switch loss epoch
        switch_loss_step = iters_per_epoch * switch_loss_epoch
        # Assign the switch loss variable with the step we just calculated
        self.sesh.run(tf.assign(self.switch_loss, switch_loss_step))

        if self.save_graph_def:
            # Initialize a tensorflow Saver object
            saver = tf.train.Saver()
            # Initialize a tensorflow summary writer so we can save logs
            writer = tf.summary.FileWriter(self.logdir + '/', graph=self.sesh.graph)

        # Iterate over the number of epochs we want to train for
        for e in range(num_epochs):
            print('\nEPOCH:', e + 1)
            # Get a batch worth of data
            for p, t, d in utils.chunks(self.batch_size, pivot_words, target_words, doc_ids):
                
                # Create the feed dict from the batched data
                feed_dict = {self.x: p, self.y: t, self.docs: d}
                
                # Values we want to fetch whenever we run the model
                fetches = [self.merged, self.optimizer, self.loss,
                           self.loss_word2vec, self.loss_lda, self.step]
                
                # Run a step of the model
                summary, _, l, lw2v, llda, step = self.sesh.run(fetches, feed_dict=feed_dict)

            # Prints log every "report_every" epoch
            if (e+1) % report_every == 0:
                print('LOSS', l, 'w2v', lw2v, 'lda', llda)

            # Saves model every "save_every" epoch
            if (e+1) % save_every == 0 and self.save_graph_def:
                writer.add_summary(summary, step)
                writer.flush()
                writer.close()
                save_path = saver.save(self.sesh, self.logdir + '/model.ckpt')
                writer = tf.summary.FileWriter(self.logdir + '/', graph=self.sesh.graph)
            
            # Prints out membership of words in each topic every "print_topics_every" epoch
            if e>0 and (e+1)%print_topics_every==0:
                idxs = np.arange(self.num_topics)
                words, sims = self.get_k_closest(idxs, in_type='topic', idx_to_word=idx_to_word, k=10, verbose=True)

        # Save after all epochs are finished, but only if we didn't just save
        if self.save_graph_def and (e+1) % save_every != 0:
            writer.add_summary(summary, step)
            writer.flush()
            writer.close()        
            save_path = saver.save(self.sesh, self.logdir + '/model.ckpt')

    def compute_normed_embeds(self):
        """Normalizes embeddings so we can measure cosine similarity
        between different embedding matrixes.
        """
        self.normed_embed_dict = {}
        norm = tf.sqrt(tf.reduce_sum(self.mixture.topic_embedding ** 2, 1, keep_dims=True))
        self.normed_embed_dict['topic'] = self.mixture.topic_embedding / norm
        norm = tf.sqrt(tf.reduce_sum(self.w_embed.embedding ** 2, 1, keep_dims=True))
        self.normed_embed_dict['word'] = self.w_embed.embedding / norm
        norm = tf.sqrt(tf.reduce_sum(self.mixture.doc_embedding ** 2, 1, keep_dims=True))
        self.normed_embed_dict['doc'] = self.mixture.doc_embedding / norm
        self.idxs_in = tf.placeholder(tf.int32, shape=[None], name='idxs')
        self.compute_normed = True

    def get_k_closest(self, idxs, in_type='word', vs_type='word', k=10, idx_to_word=None, verbose=False):
        """Gets k closest vs_type embeddings for every idx of in_type embedding given.
        Options for the in_type and vs_type are ["word", "topic", "doc"].
        
        Args:
            idxs (np.array): Array of indexes you want to get similarities to
            in_type (str, optional): idxs will query this embedding matrix
            vs_type (str, optional): embeddings to compare to in_type embedding lookup
            k (int, optional): Number of vs_type embeddings to return per idx
            idx_to_word (dict, optional): IDX to word mapping
            verbose (bool, optional): Should we print out the top k words per epoch? False by default.
                                      Only prints if idx_to_word is passed too. 
        
        Returns:
            sim: Actual embeddings that are similar to each idx. shape [idxs.shape[0], k, self.embed_size]
            sim_idxs: Indexes of the sim embeddings. shape [idxs.shape[0], k]

        NOTE: Acceptable pairs include:
            word - word
            word - topic
            topic - word
            doc - doc
        """
        if self.compute_normed == False:
            self.compute_normed_embeds()
        self.batch_array = tf.nn.embedding_lookup(self.normed_embed_dict[in_type], self.idxs_in)
        self.cosine_similarity = tf.matmul(self.batch_array, tf.transpose(self.normed_embed_dict[vs_type], [1, 0]))
        feed_dict = {self.idxs_in: idxs}
        sim, sim_idxs = self.sesh.run(tf.nn.top_k(self.cosine_similarity, k=k), feed_dict=feed_dict)
        if idx_to_word:
            if verbose and vs_type=="word":
                print('---------Closest {} words to given indexes----------'.format(k))

            for i, idx in enumerate(idxs):
                if in_type == 'word':
                    in_word = idx_to_word[idx]
                else:
                    in_word = 'Topic ' + str(idx)
                vs_word_list = []
                for vs_i in range(sim_idxs[i].shape[0]):
                    vs_idx = sim_idxs[i][vs_i]
                    vs_word = idx_to_word[vs_idx]
                    vs_word_list.append(vs_word)
                if verbose and vs_type=="word":
                    print(in_word, ':', (', ').join(vs_word_list))

        return (sim, sim_idxs)

    def save_weights_to_file(self, word_embed_path='word_weights', doc_embed_path='doc_weights',
                             topic_embed_path='topic_weights'):
        """Saves embedding matrixes to file.
        
        Args:
            word_embed_path (str, optional): Path and name where you want to save word embeddings
            doc_embed_path (str, optional): Path and name where you want to save doc embeddings
            topic_embed_path (str, optional): Path and name where you want to save topic embeddings
        """
        word_embeds = self.sesh.run(self.word_embedding)
        np.save(word_embed_path, word_embeds)
        doc_embeds = self.sesh.run(self.doc_embedding)
        np.save(doc_embed_path, doc_embeds)
        topic_embeds = self.sesh.run(self.topic_embedding)
        np.save(topic_embed_path, topic_embeds)