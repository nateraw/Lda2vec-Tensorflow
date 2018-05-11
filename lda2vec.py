import tensorflow as tf
import word_embedding as W
import embedding_mixture as M
import dirichlet_likelihood as DL


class Lda2vec():

    RESTORE_KEY = "to_restore"

    def __init__(self, num_unique_documents, vocab_size,
                 num_topics, freqs=None, load_embeds=False, pretrained_embeddings=False,
                 save_graph_def=True, embedding_size=128, num_sampled=40,
                 learning_rate=1E-3, lmbda=150., alpha=None, power=.75, batch_size=500,
                 logdir="logdir", restore=False):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sesh = tf.Session(config=self.config)
        self.moving_avgs = tf.train.ExponentialMovingAverage(0.9)

        self.num_unique_documents = num_unique_documents
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.freqs = freqs
        self.load_embeds = load_embeds
        self.pretrained_embeddings = pretrained_embeddings
        self.save_graph_def = save_graph_def
        self.logdir = logdir
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.alpha = alpha
        self.power = power
        self.batch_size = batch_size

        # This will be set to true if compute_normed_embeds is run so it doesnt get run more than once
        self.compute_normed = False

        if not restore:
            self.w_embed = W.Word_Embedding(self.embedding_size,
                                            self.vocab_size,
                                            self.num_sampled,
                                            load_embeds=self.load_embeds,
                                            pretrained_embeddings=self.pretrained_embeddings,
                                            freqs=self.freqs,
                                            power=self.power)

            # Doc and topic mixture
            self.mixture = M.EmbedMixture(self.num_unique_documents,
                                          self.num_topics,
                                          self.embedding_size)

            handles = self._build_graph()

            for handle in handles:
                tf.add_to_collection(Lda2vec.RESTORE_KEY, handle)

            (self.x, self.y, self.docs, self.step, self.switch_loss, self.pivot,
             self.doc, self.dropout, self.context, self.loss_word2vec, self.fraction, self.loss_lda, self.loss,
             self.loss_avgs_op, self.optimizer, self.doc_embedding, self.topic_embedding,
             self.word_embedding, self.nce_weights, self.nce_biases, self.merged) = handles
        else:
            meta_graph = logdir + "/model.ckpt"
            tf.train.import_meta_graph(meta_graph + ".meta").restore(self.sesh, meta_graph)

            handles = self.sesh.graph.get_collection(Lda2vec.RESTORE_KEY)

            (self.x, self.y, self.docs, self.step, self.switch_loss, self.pivot,
             self.doc, self.dropout, self.context, self.loss_word2vec, self.fraction, self.loss_lda, self.loss,
             self.loss_avgs_op, self.optimizer, self.doc_embedding, self.topic_embedding,
             self.word_embedding, self.nce_weights, self.nce_biases, self.merged) = handles

    def prior(self):
        # defaults to inialization with uniform prior (1/n_topics)
        return DL.dirichlet_likelihood(self.mixture.Doc_Embedding, alpha=self.alpha)

    def _build_graph(self):
        """
        Args:

        x = pivot words (int)
        y = context words (int)
        docs = docs at pivot (int)
        """

        # Pivot Words
        x = tf.placeholder(tf.int32, shape=[None, ], name="x_pivot_idxs")

        # Target Words
        y = tf.placeholder(tf.int64, shape=[None, ], name="y_target_idxs")

        # Document ID
        docs = tf.placeholder(tf.int32, shape=[None, ], name="doc_ids")

        # Graph's current step
        step = tf.Variable(0, trainable=False, name="global_step")
        # Step to turn on document mixture loss
        switch_loss = tf.Variable(0, trainable=False)

        # Lookup pivot word idxs
        pivot = tf.nn.embedding_lookup(self.w_embed.Embedding, x, name="word_embed_lookup")

        # Lookup document embed
        doc = self.mixture(doc_ids=docs)

        # switch_loss = tf.Variable(0, trainable=False, name="switch_loss")

        # Context is sum of doc(mixture projected onto topics) & pivot embedding
        dropout = self.mixture.dropout

        # doc_context  = tf.nn.dropout(doc, dropout, name="doc_context")
        # word_context = tf.nn.dropout(pivot, dropout, name="word_context")
        # context = tf.add(word_context, doc_context, name="context_vector")
        doc_context = doc
        word_context = pivot
        context = tf.add(word_context, doc_context, name="context_vector")

        # Targets
        with tf.name_scope("nce_loss"):
            loss_word2vec = self.w_embed(context, y)
            tf.summary.scalar('nce_loss', loss_word2vec)

        # Dirichlet loss
        with tf.name_scope("lda_loss"):
            fraction = tf.Variable(1, trainable=False, dtype=tf.float32, name="fraction")
            loss_lda = self.lmbda * fraction * self.prior()
            tf.summary.scalar('lda_loss', loss_lda)

        loss = tf.cond(step < switch_loss,
                       lambda: loss_word2vec,
                       lambda: loss_word2vec + loss_lda)

        loss_avgs_op = self.moving_avgs.apply([loss_lda, loss_word2vec, loss])

        with tf.control_dependencies([loss_avgs_op]):
            optimizer = tf.contrib.layers.optimize_loss(loss,
                                                        tf.train.get_global_step(),
                                                        self.learning_rate,
                                                        "Adam",
                                                        clip_gradients=5.,
                                                        name="Optimizer")

        self.sesh.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()

        return (x, y, docs, step, switch_loss, pivot, doc, dropout, context,
                loss_word2vec, fraction, loss_lda, loss, loss_avgs_op, optimizer, self.mixture.Doc_Embedding,
                self.mixture.topic_embedding, self.w_embed.Embedding, self.w_embed.nce_weights, self.w_embed.nce_biases, merged)

    def train(self, pivot_words, target_words, doc_ids, data_size, num_epochs):
        '''
        Args:
        pivot_words  - List of pivot word indexes (int array)
        target_words - List of target words indexes (int array)
        doc_ids      - List of Document Id's linked to pivot and target words (int array)
        data_size    - Total amount of unique sentences (or data) before splitting into pivot/target words (int)
        num_epochs   - Integer noting how many epochs to train for. 
        '''
        # Fraction adjusts the loss term to be proportional to minibatch size
        temp_fraction = self.batch_size / data_size
        # Run the session to assign the fraction to the graph
        self.sesh.run(tf.assign(self.fraction, temp_fraction))

        num_batches = data_size // self.batch_size

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter("logdir" + "/", graph=self.sesh.graph)

        for e in range(num_epochs):
            print("\nEPOCH:", e + 1)
            for i in range(num_batches):
                x_batch = pivot_words[i * self.batch_size:i * self.batch_size + self.batch_size]
                y_batch = target_words[i * self.batch_size:i * self.batch_size + self.batch_size]
                doc_batch = doc_ids[i * self.batch_size:i * self.batch_size + self.batch_size]
                summary, _, l, lw2v, llda, step = self.sesh.run([self.merged, self.optimizer, self.loss, self.loss_word2vec, self.loss_lda, self.step],
                                                                feed_dict={self.x: x_batch, self.y: y_batch, self.docs: doc_batch})

                if step > 0 and step % 100 == 0:
                    print("STEP", step, "LOSS", l, "w2v", lw2v, "lda", llda)
                if step > 0 and step % 5000 == 0:
                    writer.add_summary(summary, step)
                    writer.flush()
                    writer.close()
                    save_path = saver.save(self.sesh, self.logdir + "/model.ckpt")
                    writer = tf.summary.FileWriter(self.logdir + "/", graph=self.sesh.graph)
        save_path = saver.save(self.sesh, self.logdir + "/model.ckpt")

    def predict(self, pivot_words, doc_ids, temp_batch_size):
        context = self.sesh.run([self.context], feed_dict={self.x: pivot_words})


    def compute_normed_embeds(self):
        # Compute and save normalized embedding matrixes for computing similarity
        self.normed_embed_dict = {}
        self.normed_embed_dict["topic"] = tf.nn.l2_normalize(self.topic_embedding, axis=1)
        self.normed_embed_dict["word"] = tf.nn.l2_normalize(self.word_embedding, axis=1)
        self.normed_embed_dict["doc"] = tf.nn.l2_normalize(self.doc_embedding, axis=1)

        self.idxs_in = tf.placeholder(tf.int32,
                                      shape=[None],  # None enables variable batch size
                                      name="idxs")  # doc or topic or word

        self.compute_normed = True

    def get_k_closest(self, idxs, in_type="word", vs_type="word", k=10, idx_to_word=None):
        """
        Args:
        idxs - numpy array of indexes to check similarity to
        in_type - string denoting what kind of embedding to check
                  similarity to. Options are "word", "doc", and "topic"
        out_type - same as above, except it will be what we are comparing the
                   in indexes to.
        k - Number of closest examples to get
        idx_to_word - index to word dictionary mapping. If passed, it will translate the indexes.

        NOTE: Acceptable pairs include
        word - word
        word - topic
        topic - word
        doc - doc

        """
        if self.compute_normed == False:
            self.compute_normed_embeds()

        batch_array = tf.nn.embedding_lookup(self.normed_embed_dict[vs_type], self.idxs_in)

        cosine_similarity = tf.matmul(batch_array, tf.transpose(self.normed_embed_dict[in_type],
                                                                [1, 0]))

        feed_dict = {self.idxs_in: idxs}

        sim, sim_idxs = self.sesh.run(tf.nn.top_k(cosine_similarity, k=k),
                                      feed_dict=feed_dict)

        print("---------Closest words to given indexes----------")
        # Translate to words if idx-to-word is passed
        if idx_to_word:
            for i, idx in enumerate(idxs):
                if in_type == "word":
                    in_word = idx_to_word[idx]
                else:
                    in_word = "Topic " + str(idx)
                vs_word_list = []
                for vs_i in range(sim_idxs[i].shape[0]):
                    vs_idx = sim_idxs[i][vs_i]
                    vs_word = idx_to_word[vs_idx]
                    vs_word_list.append(vs_word)
                print(in_word, ":", ", ".join(vs_word_list))

        return sim, sim_idxs
