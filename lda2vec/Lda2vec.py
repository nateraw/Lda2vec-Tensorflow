import tensorflow as tf
import numpy as np
import lda2vec.word_embedding as W
import lda2vec.embedding_mixture as M
import lda2vec.dirichlet_likelihood as DL
from datetime import datetime
import os
import sys

class Lda2vec():

    RESTORE_KEY = "to_restore"

    def __init__(self, num_unique_documents, vocab_size,
                 num_topics, freqs=None, load_embeds=False, pretrained_embeddings=False,
                 save_graph_def=True, embedding_size=128, num_sampled=40,
                 learning_rate=1E-3, lmbda=150., alpha=None, power=.75, batch_size=500,
                 logdir="logdir", restore=False, W_in=None, factors_in=None,
                 additional_features_info=[], additional_features_names=[]):
        """Summary
        
        Args:
            num_unique_documents (int): Number of unique documents in your dataset
            vocab_size (int): Number of unique words/tokens in your dataset
            num_topics (int): The set number of topics to cluster your data into
            freqs (list, optional): Python list of length vocab_size with frequencies of each token
            load_embeds (bool, optional): If true, we will load embeddings from pretrained_embeddings variable
            pretrained_embeddings (np array, optional): Pretrained embeddings - shape should be (vocab_size, embedding_size)
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
            W_in (None, optional): Pretrained Doc Embedding weights (shape should be [num_unique_documents, embedding_size])
            factors_in (None, optional): Pretrained Topic Embedding (shape should be [num_topics, embedding_size])
            additional_features_info (list, optional): Pass this a list of the number of unique elements
                                                       relating the the feature passed
            additional_features_names (list, optional): A list of strings of the same length of additional_features_info
                                                       that names the corresponding additional features
        
        """
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sesh = tf.Session(config=self.config)
        self.moving_avgs = tf.train.ExponentialMovingAverage(0.9)

        self.num_unique_documents = num_unique_documents
        self.additional_features_info = additional_features_info
        self.num_additional_features = len(self.additional_features_info)
        self.additional_features_names = additional_features_names

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

        self.W_in = W_in
        self.factors_in = factors_in



        # This will be set to true if compute_normed_embeds is run so it doesnt get run more than once
        self.compute_normed = False

        if not restore:
            # Get formatted datetime from right now
            self.date = datetime.now().strftime(r"%y%m%d_%H%M")
            # Rename logdir according to current date
            self.logdir = "{}_{}".format(self.logdir, self.date)
            
            self.w_embed = W.Word_Embedding(self.embedding_size,
                                            self.vocab_size,
                                            self.num_sampled,
                                            load_embeds=self.load_embeds,
                                            pretrained_embeddings=self.pretrained_embeddings,
                                            freqs=self.freqs,
                                            power=self.power)

            # Doc and topic mixture
            self.mixture_doc = M.EmbedMixture(self.num_unique_documents,
                                          self.num_topics,
                                          self.embedding_size,
                                          name="doc")

            # Create list to hold additional feature embedding mixture objects
            self.additional_features_list = []
            # Loop through all additional features
            for feature in range(self.num_additional_features):
                # Append the embedding mixture object for each, using info and names provided
                self.additional_features_list.append(M.EmbedMixture(self.additional_features_info[feature],
                                                                    self.num_topics,
                                                                    self.embedding_size,
                                                                    name = self.additional_features_names[feature]))
            handles = self._build_graph()

            # Add graph variables to collection
            for handle in handles:
                tf.add_to_collection(Lda2vec.RESTORE_KEY, handle)

            self.x, self.y, self.docs, self.additional_features, self.step, self.switch_loss, self.pivot, self.doc, self.context, self.loss_word2vec, self.fraction, self.loss_lda, self.loss, self.loss_avgs_op, self.optimizer, self.doc_embedding, self.topic_embedding, self.word_embedding, self.nce_weights, self.nce_biases, self.merged, *kg = handles

            # If we recieved additional features, then we unpack them here
            if len(kg) > 0:
                self.additional_features_list = kg[:len(kg)//2]
                self.feature_lookup = kg[len(kg)//2:]     

        else:
            meta_graph = logdir + "/model.ckpt"
            tf.train.import_meta_graph(meta_graph + ".meta").restore(self.sesh, meta_graph)

            handles = self.sesh.graph.get_collection(Lda2vec.RESTORE_KEY)

            self.x, self.y, self.docs, self.additional_features, self.step, self.switch_loss, self.pivot, self.doc, self.context, self.loss_word2vec, self.fraction, self.loss_lda, self.loss, self.loss_avgs_op, self.optimizer, self.doc_embedding, self.topic_embedding, self.word_embedding, self.nce_weights, self.nce_biases, self.merged, *kg = handles

            # If we recieved additional features, then we unpack them here
            if len(kg) > 0:
                self.additional_features_list = kg[:len(kg)//2]
                self.feature_lookup = kg[len(kg)//2:]

    def prior(self):
        # First, compute the prior for the document
        doc_prior = DL.dirichlet_likelihood(self.mixture_doc.Doc_Embedding, alpha=self.alpha)
        # Next, set feature prior to False
        feature_prior_created = False
        # Loop through our additional features
        for i in range(self.num_additional_features):
            # Compute DL for each feature
            temp_feature_prior = DL.dirichlet_likelihood(self.additional_features_list[i].Doc_Embedding, alpha=self.alpha)
            
            # If there is more than one feature, the DL's will be added here
            if feature_prior_created:
                feature_prior += temp_feature_prior
            # If feature prior is false, set the current DL to be the feature_prior
            else:
                feature_prior_created = True
                feature_prior = temp_feature_prior
        if feature_prior_created:
            # Finally, we return the sum of the document prior as well as the summed feature prior
            return doc_prior + feature_prior
        else:
            return doc_prior

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

        # Context ID
        #contexts = tf.placeholder(tf.int32, shape=[None, ], name="context_ids")

        # Additional features are passed here with shape of [num_additional_features, batch_size]
        additional_features = tf.placeholder(tf.int32, shape=[self.num_additional_features, None])

        # Graph's current step
        step = tf.Variable(0, trainable=False, name="global_step")
        # Step to turn on document mixture loss
        switch_loss = tf.Variable(0, trainable=False)

        # Lookup pivot word idxs
        word_context = tf.nn.embedding_lookup(self.w_embed.Embedding, x, name="word_embed_lookup")

        # Lookup document embed
        doc_context = self.mixture_doc(doc_ids=docs)

        # Loop through additional features and do an embedding lookup
        feature_lookup = []
        for i in range(self.num_additional_features):
            feature_lookup.append(self.additional_features_list[i](doc_ids=additional_features[i]))

        # Lookup Context Embed
        #context_embed = self.mixture_context(doc_ids=contexts)

        # Context is sum of doc(mixture projected onto topics) & pivot embedding
        #dropout = self.mixture_doc.dropout

        # doc_context  = tf.nn.dropout(doc, dropout, name="doc_context")
        # word_context = tf.nn.dropout(pivot, dropout, name="word_context")
        # context = tf.add(word_context, doc_context, name="context_vector")
        contexts_to_add = feature_lookup
        contexts_to_add.append(word_context)
        contexts_to_add.append(doc_context)

        context = tf.add_n(contexts_to_add, name="context_vector")

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

        self.sesh.run(tf.global_variables_initializer(), options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

        merged = tf.summary.merge_all()

        to_return = [x, y, docs, additional_features, step, switch_loss, word_context, doc_context, context,
                    loss_word2vec, fraction, loss_lda, loss, loss_avgs_op, optimizer, self.mixture_doc.Doc_Embedding,
                    self.mixture_doc.topic_embedding, self.w_embed.Embedding, self.w_embed.nce_weights,
                    self.w_embed.nce_biases, merged]

        if self.num_additional_features:
            # Append the embeddings for each of the additional features
            for i in range(self.num_additional_features):
                to_return.append(self.additional_features_list[i].Doc_Embedding)
                to_return.append(self.additional_features_list[i].topic_embedding)

            # Append on additional features' pivot lookups
            to_return.extend(feature_lookup)

        return to_return


    def train(self, pivot_words, target_words, doc_ids, data_size,
              num_epochs,context_ids=False, switch_loss_epoch=0, save_every=5000, report_every=100):
        '''
        Args:
        pivot_words  - List of pivot word indexes (int array)
        target_words - List of target words indexes (int array)
        doc_ids      - List of Document Id's linked to pivot and target words (int array)
        context_ids  - List of Additonal contexts (ex. zip code)
        data_size    - Total amount of unique sentences (or data) before splitting into pivot/target words (int)
        num_epochs   - Integer noting how many epochs to train for. 
        '''
        # Fraction adjusts the loss term to be proportional to minibatch size
        temp_fraction = self.batch_size / data_size

        # Run the session to assign the fraction to the graph
        self.sesh.run(tf.assign(self.fraction, temp_fraction))
        # Computes how many batches we have in our given training data
        num_batches = data_size // self.batch_size
        self.num_batches = num_batches
        
        # Calculate the number of iterations per epoch
        iters_per_epoch = (int(data_size / self.batch_size) + np.ceil(data_size % self.batch_size))
        # Compute the step at which we will turn on lda loss (to train words before documents)
        switch_loss_step = iters_per_epoch * switch_loss_epoch
        # Assign switch loss variable
        self.sesh.run(tf.assign(self.switch_loss, switch_loss_step))


        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.logdir + "/", graph=self.sesh.graph)

        for e in range(num_epochs):
            print("\nEPOCH:", e + 1)
            for i in range(num_batches+1):
                if i < num_batches:
                    x_batch = pivot_words[i * self.batch_size:i * self.batch_size + self.batch_size]
                    y_batch = target_words[i * self.batch_size:i * self.batch_size + self.batch_size]
                    doc_batch = doc_ids[i * self.batch_size:i * self.batch_size + self.batch_size]
                    if type(context_ids) == bool:
                        pass
                    elif context_ids.shape[0] == 1:
                        feature_batch = context_ids[0][i * self.batch_size:i * self.batch_size + self.batch_size]
                        feature_batch = np.expand_dims(feature_batch, 0)
                    else:
                        feature_batch = context_ids[:, i * self.batch_size:i * self.batch_size + self.batch_size]
                else:
                    x_batch = pivot_words[i * self.batch_size:]
                    y_batch = target_words[i * self.batch_size:]
                    doc_batch = doc_ids[i * self.batch_size:]
                    if type(context_ids) == bool:
                        pass
                    elif context_ids.shape[0] ==1:
                        feature_batch = context_ids[0][i * self.batch_size:]
                        feature_batch = np.expand_dims(feature_batch, 0)
                    else:
                        feature_batch = context_ids[:, i*self.batch_size:]

                if type(context_ids) == bool:
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.docs: doc_batch}
                else:
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.docs: doc_batch, self.additional_features: feature_batch}    
                
                summary, _, l, lw2v, llda, step = self.sesh.run([self.merged, self.optimizer, self.loss, self.loss_word2vec, self.loss_lda, self.step],
                                                                feed_dict=feed_dict)

                if step > 0 and step % report_every == 0:
                    print("STEP", step, "LOSS", l, "w2v", lw2v, "lda", llda)
                
                if step > 0 and step % save_every == 0:
                    writer.add_summary(summary, step)
                    writer.flush()
                    writer.close()
                    save_path = saver.save(self.sesh, self.logdir + "/model.ckpt")
                    writer = tf.summary.FileWriter(self.logdir + "/", graph=self.sesh.graph)
        save_path =  saver.save(self.sesh, self.logdir + "/model.ckpt")

    def predict(self, pivot_words, doc_ids, temp_batch_size):
        context = self.sesh.run([self.context], feed_dict={self.x: pivot_words})


    def compute_normed_embeds(self):
        # Compute and save normalized embedding matrixes for computing similarity
        self.normed_embed_dict = {}
        self.normed_embed_dict["topic"] = tf.nn.l2_normalize(self.topic_embedding, axis=1) # axis=1
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

        # TODO - warning, edit this
        if self.compute_normed == False:
            self.compute_normed_embeds()

        self.batch_array = tf.nn.embedding_lookup(self.normed_embed_dict[in_type], self.idxs_in) #self.idxs_in

        self.cosine_similarity = tf.matmul(self.batch_array, tf.transpose(self.normed_embed_dict[vs_type],
                                                                [1, 0]))

        feed_dict = {self.idxs_in: idxs}

        sim, sim_idxs = self.sesh.run(tf.nn.top_k(self.cosine_similarity, k=k),
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

    def save_weights_to_file(self, word_embed_path="word_weights", doc_embed_path="doc_weights", topic_embed_path="topic_weights"):
        # TODO - Currently not supporting the extra context variables

        # Get numpy array representations of each and save to file
        word_embeds = self.sesh.run(self.word_embedding)
        np.save(word_embed_path, word_embeds)
        doc_embeds = self.sesh.run(self.doc_embedding)
        np.save(doc_embed_path, doc_embeds)
        topic_embeds = self.sesh.run(self.topic_embedding)
        np.save(topic_embed_path, topic_embeds)
