from __future__ import print_function

from abc import abstractmethod

import os
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input
# AttributeError: 'tuple' object has no attribute 'layer' #478
# https://github.com/tensorflow/probability/issues/478
from tensorflow.keras.layers import Embedding, Conv1D, Lambda, LSTM, Dense, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras.optimizers import Adam
from attention_unif import AttentionLayer, AttentionLayerWithBatchNormalization
from average_unif import AverageLayer

import numpy as np


class LanguageModel:
    def __init__(self, config):
        #self.question = Input(shape=(config.question_len(),), dtype='int32', name='question_base')
        #self.answer_good = Input(shape=(config.answer_len(),), dtype='int32', name='answer_good_base')
        #self.answer_bad = Input(shape=(config.answer_len(),), dtype='int32', name='answer_bad_base')
        self.question_len = config.question_len()
        self.answer_len = config.answer_len()

        self.config = config
        self.params = config.similarity_params()
        self.kernel_size = self.config.kernel_size()

        # initialize a bunch of variables that will be set later
        self._models = None
        self._similarities = None
        self._answer = None
        self._qa_model = None

        training_model, prediction_model = self.create_model()
        self.training_model = training_model
        self.prediction_model = prediction_model

    @abstractmethod
    def build(self):
        return

    def hinge_loss(self, x):
        margin = self.config.margin()
        return K.relu(margin - x[0] + x[1])

    def custom_loss(self, y_true, y_pred):
        return y_pred

    def create_model(self):
        question = Input(shape=(self.question_len,), dtype='int32', name='question_base')
        answer_good = Input(shape=(self.answer_len,), dtype='int32', name='answer_good')
        answer_bad = Input(shape=(self.answer_len,), dtype='int32', name='answer_bad')

        qa_model = self.build()
        good_similarity = qa_model([question, answer_good])
        bad_similarity = qa_model([question, answer_bad])

        loss = Lambda(self.hinge_loss,
                      output_shape=lambda x: x[0], name='hinge_loss')([good_similarity, bad_similarity])

        training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss,
                        name='training_model')

        prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity,
                name='prediction_model')

        return training_model, prediction_model

    def compile(self, optimizer, **kwargs):
        self.prediction_model.compile(loss=self.custom_loss, optimizer=Adam(), **kwargs)

        self.training_model.compile(loss=self.custom_loss, optimizer=Adam(), **kwargs)

    def training_summary(self):
        self.training_model.summary()

    def prediction_summary(self):
        self.prediction_model.summary()

    def save_training_plot_model(self, filename):
        # visualize model
        plot_model(self.training_model, to_file=filename, show_shapes=True, expand_nested=True)

    def save_prediction_plot_model(self, filename):
        # visualize model
        plot_model(self.prediction_model, to_file=filename, show_shapes=True, expand_nested=True)

    def fit(self, x, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=(x[0].shape[0],))  # doesn't get used
        return self.training_model.fit(x, y, **kwargs)

    def predict(self, x):
        assert self.prediction_model is not None and isinstance(self.prediction_model, Model)
        return self.prediction_model.predict_on_batch(x)

    def save_json(self, filename, **kwargs):
        # Save JSON config to disk
        json_config = self.prediction_model.to_json()
        with open(filename, 'w') as json_file:
            json_file.write(json_config)


    def save_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model before saving weights'
        self.prediction_model.save_weights(file_name, **kwargs)

    def load_json(self, filename, **kwargs):
        with open(filename) as json_file:
            json_config = json_file.read()

        self.prediction_model = model_from_json(json_config, custom_objects={'AttentionLayer': AttentionLayer,
                                   'AttentionLayerWithBatchNormalization': AttentionLayerWithBatchNormalization,
                                   'AverageLayer': AverageLayer})

    def load_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model loading weights'
        self.prediction_model.load_weights(file_name, **kwargs)


class EmbeddingModel(LanguageModel):
    def build(self):
        question = Input(shape=(self.question_len,), dtype='int32', name='question')
        answer = Input(shape=(self.answer_len,), dtype='int32', name='answer')

        # add embedding layers
        question_weights = np.load(self.config.initial_question_weights())
        q_embedding = Embedding(input_dim=question_weights.shape[0],
                                output_dim=question_weights.shape[1],
                                weights=[question_weights],
                                name='question_embedding')
        question_embedding = q_embedding(question)

        answer_weights = np.load(self.config.initial_answer_weights())
        a_embedding = Embedding(input_dim=answer_weights.shape[0],
                                output_dim=answer_weights.shape[1],
                                weights=[answer_weights],
                                name='answer_embedding')
        answer_embedding = a_embedding(answer)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
                         name='max')
        maxpool.supports_masking = True
        question_pool = maxpool(question_embedding)
        answer_pool = maxpool(answer_embedding)
        
        cos_similarity = Lambda(lambda x: cosine_similarity(x[0], x[1], axes=1)
                                       , output_shape=lambda _: (None, 1), name='similarity')([question_pool,
                                                                                               answer_pool])

        return Model(inputs=[question, answer], outputs=cos_similarity,
                                   name='qa_model')


class SharedConvolutionModel(LanguageModel):
    def build(self):
        assert self.config.question_len() == self.config.answer_len()

        question = Input(shape=(self.question_len,), dtype='int32', name='question')
        answer = Input(shape=(self.answer_len,), dtype='int32', name='answer')

        # add embedding layers
        question_weights = np.load(self.config.initial_question_weights())
        q_embedding = Embedding(input_dim=question_weights.shape[0],
                                output_dim=question_weights.shape[1],
                                weights=[question_weights],
                                name='question_embedding')
        question_embedding = q_embedding(question)

        answer_weights = np.load(self.config.initial_answer_weights())
        a_embedding = Embedding(input_dim=answer_weights.shape[0],
                                output_dim=answer_weights.shape[1],
                                weights=[answer_weights],
                                name='answer_embedding')
        answer_embedding = a_embedding(answer)

        # cnn
        filters = self.config.filters()
        kernel_size = self.kernel_size

        question_cnn = None
        answer_cnn = None

        if len(kernel_size) > 1:
            cnns = [Conv1D(kernel_size=k,
                           filters=filters,
                           activation='relu',
                           padding='same',
                           name=f'shared_conv1d_{k}') for k in kernel_size]
            # question_cnn = merge([cnn(question_embedding) for cnn in cnns], mode='concat')
            question_cnn = concatenate([cnn(question_embedding) for cnn in cnns])
            # answer_cnn = merge([cnn(answer_embedding) for cnn in cnns], mode='concat')

            answer_cnn = concatenate([cnn(answer_embedding) for cnn in cnns])
        else:
            k = kernel_size[0]
            cnn = Conv1D(kernel_size=k,
                           filters=filters,
                           activation='relu',
                           padding='same',
                           name=f'shared_conv1d_{k}')
            # question_cnn = merge([cnn(question_embedding) for cnn in cnns], mode='concat')
            question_cnn = cnn(question_embedding)
            # answer_cnn = merge([cnn(answer_embedding) for cnn in cnns], mode='concat')

            answer_cnn = cnn(answer_embedding)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
                         name='max')
        maxpool.supports_masking = True
        # enc = Dense(100, activation='tanh')
        # question_pool = enc(maxpool(question_cnn))
        # answer_pool = enc(maxpool(answer_cnn))
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        cos_similarity = Lambda(lambda x: cosine_similarity(x[0], x[1], axes=1)
                                       , output_shape=lambda _: (None, 1), name='similarity')([question_pool,
                                                                                               answer_pool])
        

        return Model(inputs=[question, answer], outputs=cos_similarity,
                                   name='qa_model')


class SharedConvolutionModelWithBatchNormalization(LanguageModel):
    def build(self):
        assert self.config.question_len() == self.config.answer_len()

        question = Input(shape=(self.question_len,), dtype='int32', name='question')
        answer = Input(shape=(self.answer_len,), dtype='int32', name='answer')

        # add embedding layers
        question_weights = np.load(self.config.initial_question_weights())
        q_embedding = Embedding(input_dim=question_weights.shape[0],
                                output_dim=question_weights.shape[1],
                                weights=[question_weights],
                                name='question_embedding')
        question_embedding = q_embedding(question)

        answer_weights = np.load(self.config.initial_answer_weights())
        a_embedding = Embedding(input_dim=answer_weights.shape[0],
                                output_dim=answer_weights.shape[1],
                                weights=[answer_weights],
                                name='answer_embedding')
        answer_embedding = a_embedding(answer)

        # cnn
        filters = self.config.filters()
        kernel_size = self.kernel_size
        question_cnn = None
        answer_cnn = None

        if len(kernel_size) > 1:
            cnns = [Conv1D(kernel_size=k,
                           filters=filters,
                           padding='same',
                           name=f'shared_conv1d_with_bn_{k}') for k in kernel_size]
            # question_cnn = merge([cnn(question_embedding) for cnn in cnns], mode='concat')

            question_outputs_cnn = [cnn(question_embedding) for cnn in cnns]
            bn_question_outputs_cnn = [BatchNormalization()(output) for output in question_outputs_cnn]
            activation_question_outputs_cnn = [Activation('relu')(output) for output in bn_question_outputs_cnn]

            question_cnn = concatenate([activation_output for activation_output in activation_question_outputs_cnn])
            # answer_cnn = merge([cnn(answer_embedding) for cnn in cnns], mode='concat')

            answer_outputs_cnn = [cnn(answer_embedding) for cnn in cnns]
            bn_answer_outputs_cnn = [BatchNormalization()(output) for output in answer_outputs_cnn]
            activation_answer_outputs_cnn = [Activation('relu')(output) for output in bn_answer_outputs_cnn]

            answer_cnn = concatenate([activation_output for activation_output in activation_answer_outputs_cnn])
        else:
            k = kernel_size[0]
            cnn = Conv1D(kernel_size=k,
                           filters=filters,
                           padding='same',
                           name=f'shared_conv1d_with_bn_{k}')
            # question_cnn = merge([cnn(question_embedding) for cnn in cnns], mode='concat')

            question_output_cnn = cnn(question_embedding)
            bn_question_output_cnn = BatchNormalization()(question_output_cnn)
            activation_question_output_cnn = Activation('relu')(bn_question_output_cnn)

            question_cnn = activation_question_output_cnn
            # answer_cnn = merge([cnn(answer_embedding) for cnn in cnns], mode='concat')

            answer_output_cnn = cnn(answer_embedding)
            bn_answer_output_cnn = BatchNormalization()(answer_output_cnn)
            activation_answer_output_cnn = Activation('relu')(bn_answer_output_cnn)

            answer_cnn = activation_answer_output_cnn

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
                         name='max')
        maxpool.supports_masking = True
        # enc = Dense(100, activation='tanh')
        # question_pool = enc(maxpool(question_cnn))
        # answer_pool = enc(maxpool(answer_cnn))
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        cos_similarity = Lambda(lambda x: cosine_similarity(x[0], x[1], axes=1)
                                       , output_shape=lambda _: (None, 1), name='similarity')([question_pool,
                                                                                               answer_pool])

        

        return Model(inputs=[question, answer], outputs=cos_similarity,
                                   name='qa_model')


class ConvolutionModel(LanguageModel):
    def build(self):
        assert self.config.question_len() == self.config.answer_len()

        question = Input(shape=(self.question_len,), dtype='int32', name='question')
        answer = Input(shape=(self.answer_len,), dtype='int32', name='answer')

        # add embedding layers
        question_weights = np.load(self.config.initial_question_weights())
        q_embedding = Embedding(input_dim=question_weights.shape[0],
                                output_dim=question_weights.shape[1],
                                weights=[question_weights],
                                name='question_embedding')
        question_embedding = q_embedding(question)

        answer_weights = np.load(self.config.initial_answer_weights())
        a_embedding = Embedding(input_dim=answer_weights.shape[0],
                                output_dim=answer_weights.shape[1],
                                weights=[answer_weights],
                                name='answer_embedding')
        answer_embedding = a_embedding(answer)

        # cnn
        filters = self.config.filters()
        kernel_size = self.kernel_size

        question_cnn = None
        answer_cnn = None

        if len(kernel_size) > 1:
            q_cnns = [Conv1D(kernel_size=k,
                             filters=filters,
                             activation='relu',
                             padding='same',
                             name=f'question_conv1d_{k}') for k in kernel_size]
            # question_cnn = merge([cnn(question_embedding) for cnn in cnns], mode='concat')
            question_cnn = concatenate([cnn(question_embedding) for cnn in q_cnns])
            # answer_cnn = merge([cnn(answer_embedding) for cnn in cnns], mode='concat')
            a_cnns = [Conv1D(kernel_size=k,
                             filters=filters,
                             activation='relu',
                             padding='same',
                             name=f'answer_conv1d_{k}') for k in kernel_size]
            answer_cnn = concatenate([cnn(answer_embedding) for cnn in a_cnns])
        else:
            k = kernel_size[0]
            q_cnn = Conv1D(kernel_size=k,
                             filters=filters,
                             activation='relu',
                             padding='same',
                             name=f'question_conv1d_{k}')
            # question_cnn = merge([cnn(question_embedding) for cnn in cnns], mode='concat')
            question_cnn = q_cnn(question_embedding)
            # answer_cnn = merge([cnn(answer_embedding) for cnn in cnns], mode='concat')
            a_cnn = Conv1D(kernel_size=k,
                             filters=filters,
                             activation='relu',
                             padding='same',
                             name=f'answer_conv1d_{k}')
            answer_cnn = a_cnn(answer_embedding)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
                         name='max')
        maxpool.supports_masking = True
        # enc = Dense(100, activation='tanh')
        # question_pool = enc(maxpool(question_cnn))
        # answer_pool = enc(maxpool(answer_cnn))
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        cos_similarity = Lambda(lambda x: cosine_similarity(x[0], x[1], axes=1)
                                       , output_shape=lambda _: (None, 1), name='similarity')([question_pool,
                                                                                               answer_pool])

        return Model(inputs=[question, answer], outputs=cos_similarity,
                                   name='qa_model')

class ConvolutionModelWithBatchNormalization(LanguageModel):
    def build(self):
        assert self.config.question_len() == self.config.answer_len()

        question = Input(shape=(self.question_len,), dtype='int32', name='question')
        answer = Input(shape=(self.answer_len,), dtype='int32', name='answer')

        # add embedding layers
        question_weights = np.load(self.config.initial_question_weights())
        q_embedding = Embedding(input_dim=question_weights.shape[0],
                                output_dim=question_weights.shape[1],
                                weights=[question_weights],
                                name='question_embedding')
        question_embedding = q_embedding(question)

        answer_weights = np.load(self.config.initial_answer_weights())
        a_embedding = Embedding(input_dim=answer_weights.shape[0],
                                output_dim=answer_weights.shape[1],
                                weights=[answer_weights],
                                name='answer_embedding')
        answer_embedding = a_embedding(answer)

        # cnn
        filters = self.config.filters()
        kernel_size = self.kernel_size

        question_cnn = None
        answer_cnn = None


        if len(kernel_size) > 1:
            q_cnns = [Conv1D(kernel_size=k,
                             filters=filters,
                             padding='same',
                             name=f'question_conv1d_with_bn_{k}') for k in kernel_size]
            # question_cnn = merge([cnn(question_embedding) for cnn in cnns], mode='concat')

            question_outputs_cnn = [cnn(question_embedding) for cnn in q_cnns]
            bn_question_outputs_cnn = [BatchNormalization()(output) for output in question_outputs_cnn]
            activation_question_outputs_cnn = [Activation('relu')(output) for output in bn_question_outputs_cnn]

            question_cnn = concatenate([activation_output for activation_output in activation_question_outputs_cnn])

            # answer_cnn = merge([cnn(answer_embedding) for cnn in cnns], mode='concat')
            a_cnns = [Conv1D(kernel_size=k,
                             filters=filters,
                             padding='same',
                             name=f'answer_conv1d_with_bn_{k}') for k in kernel_size]

            answer_outputs_cnn = [cnn(answer_embedding) for cnn in a_cnns]
            bn_answer_outputs_cnn = [BatchNormalization()(output) for output in answer_outputs_cnn]
            activation_answer_outputs_cnn = [Activation('relu')(output) for output in bn_answer_outputs_cnn]

            answer_cnn = concatenate([activation_output for activation_output in activation_answer_outputs_cnn])
        else:
            k = kernel_size[0]
            q_cnn = Conv1D(kernel_size=k,
                             filters=filters,
                             padding='same',
                             name=f'question_conv1d_with_bn_{k}')
            # question_cnn = merge([cnn(question_embedding) for cnn in cnns], mode='concat')

            question_output_cnn = q_cnn(question_embedding)
            bn_question_output_cnn = BatchNormalization()(question_output_cnn)
            activation_question_output_cnn = Activation('relu')(bn_question_output_cnn)

            question_cnn = activation_question_output_cnn

            # answer_cnn = merge([cnn(answer_embedding) for cnn in cnns], mode='concat')
            a_cnn = Conv1D(kernel_size=k,
                             filters=filters,
                             padding='same',
                             name=f'answer_conv1d_with_bn_{k}')

            answer_output_cnn = a_cnn(answer_embedding)
            bn_answer_output_cnn = BatchNormalization()(answer_output_cnn)
            activation_answer_output_cnn = Activation('relu')(bn_answer_output_cnn)

            answer_cnn = activation_answer_output_cnn

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
                         name='max')
        maxpool.supports_masking = True
        # enc = Dense(100, activation='tanh')
        # question_pool = enc(maxpool(question_cnn))
        # answer_pool = enc(maxpool(answer_cnn))
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        

        cos_similarity = Lambda(lambda x: cosine_similarity(x[0], x[1], axes=1)
                                       , output_shape=lambda _: (None, 1), name='similarity')([question_pool,
                                                                                               answer_pool])

        return Model(inputs=[question, answer], outputs=cos_similarity,
                                   name='qa_model')



class ConvolutionalLSTM(LanguageModel):
    def build(self):
        question = Input(shape=(self.question_len,), dtype='int32', name='question')
        answer = Input(shape=(self.answer_len,), dtype='int32', name='answer')

        # add embedding layers
        question_weights = np.load(self.config.initial_question_weights())
        q_embedding = Embedding(input_dim=question_weights.shape[0],
                                output_dim=question_weights.shape[1],
                                weights=[question_weights])
        question_embedding = q_embedding(question)

        answer_weights = np.load(self.config.initial_answer_weights())
        a_embedding = Embedding(input_dim=answer_weights.shape[0],
                                output_dim=answer_weights.shape[1],
                                weights=[answer_weights])
        answer_embedding = a_embedding(answer)

        f_rnn = LSTM(141, return_sequences=True, implementation=1)
        b_rnn = LSTM(141, return_sequences=True, implementation=1, go_backwards=True)

        qf_rnn = f_rnn(question_embedding)
        qb_rnn = b_rnn(question_embedding)
        # question_pool = merge([qf_rnn, qb_rnn], mode='concat', concat_axis=-1)
        question_pool = concatenate([qf_rnn, qb_rnn], axis=-1)

        af_rnn = f_rnn(answer_embedding)
        ab_rnn = b_rnn(answer_embedding)
        # answer_pool = merge([af_rnn, ab_rnn], mode='concat', concat_axis=-1)
        answer_pool = concatenate([af_rnn, ab_rnn], axis=-1)

        # cnn
        filters = self.config.filters()

        cnns = [Conv1D(kernel_size=kernel_size,
                       filters=filters,
                       activation='tanh',
                       padding='same') for kernel_size in [1, 2, 3, 5]]
        # question_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat')
        question_cnn = concatenate([cnn(question_pool) for cnn in cnns])
        # answer_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')
        answer_cnn = concatenate([cnn(answer_pool) for cnn in cnns])

        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        cos_similarity = Lambda(lambda x: cosine_similarity(x[0], x[1], axes=1)
                                       , output_shape=lambda _: (None, 1), name='similarity')([question_pool,
                                                                                               answer_pool])

        return Model(inputs=[question, answer], outputs=cos_similarity,
                                   name='qa_model')


class UnifModel(LanguageModel):
    def build(self):
        print(tf.__version__)
        print(tf.keras.__version__)
        question = Input(shape=(self.question_len,), dtype='int32', name='question')
        answer = Input(shape=(self.answer_len,), dtype='int32', name='answer')

        # add embedding layers
        question_weights = np.load(self.config.initial_question_weights())
        q_embedding = Embedding(input_dim=question_weights.shape[0],
                                output_dim=question_weights.shape[1],
                                weights=[question_weights],
                                name='question_embedding')
        question_embedding = q_embedding(question)

        f_average_layer = AverageLayer(name='average')
        e_q = f_average_layer([question_embedding])

        answer_weights = np.load(self.config.initial_answer_weights())
        a_embedding = Embedding(input_dim=answer_weights.shape[0],
                                output_dim=answer_weights.shape[1],
                                weights=[answer_weights],
                                name='answer_embedding')
        answer_embedding = a_embedding(answer)

        f_attention_layer = AttentionLayer(name='attention')
        e_c = f_attention_layer([answer_embedding])
        
        cos_similarity = Lambda(lambda x: cosine_similarity(x[0], x[1], axes=1)
                                       , output_shape=lambda _: (None, 1), name='similarity')([e_q,
                                                                                               e_c])

        return Model(inputs=[question, answer], outputs=cos_similarity,
                                   name='qa_model')

class UnifModelWithBatchNormalization(LanguageModel):
    def build(self):
        print(tf.__version__)
        print(tf.keras.__version__)
        question = Input(shape=(self.question_len,), dtype='int32', name='question')
        answer = Input(shape=(self.answer_len,), dtype='int32', name='answer')

        # add embedding layers
        question_weights = np.load(self.config.initial_question_weights())
        q_embedding = Embedding(input_dim=question_weights.shape[0],
                                output_dim=question_weights.shape[1],
                                weights=[question_weights],
                                name='question_embedding')
        question_embedding = q_embedding(question)

        f_average_layer = AverageLayer(name='average')
        e_q = f_average_layer([question_embedding])

        answer_weights = np.load(self.config.initial_answer_weights())
        a_embedding = Embedding(input_dim=answer_weights.shape[0],
                                output_dim=answer_weights.shape[1],
                                weights=[answer_weights],
                                name='answer_embedding')
        answer_embedding = a_embedding(answer)

        f_attention_layer = AttentionLayerWithBatchNormalization(name='attention-with-bn')
        e_c = f_attention_layer([answer_embedding])

        

        cos_similarity = Lambda(lambda x: cosine_similarity(x[0], x[1], axes=1)
                                       , output_shape=lambda _: (None, 1), name='similarity')([e_q,
                                                                                               e_c])

        return Model(inputs=[question, answer], outputs=cos_similarity,
                     name='qa_model')
