import os

import sys
import random
from time import strftime, gmtime, time
from report_result import ReportResult
from configuration import Conf

import argparse

import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K


from scipy.stats import rankdata
import logging
import numpy as np
import tensorflow as tf
import pandas as pd


def clear_session():
    K.clear_session()


class Evaluator:
    def __init__(self, conf_json, model, optimizer=None, name=None):
        try:
            data_path = os.environ['STACK_OVER_FLOW_QA']
        except KeyError:
            logger.warning("STACK_OVER_FLOW_QA is not set. Set it to your clone of https://github.com/mrezende/stack_over_flow_python")
            sys.exit(1)
        self.conf = Conf(conf_json)
        self.model = model(self.conf)
        if name is None:
            self.name = self.conf.name() + '_' + model.__name__
            logger.info(f'Initializing Evaluator ...')
            logger.info(f'Name: {self.name}')
        else:
            self.name = name

        self.path = data_path
        self.params = self.conf.training_params()
        self.optimizer = self.params['optimizer'] if optimizer is None else optimizer

        self.answers = self.load('answers.json') # self.load('generated')
        self.answers_index = self.load('answers_index.json')
        self.training_data = self.load('training.json')
        self.dev_data = self.load('dev.json')
        self.eval_data = self.load('eval.json')
        self._vocab = None
        self._reverse_vocab = None
        self._eval_sets = None
        self.top1_ls = []
        self.mrr_ls = []

    ##### Resources #####

    def save_conf(self):
        self.conf.save_conf()

    def load(self, name):
        return json.load(open(os.path.join(self.path, name), 'r'))

    def vocab(self):
        if self._vocab is None:
            reverse_vocab = self.reverse_vocab()
            self._vocab = dict((v, k.lower()) for k, v in reverse_vocab.items())
        return self._vocab

    def reverse_vocab(self):
        if self._reverse_vocab is None:
            samples = self.load('samples_for_tokenizer.json')

            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(samples)

            self._reverse_vocab = tokenizer.word_index
        return self._reverse_vocab

    def describe(self):
        logger.info(f'Training Summary: {self.name}')
        self.model.training_summary()

        logger.info(f'Prediction Summary: {self.name}')
        self.model.prediction_summary()

        path = 'models/summary/'
        if not os.path.exists(path):
            os.makedirs(path)

        training_file = f'plot_training_{self.name}.png'
        training_path = path + training_file
        self.model.save_training_plot_model(training_path)

        prediction_file = f'plot_prediction_{self.name}.png'
        prediction_path = path + prediction_file
        self.model.save_prediction_plot_model(prediction_path)

    def compile(self):
        self.model.compile(self.optimizer)

    ##### Loading / saving #####

    def save_json(self, name = None):
        path = 'models/weights/json/'
        if not os.path.exists(path):
            os.makedirs(path)
        suffix = self.name if name is None else name
        logger.info(f'Saving config json: {path}config_{suffix}.json')
        logger.info(f'Saving config json: {path}config_{suffix}_best.json')
        self.model.save_json(f'{path}config_{suffix}.json', overwrite=True)
        self.model.save_json(f'{path}config_{suffix}_best.json', overwrite=True)

    def save_epoch(self, name = None):
        path = 'models/weights/'
        if not os.path.exists(path):
            os.makedirs(path)
        suffix = self.name if name is None else name
        logger.info(f'Saving weights: {path}weights_epoch_{suffix}.h5')
        self.model.save_weights(f'{path}weights_epoch_{suffix}.h5', overwrite=True)

    def load_json(self, name = None):
        path = 'models/weights/json/'
        suffix = self.name if name is None else name

        assert os.path.exists(f'{path}config_{suffix}.json'), f'Weights at epoch {suffix} not found'

        logger.info(f'Loading config json: {path}config_{suffix}.json')
        self.model.load_json(f'{path}config_{suffix}.json')

    def load_epoch(self, name):
        path = 'models/weights/'
        suffix = name
        assert os.path.exists(f'{path}weights_epoch_{suffix}.h5'), f'Weights at epoch {suffix} not found'
        logger.info(f'Loading weights: {path}weights_epoch_{suffix}.h5')
        self.model.load_weights(f'{path}weights_epoch_{suffix}.h5')

    def load_best_json(self, name = None):
        path = 'models/weights/json/'
        suffix = self.name if name is None else name
        suffix += '_best'
        assert os.path.exists(f'{path}config_{suffix}.json'), f'Weights at epoch {suffix} not found'

        logger.info(f'Loading best val loss config json: {path}config_{suffix}.json')
        self.model.load_json(f'{path}config_{suffix}.json')

    def load_best_epoch(self, name):
        path = 'models/weights/'
        suffix = name + '_best'
        assert os.path.exists(f'{path}weights_epoch_{suffix}.h5'), f'Weights at epoch {suffix} not found'
        logger.info(f'Loading best val loss weights: {path}weights_epoch_{suffix}.h5')
        self.model.load_weights(f'{path}weights_epoch_{suffix}.h5')

    ##### Converting / reverting #####

    def convert(self, words):
        rvocab = self.reverse_vocab()
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [rvocab.get(w, 0) for w in words]

    def revert(self, indices):
        vocab = self.vocab()
        return [vocab.get(i, 'X') for i in indices]

    ##### Padding #####

    def padq(self, data):
        return self.pad(data, self.conf.question_len())

    def pada(self, data):
        return self.pad(data, self.conf.answer_len())

    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Training #####

    def get_time(self):
        return strftime('%Y-%m-%d %H:%M:%S', gmtime())

    def train_and_evaluate(self, mode='train'):
        val_losses = []

        self.describe()

        if mode == 'train':
            self.compile()
            self.save_json()
            val_loss = self.train(self.training_data)
            val_losses.append(val_loss)
            logger.info(f'Val loss: {val_loss}')

        elif mode == 'evaluate':
            self.load_json()
            results = self.evaluate(verbose=True)

            # results:

            logger.info(f'final_results: {results}')

            df = pd.DataFrame(results)
            top1_desc = df.describe()['top1']
            mrr_desc = df.describe()['mrr']

            # save histogram plot
            report = ReportResult({'positions': np.append([], results['positions'])}, index=[i for i in range(1, len(np.append([], results['positions'])) + 1)], plot_name = f'histogram_{self.name}')
            report.generate_histogram()
            report.save_plot()

            logger.info(f'Top1 Description: {top1_desc}')
            logger.info(f'MRR Description: {mrr_desc}')
        elif mode == 'evaluate-best':
            self.load_best_json()
            results = self.evaluate_best(verbose=True)

            # results:

            logger.info(f'model_best_val_loss final_results: {results}')

            df = pd.DataFrame(results)
            top1_desc = df.describe()['top1']
            mrr_desc = df.describe()['mrr']

            # save histogram plot
            report = ReportResult({'positions': np.append([], results['positions'])},
                                  index=[i for i in range(1, len(np.append([], results['positions'])) + 1)],
                                  plot_name=f'histogram_best_{self.name}')
            report.generate_histogram()
            report.save_plot()

            logger.info(f'Top1 Description: {top1_desc}')
            logger.info(f'MRR Description: {mrr_desc}')
        elif mode == 'evaluate-code-by-length':
            self.load_json()


            filenames = ['eval_15.json', 'eval_25.json', 'eval_35.json', 'eval_50.json', 
            'eval_75.json', 'eval_100.json', 'eval_larger_100.json']


            for filename in filenames:
                X = self.load(filename)

                results = self.evaluate(X=X, verbose=True)

                # results:

                logger.info(f'----------- eval: {filename} ------------')

                logger.info(f'{filename} final_results: {results}')

                df = pd.DataFrame(results)
                top1_desc = df.describe()['top1']
                mrr_desc = df.describe()['mrr']

                # save histogram plot
                report = ReportResult({'positions': np.append([], results['positions'])},
                                      index=[i for i in range(1, len(np.append([], results['positions'])) + 1)],
                                      plot_name=f'histogram_best_{self.name}')
                report.generate_histogram()
                report.save_plot()

                logger.info(f'Top1 Description: {top1_desc}')
                logger.info(f'MRR Description: {mrr_desc}')


        elif mode == 'save_config':
            self.save_json()

    def evaluate(self, X = None, name = None, verbose=False):
        name = self.name if name is None else name
        self.load_epoch(name)
        data = self.eval_data if X is None else X
        results = {'top1': [], 'mrr': [], 'positions': []}
        logger.info('Evaluating...')
        for i in range(0, 20):
            top1, mrr, positions = self.get_score(data, verbose=verbose)
            results['top1'].append(top1)
            results['mrr'].append(mrr)
            results['positions'].append(positions)
            logger.info(f'Iteration: {i}: Top-1 Precision {top1}, MRR {mrr}, Positions: {positions}')
        return results

    def evaluate_best(self, X = None, name = None, verbose=False):
        name = self.name if name is None else name
        self.load_best_epoch(name)
        data = self.eval_data if X is None else X
        results = {'top1': [], 'mrr': [], 'positions': []}
        logger.info('Evaluating...')
        for i in range(0, 20):
            top1, mrr, positions = self.get_score(data, verbose=verbose)
            results['top1'].append(top1)
            results['mrr'].append(mrr)
            results['positions'].append(positions)
            logger.info(f'Iteration: {i}: Top-1 Precision {top1}, MRR {mrr}, Positions: {positions}')
        return results

    def train(self, X):
        batch_size = self.params['batch_size']
        validation_split = self.params['validation_split']
        nb_epoch = self.params['nb_epoch']

        # top_50 = self.load('top_50')

        questions = list()
        good_answers = list()

        for j, q in enumerate(X):
            questions += [q['question']] * len(q['good_answers'])
            good_answers += q['good_answers']

        logger.info('Began training at %s on %d samples' % (self.get_time(), len(questions)))

        questions = self.padq(questions)
        good_answers = self.pada(good_answers)

        best_top1_mrr = {'top1': 0, 'mrr': 0}
        hist_losses = {'val_loss': [], 'loss': []}
        hist_results = {'results': []}
        best_val_loss = 10 # positive number, as long val_loss is decimal
        val_loss_without_improve = 0
        
        patience = nb_epoch / 20 # 5% of number of epochs
        

        for i in range(1, nb_epoch + 1):

            bad_answers = self.pada(random.sample(self.answers, len(good_answers)))

            logger.info(f'Fitting epoch {i}')
            hist = self.model.fit([questions, good_answers, bad_answers], epochs=1, 
                                  batch_size=batch_size,
                                  validation_split=validation_split, verbose=1)
            
            val_loss = hist.history['val_loss'][0]
            loss = hist.history['loss'][0]
            hist_losses['val_loss'].append(val_loss)
            hist_losses['loss'].append(loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_loss_without_improve = 0
                logger.info(f'Saving best val_loss weights: Epoch {i} best val_loss: {best_val_loss}')
                temp_filename = f'{self.name}_best'
                self.save_epoch(temp_filename)
            else:
                val_loss_without_improve += 1

            # temporary weights from last training
            temp_filename = f'{self.name}_aux'
            self.save_epoch(temp_filename)

            # check MRR
            results = self.evaluate(self.dev_data, temp_filename)
            df = pd.DataFrame(results)
            mrr = df.mean()['mrr']
            top1 = df.mean()['top1']
            hist_results['results'].append(results)

            if mrr > best_top1_mrr['mrr']:
                best_top1_mrr['top1'] = top1
                best_top1_mrr['mrr'] = mrr
                logger.info(f'Epoch {i} Loss = {loss}, Validation Loss = {val_loss} ' +
                            f'(Best average: TOP1 = {top1}, MRR = {mrr})')

                # saving weights
                self.save_epoch()

            # early stopping like Staqc (Yao et al.)
            # see source code: https://github.com/mrezende/StackOverflow-Question-Code-Dataset/blob/master/BiV_HNN/run.py
            if val_loss_without_improve > patience or loss < 1e-4:
                break

        # save plot val_loss, loss
        report = ReportResult(hist_losses, [i for i in range(1, len(hist_losses['loss']) + 1)], self.name)
        plot = report.generate_line_report()
        report.save_plot()

        # top1, mrr, positions:
        logger.info(f'hist_results: {hist_results}')


        logger.info(f'saving loss, val_loss plot')

        # save conf
        self.save_conf()

        clear_session()
        return val_loss

    def get_score(self, X, verbose=False, shuffle=False):
        c_1, c_2 = 0, 0

        logger.info(f'len X: {len(X)}')
        positions = []
        for i, d in enumerate(X):
            bad_answers = random.sample(self.answers, 49)
            answers = d['good_answers'] + bad_answers
            answers = self.pada(answers)
            question = self.padq([d['question']] * len(answers))

            sims = self.model.predict([question, answers])

            n_good = len(d['good_answers'])
            max_r = np.argmax(sims)
            max_n = np.argmax(sims[:n_good])

            r = rankdata(sims, method='max')
            sims_index_sorted = np.argsort(sims)[::-1][:len(sims)]

            if verbose:
                min_r = np.argmin(sims)
                amin_r = answers[min_r]
                amax_r = answers[max_r]
                amax_n = answers[max_n]

                logger.info(' ----- begin question ----- ')
                logger.info(' '.join(self.revert(d['question'])))
                logger.info('Predicted: ({}) '.format(sims[max_r]) + ' '.join(self.revert(amax_r)))
                logger.info('Expected: ({}) Rank = {} '.format(sims[max_n], r[max_n]) + ' '.join(self.revert(amax_n)))
                logger.info('Worst: ({})'.format(sims[min_r]) + ' '.join(self.revert(amin_r)))
                logger.info(' ----- end question ----- ')

                logger.info('------ begin correct answer ----------')

                for good_answer in d['good_answers']:
                    logger.info(' '.join(self.revert(good_answer)))

                logger.info('------ end correct answer ----------')

                logger.info('------ begin bad answers ----------')
                for sim_index in sims_index_sorted:
                    is_good_answer = False
                    answer = answers[sim_index][0]
                    print('answer ----: ' + str(answer))
                    for good_answer in d['good_answers']:
                        print(good_answer)
                        if np.array_equal(answer, good_answer):
                            is_good_answer = True
                            break
                    print('is_good_answer----: ' + str(is_good_answer))
                    if is_good_answer == False:
                        question_id = self.find_question_id(answer)
                        answer_index = answers.index(answer)
                        answer_rank = r[answer_index]
                        str_answer = 'Question Id (sof): ' + str(question_id) + ' - Rank: ' + answer_rank + ' - ' + ' '.join(self.revert(answer))
                        logger.info(str_answer)

                logger.info('------ end bad answers ----------')

            c_1 += 1 if max_r == max_n else 0
            position = r[max_r] - r[max_n] + 1
            c_2 += 1 / float(position)
            positions.append(position)

        top1 = c_1 / float(len(X))
        mrr = c_2 / float(len(X))

        logger.info('Top-1 Precision: %f' % top1)
        logger.info('MRR: %f' % mrr)

        return top1, mrr, positions
    def find_question_id(self, answer):
        index = self.answers.index(answer)
        question_id = self.answers_index[index]
        return question_id

    def save_score(self):
        with open('results_conf.txt', 'a+') as append_file:
            conf_json, name = self.conf.conf_json_and_name()
            top1_precisions = ','.join(self.top1_ls)
            mrrs = ','.join(self.mrr_ls)
            append_file.write(f'{name}; {conf_json}; top-1 precision: {top1_precisions}; MRR: {mrrs}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run question answer selection')
    parser.add_argument('--conf_file', metavar='CONF_FILE', type=str, default="stack_over_flow_conf.json", help='conf json file: stack_over_flow_conf.json')
    parser.add_argument('--mode', metavar='MODE', type=str, default="train", help='mode: train|evaluate|evaluate-best|evaluate-code-by-length|save_config')
    parser.add_argument('--conf_name', metavar='CONF_NAME', type=str, default=None, help='conf_name: part of name of weights file')
    parser.add_argument('--model', metavar='MODEL', type=str, default='cnn-lstm',
                        help='model name: embedding|cnn|cnn-lstm|rnn-attention')

    args = parser.parse_args()

    # configure logging
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))

    conf_file = args.conf_file
    mode = args.mode
    conf_name = args.conf_name
    model = args.model

    confs = json.load(open(conf_file, 'r'))
    from keras_models import EmbeddingModel, ConvolutionModel, ConvolutionalLSTM, UnifModel, SharedConvolutionModel
    from keras_models import SharedConvolutionModelWithBatchNormalization, ConvolutionModelWithBatchNormalization
    from keras_models import UnifModelWithBatchNormalization

    for conf in confs:
        logger.info(f'Conf.json: {conf}')
        evaluator = None
        if model == 'cnn-lstm':
            evaluator = Evaluator(conf, model=ConvolutionalLSTM, name=conf_name)
        elif model == 'embedding':
            evaluator = Evaluator(conf, model=EmbeddingModel, name=conf_name)
        elif model == 'cnn':
            evaluator = Evaluator(conf, model=ConvolutionModel, name=conf_name)
        elif model == 'shared-cnn':
            evaluator = Evaluator(conf, model=SharedConvolutionModel, name=conf_name)
        elif model == 'cnn-with-bn':
            evaluator = Evaluator(conf, model=ConvolutionModelWithBatchNormalization, name=conf_name)
        elif model == 'shared-cnn-with-bn':
            evaluator = Evaluator(conf, model=SharedConvolutionModelWithBatchNormalization, name=conf_name)
        elif model == 'attention':
            evaluator = Evaluator(conf, model=UnifModel, name=conf_name)
        elif model == 'attention-with-bn':
            evaluator = Evaluator(conf, model=UnifModelWithBatchNormalization, name=conf_name)

        # train and evaluate the model
        if evaluator is not None:
            evaluator.train_and_evaluate(mode)
        else:
            parser.print_help()
            sys.exit()

