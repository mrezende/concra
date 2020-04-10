import json
import hashlib

class Conf:
    def __init__(self, conf):
        if isinstance(conf, str):
            conf = json.load(open(conf, 'rb'))
        self.conf = conf


    def name(self):
        str = json.dumps(self.conf)
        m = hashlib.sha256()
        m.update(str.encode('utf-8'))
        return m.hexdigest()[:6]

    def training_params(self):
        return self.conf['training']

    def question_len(self):
        return self.conf.get('question_len', None)

    def answer_len(self):
        return self.conf.get('answer_len', None)

    def similarity_params(self):
        return self.conf.get('similarity', dict())

    def margin(self):
        return self.conf['margin']

    def filters(self):
        return self.conf.get('filters', None)

    def kernel_size(self):
        return self.conf.get('kernel_size')

    def initial_embed_weights(self):
        return self.conf['initial_embed_weights']

    def initial_question_weights(self):
        return self.conf['initial_question_weights']

    def initial_answer_weights(self):
        return self.conf['initial_answer_weights']

    def conf_json_and_name(self):
        conf_json = json.dumps(self.conf)
        name = self.name()
        return conf_json, name

    def save_conf(self):
        conf_json, name = self.conf_json_and_name()
        with open(f'output/conf_list_{name}.txt', 'a+') as append_file:
            append_file.write(f'{name};{conf_json}\n')