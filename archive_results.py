import os
import shutil
import datetime
from configuration import Conf


class ArchiveResults:
    def __init__(self):
        self.base_folder = datetime.datetime.now().strftime(
            os.path.join('reports', 'training', 'results_%m_%d_%Y', '%H_%M_%S'))
        os.makedirs(self.base_folder, exist_ok=True)

    def save_training_results(self, conf_json):
        conf = Conf(conf_json)
        # move plots to archive folder
        # plot_folder = 'plots'
        # plot_filename = f'{conf.name()}_plot.png'
        # plot_file = os.path.join(plot_folder, plot_filename)

        # os.makedirs(os.path.join(self.base_folder, plot_folder), exist_ok=True)
        # archive_plot_result_file = os.path.join(self.base_folder, plot_file)

        # self.move(plot_file, archive_plot_result_file)

        # copy models to archive folder
        models_folder = 'models'
        models_file = f'models/weights_epoch_{conf.name()}.h5'

        os.makedirs(os.path.join(self.base_folder, models_folder), exist_ok=True)
        archive_models_file = os.path.join(self.base_folder, models_file)

        self.copy(models_file, archive_models_file)



    def save_predict_results(self, conf_json):
        conf = Conf(conf_json)
        # copy models to archive folder
        models_folder = 'models'
        models_file = f'models/weights_epoch_{conf.name()}.h5'

        os.makedirs(os.path.join(self.base_folder, models_folder), exist_ok=True)
        archive_models_file = os.path.join(self.base_folder, models_file)

        self.copy(models_file, archive_models_file)

        score_file = 'results_conf.txt'
        archive_score_file = os.path.join(self.base_folder, score_file)
        self.move(score_file, archive_score_file)

    def save_conf_list(self):
        conf_list = 'conf_list.txt'
        archive_conf_list = os.path.join(self.base_folder, conf_list)
        self.move(conf_list, archive_conf_list)

    def save_conf_file(self, conf_file):
        archive_conf_file = os.path.join(self.base_folder, conf_file)
        self.copy(conf_file, archive_conf_file)

    def move(self, src, dest):
        shutil.move(src, dest)

    def copy(self, src, dest):
        shutil.copy(src, dest)
