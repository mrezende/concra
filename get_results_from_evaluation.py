from os import listdir
from os.path import isfile, join
import re
from itertools import groupby
from operator import attrgetter

class Result:
    top1 = 0
    top1_std = 0
    mrr = 0
    mrr_std = 0
    def __init__(self, full_model_name, model_name, filters, kernel_size, margin):
        self.full_model_name = full_model_name
        self.model_name = model_name
        self.filters = filters
        self.kernel = kernel_size
        self.margin = margin


def get_decimal(line):
    decimal_regx = re.compile('0\.[0-9]+')
    match = decimal_regx.search(line)
    if match:
        return match.group()
    else:
        return ''

results = []
model_name_regx = re.compile('((([a-z]{2,9}-)+)([0-9]{2,4}))-(k-[0-9])?-?(m-[0-9]{2,3})?-?out\.txt')
mypath = 'output/evaluation/'
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

top1_line = 'Top1 Description'
mrr_line = 'MRR Description'

default_kernel_size = '1,2,3,5,7'
default_margin = '0009'

for filename in filenames:
    m = model_name_regx.match(filename)
    full_model_name = m.group(1)
    model_name = m.group(2)

    filters = int(m.group(4))

    kernel_size = '2'
    kernel_match = m.group(5)
    if kernel_match is None:
        if 'cnn' in model_name:
            kernel_size = default_kernel_size
    else:
        kernel_size = kernel_match.split('-')[1]

    margin = '0009'
    margin_match = m.group(6)
    if margin_match is None:
        if 'cnn' in model_name:
            margin = default_margin
    else:
        margin = margin_match.split('-')[1]

    result = Result(full_model_name, model_name, filters, kernel_size, margin)

    with open(join(mypath, filename), 'r') as f:
        line = f.readline()
        while line:
            if top1_line in line or mrr_line in line:

                mean_line = f.readline()
                mean = get_decimal(mean_line)

                std_line = f.readline()
                std = get_decimal(std_line)

                if top1_line in line:
                    result.top1 = mean
                    result.top1_std = std
                else:
                    result.mrr = mean
                    result.mrr_std = std

                line = f.readline()
            else:
                line = f.readline()

    results.append(result)


sorted_results = sorted(results, key=attrgetter('model_name', 'filters'))

for k, g in groupby(sorted_results, lambda m: m.full_model_name):
    count = 0
    line = ''
    for i in sorted(g, key=attrgetter('kernel', 'margin')):
        if count == 0:
            line += f'{i.full_model_name}, '
            if i.kernel != default_kernel_size:
                line += f'X,X,{i.mrr} +- {i.mrr_std}, {i.top1} +- {i.top1_std},'
                count += 1
            else:
                line += f'{i.mrr} +- {i.mrr_std}, {i.top1} +- {i.top1_std},'
        else:
            line += f'{i.mrr} +- {i.mrr_std}, {i.top1} +- {i.top1_std},'

        count += 1
    while count < 7:
        line += f'X,X,'
        count += 1
    print(line)