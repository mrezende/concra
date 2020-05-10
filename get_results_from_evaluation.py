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
    top5 = None
    top10 = None
    eval_code_length = None
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

def calculate_top5(positions):
    count = 0
    for pos in positions:
        if pos <= 5:
            count += 1

    return count / len(positions)

def calculate_top10(positions):
    count = 0
    for pos in positions:
        if pos <= 10:
            count += 1

    return count / len(positions)

results = []
model_name_regx = re.compile('((([a-z]{2,9}-)+)([0-9]{2,4}))-(k-[0-9])?-?(m-[0-9]{2,3})?-?out\.txt')
mypath = 'output/evaluation/code-length/'
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

top1_line = 'Top1 Description'
mrr_line = 'MRR Description'
final_results_desc = 'INFO: final_results:'
eval_desc = 'eval: '

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

    
    

    with open(join(mypath, filename), 'r') as f:
        line = f.readline()
        result = None
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
            elif final_results_desc in line:
                aux = line
                open_curly_braces_pos = aux.find('{')
                final_results = eval(aux[open_curly_braces_pos:])
                positions = [pos for i in final_results['positions'] for pos in i]

                result = Result(full_model_name, model_name, filters, kernel_size, margin)

                result.top5 = calculate_top5(positions)
                result.top10 = calculate_top10(positions)

                results.append(result)

                line = f.readline()
            elif eval_desc in line:
                aux = line
                start_pos = aux.find('eval_')
                end_pos = aux.find('.json')
                eval_code_length = aux[start_pos: end_pos]

                line = f.readline()

                aux = line
                open_curly_braces_pos = aux.find('{')
                final_results = eval(aux[open_curly_braces_pos:])
                positions = [pos for i in final_results['positions'] for pos in i]

                result = Result(full_model_name, model_name, filters, kernel_size, margin)

                result.top5 = calculate_top5(positions)
                result.top10 = calculate_top10(positions)
                result.eval_code_length = eval_code_length

                results.append(result)

                line = f.readline()

            else:
                line = f.readline()

    


sorted_results = sorted(results, key=attrgetter('model_name', 'filters'))

for k, g in groupby(sorted_results, lambda m: m.full_model_name if m.eval_code_length is None else f'{m.full_model_name} - {m.eval_code_length}'):
    count = 0
    line = ''
    for i in sorted(g, key=attrgetter('kernel', 'margin')):
        if count == 0:
            line += f'{i.full_model_name}, ' if i.eval_code_length is None else f'{i.full_model_name} - {i.eval_code_length},'
            if i.kernel != default_kernel_size:
                line += f'X,X,X,X,{i.mrr} +- {i.mrr_std}, {i.top1} +- {i.top1_std},{i.top5}, {i.top10},'
                count += 1
            else:
                line += f'{i.mrr} +- {i.mrr_std}, {i.top1} +- {i.top1_std},{i.top5}, {i.top10},'
        else:
            line += f'{i.mrr} +- {i.mrr_std}, {i.top1} +- {i.top1_std},{i.top5}, {i.top10},'

        count += 1
    while count < 7:
        line += f'X,X,X,X,'
        count += 1
    print(line)