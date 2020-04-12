import argparse

def generate_train_commands(dict):
    execution_command = ''
    wait_until_training_ends = ''
    git_command = ''

    if "margin" in dict:
        margin = str(dict["margin"]).replace('.', '')
        execution_command = f'!python3 stack_over_flow_qa_eval.py --model {dict["model"]} ' \
                            f'--conf_file conf/stack_over_flow_conf_f_{dict["filters"]}_k_{dict["kernel_size"]}_m_{margin}.json' \
                            f' &> output/training/{dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-m-{margin}-out.txt &'
        wait_until_training_ends = f'!while ! grep "<Figure" ' \
                                   f'output/training/{dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-m-{margin}-out.txt;do ' \
                                   f'tail -n 30 output/training/{dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-m-{margin}-out.txt; sleep 10;done'
        git_command = '!git status\n' \
                      '!git pull origin master\n' \
                      '!git add -A\n' \
                      f'!git commit -am "push {dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-m-{margin} results to github"\n' \
                      '!git push origin master\n'
    elif 'kernel_size' in dict:
        execution_command = f'!python3 stack_over_flow_qa_eval.py --model {dict["model"]} ' \
                            f'--conf_file conf/stack_over_flow_conf_f_{dict["filters"]}_k_{dict["kernel_size"]}.json' \
                            f' &> output/training/{dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-out.txt &'
        wait_until_training_ends = f'!while ! grep "<Figure" ' \
                                   f'output/training/{dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-out.txt;do ' \
                                   f'tail -n 30 output/training/{dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-out.txt; sleep 10;done'
        git_command = '!git status\n' \
                      '!git pull origin master\n' \
                      '!git add -A\n' \
                      f'!git commit -am "push {dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]} results to github"\n' \
                      '!git push origin master\n'
    else:
        execution_command = f'!python3 stack_over_flow_qa_eval.py --model {dict["model"]} ' \
                            f'--conf_file conf/stack_over_flow_conf_f_{dict["filters"]}.json' \
                            f' &> output/training/{dict["model"]}-{dict["filters"]}-out.txt &'
        wait_until_training_ends = f'!while ! grep "<Figure" ' \
                                   f'output/training/{dict["model"]}-{dict["filters"]}-out.txt;do ' \
                                   f'tail -n 30 output/training/{dict["model"]}-{dict["filters"]}-out.txt; sleep 10;done'
        git_command = '!git status\n' \
                      '!git pull origin master\n' \
                      '!git add -A\n' \
                      f'!git commit -am "push {dict["model"]}-{dict["filters"]} results to github"\n' \
                      '!git push origin master\n'

    if len(execution_command) > 0:
        print(execution_command)
        print()
        print(wait_until_training_ends)
        print()
        print(git_command)


def generate_evaluate_commands(dict):
    execution_command = ''
    git_command = ''

    if "margin" in dict:
        margin = str(dict["margin"]).replace('.', '')
        execution_command = f'!python3 stack_over_flow_qa_eval.py --model {dict["model"]} --mode evaluate ' \
                            f'--conf_file conf/stack_over_flow_conf_f_{dict["filters"]}_k_{dict["kernel_size"]}_m_{margin}.json' \
                            f' &> output/evaluation/{dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-m-{margin}-out.txt'
        git_command = '!git status\n' \
                      '!git pull origin master\n' \
                      '!git add -A\n' \
                      f'!git commit -am "push evaluation {dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-m-{margin} results to github"\n' \
                      '!git push origin master\n'
    elif "kernel_size" in dict:
        execution_command = f'!python3 stack_over_flow_qa_eval.py --model {dict["model"]} --mode evaluate ' \
                            f'--conf_file conf/stack_over_flow_conf_f_{dict["filters"]}_k_{dict["kernel_size"]}.json' \
                            f' &> output/evaluation/{dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-out.txt'

        git_command = '!git status\n' \
                      '!git pull origin master\n' \
                      '!git add -A\n' \
                      f'!git commit -am "push evaluation {dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]} results to github"\n' \
                      '!git push origin master\n'
    else:
        execution_command = f'!python3 stack_over_flow_qa_eval.py --model {dict["model"]} --mode evaluate ' \
                            f'--conf_file conf/stack_over_flow_conf_f_{dict["filters"]}.json' \
                            f' &> output/evaluation/{dict["model"]}-{dict["filters"]}-out.txt'
        git_command = '!git status\n' \
                      '!git pull origin master\n' \
                      '!git add -A\n' \
                      f'!git commit -am "push evaluation {dict["model"]}-{dict["filters"]} results to github"\n' \
                      '!git push origin master\n'


    if len(execution_command) > 0:
        print(execution_command)
        print()
        print(git_command)


def generate_evaluate_best_val_loss_commands(dict):
    execution_command = ''
    git_command = ''

    if "margin" in dict:
        margin = str(dict["margin"]).replace('.', '')
        execution_command = f'!python3 stack_over_flow_qa_eval.py --model {dict["model"]} --mode evaluate-best ' \
                            f'--conf_file conf/stack_over_flow_conf_f_{dict["filters"]}_k_{dict["kernel_size"]}_m_{margin}.json' \
                            f' &> output/evaluation/best/{dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-m-{margin}-out.txt'
        git_command = '!git status\n' \
                      '!git pull origin master\n' \
                      '!git add -A\n' \
                      f'!git commit -am "push evaluation best val loss {dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-m-{margin} results to github"\n' \
                      '!git push origin master\n'
    elif "kernel_size" in dict:
        execution_command = f'!python3 stack_over_flow_qa_eval.py --model {dict["model"]} --mode evaluate-best ' \
                            f'--conf_file conf/stack_over_flow_conf_f_{dict["filters"]}_k_{dict["kernel_size"]}.json' \
                            f' &> output/evaluation/best/{dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]}-out.txt'

        git_command = '!git status\n' \
                      '!git pull origin master\n' \
                      '!git add -A\n' \
                      f'!git commit -am "push evaluation best val loss {dict["model"]}-{dict["filters"]}-k-{dict["kernel_size"]} results to github"\n' \
                      '!git push origin master\n'
    else:
        execution_command = f'!python3 stack_over_flow_qa_eval.py --model {dict["model"]} --mode evaluate-best ' \
                            f'--conf_file conf/stack_over_flow_conf_f_{dict["filters"]}.json' \
                            f' &> output/evaluation/best/{dict["model"]}-{dict["filters"]}-out.txt'
        git_command = '!git status\n' \
                      '!git pull origin master\n' \
                      '!git add -A\n' \
                      f'!git commit -am "push evaluation best val loss {dict["model"]}-{dict["filters"]} results to github"\n' \
                      '!git push origin master\n'


    if len(execution_command) > 0:
        print(execution_command)
        print()
        print(git_command)        


embeddings = [{'model': 'embedding', 'filters': 4000, 'kernel_size': 2},
        {'model': 'embedding', 'filters': 4000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'embedding', 'filters': 4000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'embedding', 'filters': 4000, 'kernel_size': 2, 'margin': 0.2}]

attentions = [{'model': 'attention', 'filters': 4000, 'kernel_size': 2},
        {'model': 'attention', 'filters': 4000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'attention', 'filters': 4000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'attention', 'filters': 4000, 'kernel_size': 2, 'margin': 0.2}]

attentions_with_bn = [{'model': 'attention-with-bn', 'filters': 4000, 'kernel_size': 2},
        {'model': 'attention-with-bn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'attention-with-bn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'attention-with-bn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.2}]


cnns = [{'model': 'cnn', 'filters': 50},
        {'model': 'cnn', 'filters': 50, 'kernel_size': 2},
        {'model': 'cnn', 'filters': 100},
        {'model': 'cnn', 'filters': 100, 'kernel_size': 2},
        {'model': 'cnn', 'filters': 200},
        {'model': 'cnn', 'filters': 200, 'kernel_size': 2},
        {'model': 'cnn', 'filters': 500},
        {'model': 'cnn', 'filters': 500, 'kernel_size': 2},
        {'model': 'cnn', 'filters': 1000},
        {'model': 'cnn', 'filters': 1000, 'kernel_size': 2},
        {'model': 'cnn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'cnn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'cnn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.2},
        {'model': 'cnn', 'filters': 1000, 'kernel_size': 3},
        {'model': 'cnn', 'filters': 1000, 'kernel_size': 4},
        {'model': 'cnn', 'filters': 2000},
        {'model': 'cnn', 'filters': 2000, 'kernel_size': 2},
        {'model': 'cnn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'cnn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'cnn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.2},
        {'model': 'cnn', 'filters': 2000, 'kernel_size': 3},
        {'model': 'cnn', 'filters': 2000, 'kernel_size': 4},
        {'model': 'cnn', 'filters': 4000},
        {'model': 'cnn', 'filters': 4000, 'kernel_size': 2},
        {'model': 'cnn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'cnn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'cnn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.2},
        {'model': 'cnn', 'filters': 4000, 'kernel_size': 3},
        {'model': 'cnn', 'filters': 4000, 'kernel_size': 4}]

shared_cnns = [

        {'model': 'shared-cnn', 'filters': 1000, 'kernel_size': 2},
        {'model': 'shared-cnn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'shared-cnn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'shared-cnn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.2},


        {'model': 'shared-cnn', 'filters': 2000, 'kernel_size': 2},
        {'model': 'shared-cnn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'shared-cnn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'shared-cnn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.2},

        {'model': 'shared-cnn', 'filters': 4000, 'kernel_size': 2},
        {'model': 'shared-cnn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'shared-cnn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'shared-cnn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.2}

               ]

cnns_with_bn = [

        {'model': 'cnn-with-bn', 'filters': 1000, 'kernel_size': 2},
        {'model': 'cnn-with-bn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'cnn-with-bn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'cnn-with-bn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.2},

        {'model': 'cnn-with-bn', 'filters': 2000, 'kernel_size': 2},
        {'model': 'cnn-with-bn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'cnn-with-bn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'cnn-with-bn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.2},

        {'model': 'cnn-with-bn', 'filters': 4000, 'kernel_size': 2},
        {'model': 'cnn-with-bn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'cnn-with-bn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'cnn-with-bn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.2}

                ]

shared_cnns_with_bn = [

        {'model': 'shared-cnn-with-bn', 'filters': 1000, 'kernel_size': 2},
        {'model': 'shared-cnn-with-bn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'shared-cnn-with-bn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'shared-cnn-with-bn', 'filters': 1000, 'kernel_size': 2, 'margin': 0.2},

        {'model': 'shared-cnn-with-bn', 'filters': 2000, 'kernel_size': 2},
        {'model': 'shared-cnn-with-bn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'shared-cnn-with-bn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'shared-cnn-with-bn', 'filters': 2000, 'kernel_size': 2, 'margin': 0.2},

        {'model': 'shared-cnn-with-bn', 'filters': 4000, 'kernel_size': 2},
        {'model': 'shared-cnn-with-bn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.05},
        {'model': 'shared-cnn-with-bn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.1},
        {'model': 'shared-cnn-with-bn', 'filters': 4000, 'kernel_size': 2, 'margin': 0.2}

                       ]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate train execution')

    parser.add_argument('--mode', metavar='mode', type=str, default='training',
                        help='--mode <training|evaluate|evaluate-best>')


    args = parser.parse_args()
    mode = args.mode
    if mode == 'training':
        print('--------- cnn --------------')
        for cnn in cnns:
                generate_train_commands(cnn)

        print('--------- shared-cnn --------------')
        for cnn in shared_cnns:
            generate_train_commands(cnn)

        print('--------- cnn with bn --------------')
        for cnn in cnns_with_bn:
                generate_train_commands(cnn)

        print('--------- shared-cnn with bn --------------')
        for cnn in shared_cnns_with_bn:
                generate_train_commands(cnn)
        print('--------- embedding --------------')
        for embedding in embeddings:
                generate_train_commands(embedding)

        print('--------- attention --------------')
        for attention in attentions:
            generate_train_commands(attention)

        print('--------- attentions-with-bn --------------')
        for attention_with_bn in attentions_with_bn:
            generate_train_commands(attention_with_bn)
    elif mode == 'evaluate':
        print('--------- cnn --------------')
        for cnn in cnns:
            generate_evaluate_commands(cnn)

        print('--------- shared-cnn --------------')
        for cnn in shared_cnns:
            generate_evaluate_commands(cnn)

        print('--------- cnn with bn --------------')
        for cnn in cnns_with_bn:
            generate_evaluate_commands(cnn)

        print('--------- shared-cnn with bn --------------')
        for cnn in shared_cnns_with_bn:
            generate_evaluate_commands(cnn)

        print('--------- embedding --------------')
        for embedding in embeddings:
            generate_evaluate_commands(embedding)

        print('--------- attention --------------')
        for attention in attentions:
            generate_evaluate_commands(attention)

        print('--------- attentions-with-bn --------------')
        for attention_with_bn in attentions_with_bn:
            generate_evaluate_commands(attention_with_bn)
    elif mode == 'evaluate-best':
        print('--------- cnn --------------')
        for cnn in cnns:
            generate_evaluate_best_val_loss_commands(cnn)

        print('--------- shared-cnn --------------')
        for cnn in shared_cnns:
            generate_evaluate_best_val_loss_commands(cnn)

        print('--------- cnn with bn --------------')
        for cnn in cnns_with_bn:
            generate_evaluate_best_val_loss_commands(cnn)

        print('--------- shared-cnn with bn --------------')
        for cnn in shared_cnns_with_bn:
            generate_evaluate_best_val_loss_commands(cnn)

        print('--------- embedding --------------')
        for embedding in embeddings:
            generate_evaluate_best_val_loss_commands(embedding)

        print('--------- attention --------------')
        for attention in attentions:
            generate_evaluate_best_val_loss_commands(attention)

        print('--------- attentions-with-bn --------------')
        for attention_with_bn in attentions_with_bn:
            generate_evaluate_best_val_loss_commands(attention_with_bn)
    else:
        parser.print_help()
        sys.exit()

