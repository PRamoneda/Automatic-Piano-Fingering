import os
import shutil
import subprocess

import numpy as np

from utils import load_json
# {'1': array([0.62935177, 0.69856113, 0.83650994, 0.75906329]), '2': array([0.63798815, 0.70694138, 0.84572887,
# 0.76958473]), '3': array([0.63119613, 0.70021963, 0.83931609, 0.76275095]), '4': array([0.63689489, 0.70589265,
# 0.84808504, 0.77328506]), '5': array([0.63594062, 0.70501653, 0.8421446 , 0.76650139])}

def system_call(command):
    p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    return p.stdout.read()

"""
./FingeringHMM3_train list_train.txt DataFolder paramHeader(param_FHMM1)
$./FingeringHMM3_OptHypara list_valid.txt DataFolder param.txt hypara.txt
./FingeringHMM3_Run param.txt hypara.txt in_fingering.txt out_fingering.txt
$./Evaluate_MultipleGroundTruth nGT GT_1_fin.txt GT_2_fin.txt ... GT_nGT_fin.txt est_fingering.txt
 
train.txt
"""
validation_data = load_json('PianoFingeringDataset_v1.02/validation_experiments_splits.json')

base_code = 'nakamura_SourceCode_validation/Binary'

results = {}

for number_experiment, split in validation_data.items():
    train_list = f'PianoFingeringDataset_v1.02/validation_experiments/{number_experiment}/train.txt'
    val_list = f'PianoFingeringDataset_v1.02/validation_experiments/{number_experiment}/validation.txt'
    test_input_list = f'PianoFingeringDataset_v1.02/validation_experiments/{number_experiment}/test_input.txt'
    test_output_list = f'PianoFingeringDataset_v1.02/validation_experiments/{number_experiment}/test_output.txt'

    base = f'PianoFingeringDataset_v1.02/validation_experiments/{number_experiment}'
    data = 'PianoFingeringDataset_v1.02/FingeringFiles'

    # # train the model
    # os.system(f"./{base_code}/FingeringHMM3_train {train_list} {data} {base}/param")
    # finetuning hyperparameters
    # os.system(f"./{base_code}/FingeringHMM3_OptHypara {val_list} {data} {base}/param.txt {base}/hyper_param")
    # Instatiate test
    # f = open(f"{base}/test_input.txt", "r")
    # test_input = [ff.replace('\n', "") + '_fingering.txt' for ff in f.readlines()]
    f = open(f"{base}/test_output.txt", "r")
    test_output = [ff.replace('\n', "") + '_fingering.txt' for ff in f.readlines()]
    # for test_input_score, test_output_score in zip(test_input, test_output):
    #     test_input_score = test_input_score.replace('\n', "")
    #     test_output_score = test_output_score.replace('\n', "")
    #     print(f"./{base_code}/FingeringHMM3_Run {base}/param.txt {base}/hyper_param {test_input_score} {test_output_score}")
    #     os.system(f"./{base_code}/FingeringHMM3_Run {base}/param.txt {base}/hyper_param {test_input_score} {test_output_score}")
    # evaluate
    f = open(f"{base}/test.txt", "r")
    test = [ff.replace('\n', "")[:3] for ff in f.readlines()]

    f = open(f"{base}/test_input.txt", "r")
    test_input = [ff.replace('\n', "") + '_fingering.txt' for ff in f.readlines()]
    results[number_experiment] = np.array([0, 0, 0, 0])
    for test_score in test:
        test_predicted = [tt for tt in test_output if test_score in tt][0]
        test_GT = [f'PianoFingeringDataset_v1.02/FingeringFiles/{tt}'
                   for tt in os.listdir('PianoFingeringDataset_v1.02/FingeringFiles')
                   if test_score in tt]
        result = system_call(f"./{base_code}/Evaluate_MultipleGroundTruth {len(test_GT)} {' '.join(test_GT)} {test_predicted}").decode("utf-8")
        result = np.array([float(x) for x in result.split(': ')[1].replace('\n', '').split('\t')[:4]])
        results[number_experiment] = results[number_experiment] + result
    results[number_experiment] = results[number_experiment] / len(test)

print(results)

print("=" * 14)
for number_experiment, m in results.items():
    print(f"Number validation experiment: {number_experiment}")
    print("=" * 14)
    print(f"GMR: {m[0]}, HMR: {m[1]}, SMR: {m[2]}, RMR: {m[3]}")
    print("=" * 14)


