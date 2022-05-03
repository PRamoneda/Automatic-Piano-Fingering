import os
import shutil
import subprocess

import numpy as np

from utils import load_json


def system_call(command):
    p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    return p.stdout.read()


def run_PIG():
    """
    ./FingeringHMM3_train list_train.txt DataFolder paramHeader(param_FHMM1)
    $./FingeringHMM3_OptHypara list_valid.txt DataFolder param.txt hypara.txt
    ./FingeringHMM3_Run param.txt hypara.txt in_fingering.txt out_fingering.txt
    $./Evaluate_MultipleGroundTruth nGT GT_1_fin.txt GT_2_fin.txt ... GT_nGT_fin.txt est_fingering.txt

    train.txt
    """
    validation_data = load_json('PianoFingeringDataset_v1.02/official_split.json')

    base_code = 'nakamura_SourceCode_intermittent/Binary'

    results = {}

    script_run = {'HMM1': 'FingeringHMM1_Run', 'HMM2': 'FingeringHMM2_Run', 'HMM3': 'FingeringHMM3_Run'}
    script_tune = {'HMM1': 'FingeringHMM1_OptHypara', 'HMM2': 'FingeringHMM2_OptHypara', 'HMM3': 'FingeringHMM3_OptHypara'}
    script_train = {'HMM1': 'FingeringHMM1_Train', 'HMM2': 'FingeringHMM2_Train', 'HMM3': 'FingeringHMM3_Train'}

    for name_experiment in ['HMM1', 'HMM2', 'HMM3']:
        print(name_experiment)
        train_list = f'PianoFingeringDataset_v1.02/official_results/PIG/{name_experiment}/train.txt'
        val_list = f'PianoFingeringDataset_v1.02/official_results/PIG/{name_experiment}/validation.txt'

        base = f'PianoFingeringDataset_v1.02/official_results/PIG/{name_experiment}'
        data = 'PianoFingeringDataset_v1.02/FingeringFiles'

        # 1. train the model
        # os.system(f"./{base_code}/{script_train[name_experiment]} {train_list} {data} {base}/param")
        # 2. finetuning hyperparameters
        # os.system(f"./{base_code}/{script_tune[name_experiment]} {val_list} {data} {base}/param.txt {base}/hyper_param")
        # 3. Instatiate test
        # f = open(f"{base}/test_input.txt", "r")
        # test_input = [ff.replace('\n', "") + '_fingering.txt' for ff in f.readlines()]
        f = open(f"{base}/test_output.txt", "r")
        test_output = [ff.replace('\n', "") + '_fingering.txt' for ff in f.readlines()]
        # for test_input_score, test_output_score in zip(test_input, test_output):
        #     test_input_score = test_input_score.replace('\n', "")
        #     test_output_score = test_output_score.replace('\n', "")
        #     print(f"./{base_code}/{script_run[name_experiment]} {base}/param.txt {base}/hyper_param {test_input_score} {test_output_score}")
        #     os.system(f"./{base_code}/{script_run[name_experiment]} {base}/param.txt {base}/hyper_param {test_input_score} {test_output_score}")
        # 4. evaluate
        f = open(f"{base}/test.txt", "r")
        test = [ff.replace('\n', "")[:3] for ff in f.readlines()]

        f = open(f"{base}/test_input.txt", "r")
        test_input = [ff.replace('\n', "") + '_fingering.txt' for ff in f.readlines()]
        results[name_experiment] = np.array([0, 0, 0, 0])
        for test_score in test:
            test_predicted = [tt for tt in test_output if test_score in tt][0]
            test_GT = [f'PianoFingeringDataset_v1.02/FingeringFiles/{tt}'
                       for tt in os.listdir('PianoFingeringDataset_v1.02/FingeringFiles')
                       if test_score in tt]
            result = system_call(f"./{base_code}/Evaluate_MultipleGroundTruth {len(test_GT)} {' '.join(test_GT)} {test_predicted}").decode("utf-8")
            print(f"./{base_code}/Evaluate_MultipleGroundTruth {len(test_GT)} {' '.join(test_GT)} {test_predicted}")
            print(result)
            result = np.array([float(x) for x in result.split(': ')[1].replace('\n', '').split('\t')[:4]])
            results[name_experiment] = results[name_experiment] + result
        results[name_experiment] = results[name_experiment] / len(test)

    print(results)

    print("=" * 14)
    for number_experiment, m in results.items():
        print(f"Model: {number_experiment}")
        print("=" * 14)
        print(f"GMR: {m[0]}, HMR: {m[1]}, SMR: {m[2]}, RMR: {m[3]}")
        print("=" * 14)


def run_thumbset():
    validation_data = load_json('PianoFingeringDataset_v1.02/official_split.json')

    base_code = 'nakamura_SourceCode_intermittent/Binary'

    results = {}

    script_run = {'HMM1': 'FingeringHMM1_Run', 'HMM2': 'FingeringHMM2_Run', 'HMM3': 'FingeringHMM3_Run'}
    script_tune = {'HMM1': 'FingeringHMM1_OptHypara', 'HMM2': 'FingeringHMM2_OptHypara',
                   'HMM3': 'FingeringHMM3_OptHypara'}
    script_train = {'HMM1': 'FingeringHMM1_Train', 'HMM2': 'FingeringHMM2_Train', 'HMM3': 'FingeringHMM3_Train'}
    script_semi_supervised = {
        'HMM1': 'FingeringHMM1_VitertbiTrain',
        'HMM2': 'FingeringHMM1_VitertbiTrain',
        'HMM3': 'FingeringHMM1_VitertbiTrain'
    }

    for name_experiment in ['HMM1']:  #, 'HMM2', 'HMM3']:
        print(name_experiment)
        train_list = f'PianoFingeringDataset_v1.02/official_results/thumbset/{name_experiment}/train.txt'
        train_thumbset_list = f'PianoFingeringDataset_v1.02/official_results/thumbset/{name_experiment}/train.txt'
        val_list = f'PianoFingeringDataset_v1.02/official_results/thumbset/{name_experiment}/validation.txt'

        base = f'PianoFingeringDataset_v1.02/official_results/thumbset/{name_experiment}'
        data = 'PianoFingeringDataset_v1.02/FingeringFiles'
        data_thumbset = 'PianoFingeringDataset_v1.02/MixedWithPIG'

        # Assuming data split: PIG_train, PIG_valid, PIG_test, and ThumbSet.
        # 1. Supervised learning of HMM with PIG_train
        os.system(f"./{base_code}/{script_train[name_experiment]} {train_list} {data} {base}/param")
        # 2. Hyperparameter optimization with PIG_valid
        os.system(f"./{base_code}/{script_tune[name_experiment]} {val_list} {data} {base}/param.txt {base}/hyper_param")
        # 3. Semi-supervised learning of HMM with ThumbSet and PIG_train mixed
        os.system(f"./{base_code}/{script_semi_supervised[name_experiment]} {base}/param.txt {base}/hyper_param {train_thumbset_list} {data_thumbset} {base}/param_semi")
        # 4. Hyperparameter optimization with PIG_valid with the new parameters
        os.system(f"./{base_code}/{script_tune[name_experiment]} {val_list} {data} {base}/param_semi {base}/hyper_param_semi")
        # 5. Fingering the test pieces
        f = open(f"{base}/test_input.txt", "r")
        test_input = [ff.replace('\n', "") + '_fingering.txt' for ff in f.readlines()]
        f = open(f"{base}/test_output.txt", "r")
        test_output = [ff.replace('\n', "") + '_fingering.txt' for ff in f.readlines()]
        for test_input_score, test_output_score in zip(test_input, test_output):
            test_input_score = test_input_score.replace('\n', "")
            test_output_score = test_output_score.replace('\n', "")
            print(f"./{base_code}/{script_run[name_experiment]} {base}/param_semi {base}/hyper_param_semi {test_input_score} {test_output_score}")
            os.system(f"./{base_code}/{script_run[name_experiment]} {base}/param_semi {base}/hyper_param_semi {test_input_score} {test_output_score}")
        # 6. evaluate
        f = open(f"{base}/test.txt", "r")
        test = [ff.replace('\n', "")[:3] for ff in f.readlines()]
        results[name_experiment] = np.array([0, 0, 0, 0])
        for test_score in test:
            test_predicted = [tt for tt in test_output if test_score in tt][0]
            test_GT = [f'PianoFingeringDataset_v1.02/FingeringFiles/{tt}'
                       for tt in os.listdir('PianoFingeringDataset_v1.02/FingeringFiles')
                       if test_score in tt]
            result = system_call(
                f"./{base_code}/Evaluate_MultipleGroundTruth {len(test_GT)} {' '.join(test_GT)} {test_predicted}").decode(
                "utf-8")
            result = np.array([float(x) for x in result.split(': ')[1].replace('\n', '').split('\t')[:4]])
            results[name_experiment] = results[name_experiment] + result
        results[name_experiment] = results[name_experiment] / len(test)


    print(results)

    print("=" * 14)
    for number_experiment, m in results.items():
        print(f"Model: {number_experiment}")
        print("=" * 14)
        print(f"GMR: {m[0]}, HMR: {m[1]}, SMR: {m[2]}, RMR: {m[3]}")
        print("=" * 14)


if __name__ == '__main__':
    run_PIG()
    # run_thumbset()
