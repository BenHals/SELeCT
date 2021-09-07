import argparse
import json
import os
import pathlib
import pandas as pd

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--dataset', default="cmc", type=str)
    my_parser.add_argument('--cpus', default=1, type=int)
    my_parser.add_argument('--run_minimal', action='store_true')
    args = my_parser.parse_args()

    # Run experiment
    seeds = ' '.join(map(str, (range(1, 46) if not args.run_minimal else range(1, 5))))
    # Options are:
    # Dataset -> command
    # AQS -> 'AQSex'
    # AQT -> 'AQTemp'
    # AD -> 'Arabic'
    # CMC -> 'cmc'
    # STGR -> 'STAGGERS'
    # TREE -> 'RTREESAMPLE_HARD' 
    # WIND -> 'WINDSIM' 
    dataset = args.dataset
    cmd_str = f'python ../run_experiment.py --forcegitcheck --seeds {seeds} --seedaction list --datalocation ../RawData --datasets {dataset} --experimentname cmc_testbed {"--single" if args.cpus <= 1 else "--cpu " + str(args.cpus)} --outputlocation ../output --loglocation ../experimentlog'
    os.system(f'{cmd_str}')

    # Collect results
    file_path = pathlib.Path(__file__)
    output = file_path.resolve().parents[1] / 'output' / 'cmc_testbed'
    results_files = list(output.rglob('results*'))
    results = []
    for rp in results_files:
        result = json.load(rp.open('r'))
        results.append(result)
    
    df = pd.DataFrame(results)
    
    classification_performance_metric = 'kappa'
    adaption_performance_metric = 'GT_mean_f1' # measure for C-F1
    print("**************TEST RESULTS**************")
    print(df.groupby(['data_name', 'classifier'])[[classification_performance_metric, adaption_performance_metric]].mean())



