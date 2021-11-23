This is the codebase for the paper: [Do Language Models Have Beliefs? Methods for Detecting, Updating, and Visualizing Model Beliefs](link)

## Directory Structure

```
data/ --> data folder including splits we use for FEVER, zsRE, Wikidata5m, and LeapOfThought
training_reports/ --> folder to be populated with individual training run reports produced by main.py
result_sheets/ --> folder to be populated with .csv's of results from experiments produced by main.py
aggregated_results/ --> contains combined experiment results produced by run_jobs.py
outputs/ --> folder to be populated with analysis results, including belief graphs and bootstrap outputs
models/ --> contains model wrappers for Huggingface models and the learned optimizer code
data_utils/ --> contains scripts for making all datasets used in paper
main.py --> main script for all individual experiments in the paper
metrics.py --> functions for calculing metrics reported in the paper
utils.py --> data loading and miscellaneous utilities
run_jobs.py --> script for running groups of experiments
statistical_analysis.py --> script for running bootstraps with the experimental results
data_analysis.Rmd --> R markdown file that makes plots using .csv's in result_sheets
requirements.txt --> contains required packages
```

## Requirements

The code is compatible with Python 3.6+. `data_analysis.Rmd` is an R markdown file that makes all the plots in the paper.

The required packages can be installed by running:

`pip install -r requirements.txt`

If you wish to visualize belief graphs, you should also install a few packages as so:

`sudo apt install python-pydot python-pydot-ng graphviz`

## Making Data

We include the data splits from the paper in `data/` (though the train split for Wikidata5m is divided into two files that need to be locally combined.) To construct the datasets from scratch, you can follow a few steps:

1. Set the `DATA_DIR` environment variable to where you'd like the data to be stored. Set the `CODE_DIR` to point to the directory where this code is.
2. Run the following blocks of code

Make FEVER and ZSRE 
```
cd $DATA_DIR
git clone https://github.com/facebookresearch/KILT.git
cd KILT
mkdir data
python scripts/download_all_kilt_data.py
mv data/* ./
cd $CODE_DIR
python data_utils/shuffle_fever_splits.py
python data_utils/shuffle_zsre_splits.py
```

Make Leap-Of-Thought  
```
cd $DATA_DIR
git clone https://github.com/alontalmor/LeapOfThought.git
cd LeapOfThought
python -m LeapOfThought.run -c Hypernyms --artiset_module soft_reasoning -o build_artificial_dataset -v training_mix -out taxonomic_reasonings.jsonl.gz
gunzip taxonomic_reasonings_training_mix_train.jsonl.gz taxonomic_reasonings_training_mix_dev.jsonl.gz taxonomic_reasonings_training_mix_test.jsonl.gz taxonomic_reasonings_training_mix_meta.jsonl.gz
cd $CODE_DIR
python data_utils/shuffle_leapofthought_splits.py
```

Make Wikidata5m  
```
cd $DATA_DIR
mkdir Wikidata5m
cd Wikidata5m
wget https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz
wget https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz
tar -xvzf wikidata5m_transductive.tar.gz
tar -xvzf wikidata5m_alias.tar.gz
cd $CODE_DIR
python data_utils/filter_wikidata.py
```

## Experiment Replication

Experiment commands require a few arguments: `--data_dir` points to where the data is. `--save_dir` points to where models should be saved. `--cache_dir` points to where pretrained models will be stored. `--gpu` indicates the GPU device number. `--seeds` indicates how many seeds per condition to run. We give commands below for the experiments in the paper, saving everything in `$DATA_DIR`.

To train the task and prepare the necessary data for training learned optimizers, run:

```
python run_jobs.py -e task_model --seeds 5 --dataset all --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR
python run_jobs.py -e write_LeapOfThought_preds --seeds 5 --dataset LeapOfThought --do_train false --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR
```

To get the main experiments in a single-update setting, run:

`python run_jobs.py -e learned_opt_main --seeds 5 --dataset all --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR`

For results in a sequential-update setting (with r=10) run:

`python run_jobs.py -e learned_opt_r_main --seeds 5 --dataset all --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR`

To get the corresponding off-the-shelf optimizer baselines for these experiments, run

```
python run_jobs.py -e base_optimizers --seeds 5 --do_train false  --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR
python run_jobs.py -e base_optimizers_r_main --seeds 5 --do_train false  --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR
```

To get ablations across values of r for the learned optimizer and baselines, run

```python run_jobs.py -e learned_opt_r_ablation --seeds 1 --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR
python run_jobs.py -e base_optimizers_r_ablation --seeds 1 --do_train false  --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR
```

Next we give commands for for ablations across k, the choice of training labels, the choice of evaluation labels, training objective terms, and a comparison to the objective from de Cao (in order):

```
python run_jobs.py -e learned_opt_k_ablation --seeds 1 --dataset ZSRE  --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR
python run_jobs.py -e learned_opt_label_ablation --seeds 1 --dataset ZSRE --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR
python run_jobs.py -e learned_opt_eval_ablation --seeds 1 --dataset ZSRE  --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR
python run_jobs.py -e learned_opt_objective_ablation --seeds 1 --dataset all  --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR
python run_jobs.py -e learned_opt_de_cao --seeds 5 --dataset all --data_dir $DATA_DIR --save_dir $DATA_DIR --cache_dir $DATA_DIR
```

## Analysis

### Statistical Tests

After running an experiment from above, you can compute confidence intervals and hypothesis tests using `statistical_analysis.py`.

To get confidence intervals for the main single-update learned optimizer experiments, run

`python statistical_analysis -e learned_opt_main -n 10000`

To run hypothesis tests between statistics for the learned opt experiment and its baselines, run

`python statistical_analysis -e learned_opt_main -n 10000 --hypothesis_tests true`

You can substitute the experiment name for results for other conditions.

### Belief Graphs

Add `--save_dir`, `--cache_dir`, and `--data_dir` arguments to the commands below per the instructions above.

Write preds from FEVER model:  
`python main.py --dataset FEVER --probing_style model --probe linear --model roberta-base --seed 0 --do_train false --do_eval true --write_preds_to_file true`

Write graph to file:  
`python main.py --dataset FEVER --probing_style model --probe linear --model roberta-base --seed 0 --do_train false --do_eval true --test_batch_size 64 --update_eval_truthfully false --fit_to_alt_labels true --update_beliefs true --optimizer adamw --lr 1e-6 --update_steps 100 --update_all_points true --write_graph_to_file true --use_dev_not_test false --num_random_other 10444`

Analyze graph:  
`python main.py --dataset FEVER --probing_style model --probe linear --model roberta-base --seed 0 --test_batch_size 64 --update_eval_truthfully false --fit_to_alt_labels true --update_beliefs true --use_dev_not_test false --optimizer adamw --lr 1e-6 --update_steps 100 --do_train false --do_eval false --pre_eval false --do_graph_analysis true`

Combine LeapOfThought Main Inputs and Entailed Data:  
`python data_utils/combine_leapofthought_data.py`

Write LeapOfThought preds to file:  
`python main.py --dataset LeapOfThought --probing_style model --probe linear --model roberta-base --seed 0 --do_train false --do_eval true --write_preds_to_file true --leapofthought_main main`

Write graph for LeapOfThought:  
`python main.py --dataset LeapOfThought --leapofthought_main main --probing_style model --probe linear --model roberta-base --seed 0 --do_train false --do_eval true --test_batch_size 64 --update_eval_truthfully false --fit_to_alt_labels true --update_beliefs true --optimizer sgd --update_steps 100 --lr 1e-2 --update_all_points true --write_graph_to_file true --use_dev_not_test false --num_random_other 8642`

Analyze graph (add `--num_eval_points 2000` to compute update-transitivity):  
`python main.py --dataset LeapOfThought --leapofthought_main main --probing_style model --probe linear --model roberta-base --seed 0 --do_train false --do_eval true --test_batch_size 64 --update_eval_truthfully false --fit_to_alt_labels true --update_beliefs true --optimizer sgd --update_steps 100 --lr 1e-2 --do_train false --do_eval false --pre_eval false --do_graph_analysis true`

### Plots

The `data_analysis.Rmd` R markdown file contains code for plots in the paper. It reads data from `aggregated_results` and saves plots in a `./figures` directory.
