# USAGE

## Requirements

- conda 4.10.3

## Install environment

conda env create -f requirements.yml

conda activate CS5604-proj

## Update environment

conda env export > requirements.yml

## Running
The following bash scripts should allow you to test some of the system functionality.
These should be run on the largemem_q ARC partition because of memory requirements.
These will output a "slurm-*.out" file with the results

### Example
Run the following to get a brief example of the transformer prediction process.
```
./run_example.sh
```

### Test
Run the following to get prediction metrics for the test set.
```
./run_test_set.sh
```

### Validation
Run the following to get validation metrics on epochs 1 - 27.
```
./run_validation.sh
```

### Training
Run the following to continue training the transformer (5 additional epochs. ~3hr/epoch)
```
./run_training.sh
```