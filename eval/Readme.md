# Run Evaluations First

Before regenerating the plots, please ensure you have run the evaluation cpp applications to generate the necessary text files containing the evaluation results. Follow these steps:

## For reconstruction evaluations:

```commandline
chmod +x run_test_*.sh
./run_test_reconstruction.sh
```

## For fft vs alpha=1 evaluations:

```commandline
chmod +x run_test_*.sh
./run_test_fft_comparison.sh
```

## For homomorphism evaluations:

```commandline
chmod +x run_test_*.sh
./frft_test_homomorphic.sh
```
    
    

## Generate Plots

Move the generated text files to a directory (or put them in the same directory as the plotting scripts - i.e. /eval/)

Then run the plotting scripts (e.g. plot_results.py) to generate the plots from the text files.

```terminal
cd eval/
python plot_results.py -d <directory_with_text_files>
```

e.g.

```terminal
cd eval/
python plot_results.py -d ./Arm64_M4Pro

or 

python plot_results.py -d ./AMDRyzen9_5900X_12Core_3.7Ghz

```

This will create the plots in the specified directory.