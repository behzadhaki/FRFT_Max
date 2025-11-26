1. To regenerate the plots, first run the evaluation cpp application called (frft_test_reconstruction)

2. Move the generated text files to a directory (or put them in the same directory as the plotting scripts - i.e. /eval/)

3. Run the plotting scripts (e.g. plot_results.py) to generate the plots from the text files.

```terminal
cd eval/
python plot_results.py -d <directory_with_text_files>
```
e.g.

```terminal
cd eval/
python plot_results.py -d ./Arm64_M4Pro
```
