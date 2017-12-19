# cb-dr
don't read clickbait

a clickbait classifier inspired by the [Clickbait Challenge 2017](http://www.clickbait-challenge.org/)
implemented in `sklearn` and `pytorch`


## Models
-   Logistic Regression
-   SVM
-   RNN (GRU)
-   CNN
-   Parallel Convolutional Networks

## How to run
`python main.py -h` or `python sk_main.py -h` to see particular options.
Weights were too large to save in github, but our results by iteration
can be found in `/checkpoints`


## Directory Contents
-   `checkpoints`: where session checkpoints are stored
-   `models`: a local package holding all the models built.  Models are exported out of `models/__init__.py` into a dict for argument parsing
-   `data.py`: defines data loading
-   `download_datasets.sh`: download scripts to fetch the datasets
-   `eval.py`: the Clickbait Challenge's official evaluation script, modified to fit right into the training routine
-   `main.py`: defines argument parsing, training and eval routines
-   `sk_main.py`: slightly different routine for running `scikit-learn` models
-   `util.py`: extra utilites such as progress bars, file writing, etc
