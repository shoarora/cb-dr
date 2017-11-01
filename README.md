# cb-dr
don't read clickbait

a clickbait classifier inspired by the [Clickbait Challenge 2017] (http://www.clickbait-challenge.org/)
implemented in `sklearn` and `pytorch`

## TODOs
-   sklearn harness
-   feature selection
-   figure out what's wrong with the F1 score

## Models
-   Logistic Regression
-   SVM
-   Vanilla NN
-   LSTM
-   CNN
-   Additional features from media

## Directory Contents
-   `checkpoints`: where session checkpoints are stored
-   `models`: a local package holding all the models built.  Models are exported out of `models/__init__.py` into a dict for argument parsing
-   `data.py`: defines data loading
-   `download_datasets.sh`: download scripts to fetch the datasets
-   `eval.py`: the Clickbait Challenge's official evaluation script, modified to fit right into the training routine
-   `main.py`: defines argument parsing, training and eval routines
-   `util.py`: extra utilites such as progress bars, file writing, etc
