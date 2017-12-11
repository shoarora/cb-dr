# cb-dr
don't read clickbait

a clickbait classifier inspired by the [Clickbait Challenge 2017] (http://www.clickbait-challenge.org/)
implemented in `sklearn` and `pytorch`



## ideas
-   balanced datasets [sun]
-   classify instead
-   glove dims
-   round 0-1 inputs [sun]
-   qrnn
-   cnn parameters [jeff]
-   weight decay




## TODOs
-  improve tokenization
-  use 2d weights https://nbviewer.jupyter.org/github/DSKSD/DeepNLP-models-Pytorch/blob/master/notebooks/04.Window-Classifier-for-NER.ipynb
-  get rnn working
-  get cnn working
-  learn on post titles

## Models
-   [done] Logistic Regression
-   [done] SVM
-   [done] Vanilla NN
-   RNN (pytorch rnn, gru, lstm, etc should be interchangeable)
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
