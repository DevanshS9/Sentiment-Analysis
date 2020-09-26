
## usage
`imdbReviews.py` generates `*.pkl` files which are the training and testing datasets.
First, set the dataset directory in the `imdbReviews.py`, then run the code.
```bash
python imdbReviews.py
```

You will get two `*.pkl` files which are needed for `naive.py` and `svm.py`.
To do prediction, run the following command.
```bash
python naive.py
```