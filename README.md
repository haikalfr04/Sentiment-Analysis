# Movie Review Sentiment Analysis with Transformers

Binary sentiment classification (positive / negative) of IMDB movie reviews, comparing a classical
TF-IDF baseline with fine-tuned pretrained transformers (DistilBERT, RoBERTa).

The project looks past "fine-tune BERT, report accuracy" at three questions:

1. **Is a transformer worth it?** Accuracy against parameter count, training time and inference speed, measured against a strong classical baseline.
2. **Which part of a long review matters?** Many IMDB reviews are longer than the model's token limit. The project compares keeping only the beginning (*head*) with keeping the beginning and the end (*head+tail*, Sun et al., 2019).
3. **Where does the model still fail?** Error analysis of its most confident mistakes: sarcasm, mixed opinions, negation.

> **Demo:** _add the Hugging Face Spaces link here_ · **Model:** _add the Hugging Face Hub link here_

## Results

IMDB test set, 25,000 reviews. Run `python -m src.evaluate` to generate `results/comparison.md`, then paste the table here.

| Model | Max tokens | Truncation | Accuracy | F1 | Params | Train time (T4) |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | - | - | _TBD_ | _TBD_ | - | _TBD_ |
| DistilBERT | 256 | head | _TBD_ | _TBD_ | 66M | _TBD_ |
| DistilBERT | 256 | head+tail | _TBD_ | _TBD_ | 66M | _TBD_ |
| RoBERTa-base | 512 | head+tail | _TBD_ | _TBD_ | 125M | _TBD_ |

<!-- After running, add: results/figures/length_distribution.png and a confusion matrix -->

### Key findings

_Write 3–5 bullets after running the experiments, for example:_
- _How many accuracy points the transformers gain over TF-IDF, and at what cost in speed._
- _Whether head+tail beats head truncation, and why (verdicts often sit at the end of a review)._
- _Which error patterns remain (see `results/errors/`)._

## Approach

**Data.** [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) has 25k train and 25k test reviews, balanced 50/50.
A stratified 10% of the training set is held out for validation and model selection, so the test set is only
used for the final numbers. `<br />` tags are stripped.

**Baseline.** TF-IDF on word uni- and bigrams with sublinear TF, plus logistic regression. The regularization
strength `C` is tuned on the validation split.

**Transformers.** Fine-tuned with the Hugging Face `Trainer`: AdamW, learning rate 2e-5, linear schedule with
10% warmup, 2 epochs, mixed precision. The checkpoint with the best validation F1 is kept.

**Truncation.** With *head+tail*, a review that exceeds the token budget keeps its first 25% and last 75% of
tokens, as recommended by Sun et al. (2019). The setting is saved with the model (`inference_config.json`) so
prediction tokenizes the same way as training.

## Project structure

```
├── src/
│   ├── data.py        # loading, cleaning, head/head+tail truncation, tokenization
│   ├── eda.py         # class balance, review length distribution
│   ├── baseline.py    # TF-IDF + Logistic Regression
│   ├── train.py       # transformer fine-tuning (any Hugging Face model)
│   ├── evaluate.py    # comparison table, confusion matrices, error reports
│   ├── predict.py     # inference CLI / SentimentPredictor class
│   └── metrics.py     # shared metrics and result files
├── app/app.py         # Gradio demo
├── notebooks/imdb_sentiment_colab.ipynb   # runs the whole pipeline on Colab
├── tests/             # unit tests for cleaning and truncation
└── results/           # metrics JSON, figures, error reports (committed)
```

## Quickstart

The easiest way is to open `notebooks/imdb_sentiment_colab.ipynb` in Google Colab with a T4 GPU. Locally:

```bash
pip install -r requirements.txt

python -m src.eda                                   # data analysis
python -m src.baseline                              # TF-IDF baseline
python -m src.train --model_name distilbert-base-uncased --max_length 256 --truncation head
python -m src.train --model_name distilbert-base-uncased --max_length 256 --truncation head_tail
python -m src.train --model_name roberta-base --max_length 512 --truncation head_tail --batch_size 8 --grad_accum 2
python -m src.evaluate                              # comparison table + error analysis

python -m src.predict --model_dir models/distilbert-base-uncased-256-head "What a fantastic film!"
MODEL_DIR=models/distilbert-base-uncased-256-head python app/app.py
```

For a quick check that everything runs, add `--max_train_samples 500 --max_eval_samples 200 --epochs 1`.
`python -m src.train --help` lists every option, including `--report_to wandb` for experiment tracking and
`--hub_model_id` to push the model to the Hugging Face Hub.

Run the tests with `pytest`.

## Deploying the demo

1. Push the trained model to the Hub (last section of the notebook, or `--hub_model_id`).
2. Create a Gradio Space on Hugging Face and upload `app/app.py` as `app.py`, along with the `src/` folder and `requirements.txt`.
3. In the Space settings, set the `MODEL_DIR` variable to your Hub repo id, for example `username/imdb-sentiment-roberta`.

## References

- Devlin et al., 2019. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Liu et al., 2019. *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*
- Sanh et al., 2019. *DistilBERT, a distilled version of BERT.*
- Sun et al., 2019. *How to Fine-Tune BERT for Text Classification?*
- Maas et al., 2011. *Learning Word Vectors for Sentiment Analysis* (IMDB dataset).
