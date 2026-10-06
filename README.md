# Movie Review Sentiment Analysis with Transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haikalfr04/Sentiment-Analysis/blob/main/notebooks/imdb_sentiment_colab.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformers-orange)

This project classifies IMDB movie reviews as positive or negative. It compares a traditional machine learning
model (TF-IDF with logistic regression) with pretrained language models (DistilBERT and RoBERTa) that were
fine-tuned on the same data.

**Best result: RoBERTa-base reached 95.67% accuracy on 25,000 test reviews and made 55% fewer errors than the traditional model.**

The project aims to answer three questions:

1. **Is a pretrained language model worth the extra cost?** The models are compared on accuracy, model size, training time and prediction speed.
2. **Which part of a long review should the model read?** The models can only read a limited number of tokens (words and word pieces), and 41% of the reviews are longer than 256 tokens. Two methods for shortening long reviews are compared: keeping only the beginning, and keeping both the beginning and the end (Sun et al., 2019).
3. **What kinds of reviews does the best model still get wrong?** The model's most confident mistakes are examined by hand.

<!-- Add once deployed: **Demo:** <Hugging Face Spaces link> · **Model:** <Hugging Face Hub link> -->

## Results

![Test error rate per model](results/figures/model_comparison.png)

| Model | Max tokens | Shortening method | Accuracy | F1 score | Errors (of 25,000) | Parameters | Training time* | Reviews per second* |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | - | - | 90.35% | 90.37% | 2,412 | 0.2M | 13 s | 6,402 |
| DistilBERT | 256 | Beginning only | 91.55% | 91.56% | 2,113 | 67M | 45 s | 3,677 |
| DistilBERT | 256 | Beginning + end | 92.97% | 92.99% | 1,758 | 67M | 44 s | 3,672 |
| **RoBERTa-base** | **512** | **Beginning + end** | **95.67%** | **95.68%** | **1,082** | **125M** | **150 s** | **1,210** |

<sub>*DistilBERT and RoBERTa were trained for 2 epochs on an NVIDIA RTX PRO 6000 (Blackwell) GPU. The TF-IDF model was trained and tested on a CPU. Each setup was trained once, with random seed 42.</sub>

### Key findings

- **Keeping the end of long reviews improves accuracy at no extra cost.** With the same model, the same token limit and the same training time, keeping both the beginning and the end of each long review increased DistilBERT's accuracy from 91.55% to 92.97% and reduced errors by 17% (from 2,113 to 1,758). Reviewers often give their final opinion in the last sentences, and this part is lost when only the beginning is kept.
- **The traditional model is a strong baseline.** TF-IDF with logistic regression reached 90.35% accuracy, only 1.2 percentage points below DistilBERT when DistilBERT read only the beginning of each review. It is also much smaller and faster. The pretrained models show a clear advantage only when they can read enough of each review.
- **RoBERTa with 512 tokens performed best.** It reached 95.67% accuracy and made 55% fewer errors than the traditional model. In exchange, it has almost twice as many parameters as DistilBERT and predicts about three times more slowly. Its errors are evenly split between the two classes (560 negative reviews predicted as positive, 522 positive reviews predicted as negative), so it does not favor either class.
- **Some of the remaining errors are caused by the data, not by the model.** Several of RoBERTa's most confident mistakes are sarcastic reviews or reviews whose text does not match their label (see the error analysis below). This limits the highest accuracy that any model can reach on this dataset.

### Review length

![Review length distribution](results/figures/length_distribution.png)

A typical review (the median) is 222 tokens long, and the average is 298 tokens. A small number of reviews are much
longer, up to 3,047 tokens. 41.3% of the training reviews are longer than 256 tokens, and 13.4% are longer than 512
tokens. Positive and negative reviews have almost the same length, so length alone does not indicate the sentiment.
The full statistics are in [`results/eda_stats.json`](results/eda_stats.json).

### Error analysis

The reviews that RoBERTa misclassified with about 99.9% confidence fall into four groups. The full lists are in
[`results/errors/`](results/errors/).

| Type of review | Example (shortened) | True label → prediction |
|---|---|---|
| **Sarcasm** | "This has to be one of the all time greatest horror movies … its not hard to see why Band went on to make such classics as 'Killjoy 2: Deliverance From Evil'" | negative → positive |
| **Enjoyed because it is bad** | "it had to be one of the worst movies ever. Funny, in a real bad way." | positive → negative |
| **Mixed opinion** | "Do I say how great the cinematography is … the film is boring it moves so slow, that watching paint dry would be an improvement." | positive → negative |
| **Text does not match the label** | "I found 'Strange Fruit' to be an excellent movie … the acting (…) is wonderful, the cinematography and direction excellent" | negative → positive |

The IMDB labels are based on the star rating that each reviewer gave (4 stars or fewer is negative, 7 stars or more is
positive), not on the text itself. When the rating and the text disagree, the model follows the text. For a model whose
purpose is to read the opinion in a text, this is reasonable behavior.

The confusion matrices below show how many reviews of each class were classified correctly and incorrectly, for the
traditional model (left) and for RoBERTa (right).

<p align="center">
  <img src="results/figures/tfidf-logreg_confusion.png" width="45%" alt="TF-IDF confusion matrix">
  <img src="results/figures/roberta-base-512-head_tail_confusion.png" width="45%" alt="RoBERTa confusion matrix">
</p>

## Method

**Data.** The [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) contains 25,000 training reviews and
25,000 test reviews, with equal numbers of positive and negative reviews. 10% of the training reviews (2,500) were set
aside as a validation set, with the same balance of positive and negative reviews. The validation set was used to choose
the best settings, and the test set was used only once, for the final results. HTML line breaks (`<br />`) were removed
from the text.

**Traditional model.** Each review is converted into word and word-pair frequencies weighted by TF-IDF (up to 200,000
features), and a logistic regression model is trained on these features. Several regularization strengths were tested
on the validation set, and the best one (C = 4) was used.

**Pretrained language models.** DistilBERT and RoBERTa were fine-tuned with the Hugging Face `Trainer` for 2 epochs,
with a learning rate of 2e-5. The learning rate increases gradually during the first 10% of training and then decreases
linearly. After each epoch, the model was tested on the validation set, and the version with the best F1 score was kept.

**Shortening long reviews.** When a review is longer than the token limit, the "beginning + end" method keeps the first
25% and the last 75% of the allowed tokens, as recommended by Sun et al. (2019). This setting is saved together with the
model (`inference_config.json`), so new reviews are processed in the same way during prediction as during training.

## Limitations and future work

- The RoBERTa experiment uses both a different model and a higher token limit than DistilBERT, so it is not possible to tell how much of the improvement comes from each change. Training RoBERTa with 512 tokens using only the beginning of each review would answer this.
- Each setup was trained only once. Training several times with different random seeds would show whether the smaller differences between models are reliable.
- Explanation methods such as SHAP could show which words lead the model to its mistakes on sarcastic and mixed reviews.
- Knowledge distillation (training a smaller model to copy a larger one) could give DistilBERT part of RoBERTa's accuracy while keeping DistilBERT's speed.

## Project structure

```
├── src/
│   ├── data.py        # loads and cleans the data, shortens long reviews, converts text to tokens
│   ├── eda.py         # data exploration: class balance and review length
│   ├── baseline.py    # traditional model: TF-IDF + logistic regression
│   ├── train.py       # fine-tunes a pretrained language model
│   ├── evaluate.py    # comparison table and chart, confusion matrices, error reports
│   ├── predict.py     # predicts the sentiment of new reviews
│   └── metrics.py     # shared evaluation metrics and result files
├── app/app.py         # web demo built with Gradio
├── notebooks/imdb_sentiment_colab.ipynb   # runs the full project on Google Colab
├── tests/             # unit tests for text cleaning and review shortening
└── results/           # metrics, charts and error reports
```

## How to run

The easiest way is to click the **Open in Colab** button at the top of this page, select a GPU in
**Runtime → Change runtime type**, and then choose **Runtime → Run all**.

To run the project on your own computer:

```bash
pip install -r requirements.txt

python -m src.eda                                   # explore the data
python -m src.baseline                              # train the traditional model
python -m src.train --model_name distilbert-base-uncased --max_length 256 --truncation head
python -m src.train --model_name distilbert-base-uncased --max_length 256 --truncation head_tail
python -m src.train --model_name roberta-base --max_length 512 --truncation head_tail --batch_size 8 --grad_accum 2
python -m src.evaluate                              # compare models and analyze errors

python -m src.predict --model_dir models/roberta-base-512-head_tail "What a fantastic film!"
MODEL_DIR=models/roberta-base-512-head_tail python app/app.py
```

In the commands, `head` means keeping only the beginning of each review, and `head_tail` means keeping the beginning
and the end.

To quickly check that everything works before a full run, add `--max_train_samples 500 --max_eval_samples 200 --epochs 1`
to the training command. Run `python -m src.train --help` to see all options, including `--report_to wandb` for
tracking experiments and `--hub_model_id` for uploading the model to the Hugging Face Hub.

To run the unit tests, use `pytest`.

## Publishing the demo

1. Upload the trained model to the Hugging Face Hub (see the last section of the notebook, or use `--hub_model_id`).
2. Create a new Gradio Space on Hugging Face. Upload `app/app.py` as `app.py`, together with the `src/` folder and `requirements.txt`.
3. In the Space settings, set the `MODEL_DIR` variable to the name of your model on the Hub, for example `username/imdb-sentiment-roberta`.

## References

- Devlin et al., 2019. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Liu et al., 2019. *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*
- Sanh et al., 2019. *DistilBERT, a distilled version of BERT.*
- Sun et al., 2019. *How to Fine-Tune BERT for Text Classification?*
- Maas et al., 2011. *Learning Word Vectors for Sentiment Analysis* (IMDB dataset).
