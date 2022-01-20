import io

import nltk
import torch
from google.cloud import storage
from transformers import AutoTokenizer
from tweet_cleaner import clean_tweet

BUCKET_NAME = "cloud_function_models"
MODEL_FILE = "deployable_model.pt"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = blob.download_as_string()
my_model = io.BytesIO(my_model)

model = torch.jit.load(my_model)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

nltk.download("wordnet")
nltk.download("omw-1.4")


def encode(text: str) -> list[int]:
    tokens = tokenizer.encode(text)
    tokens = tokens[: 140 - 2]
    pad_len = 140 - len(tokens)
    tokens += [0] * pad_len
    return tokens


def predict(request):
    request_json = request.get_json()
    tweet = request_json["tweet"]
    tweet = clean_tweet(tweet)
    tweet = encode(tweet)
    tweet = torch.IntTensor(tweet)
    tweet.unsqueeze_(0)
    (pred,) = model(tweet)
    pred = torch.argmax(pred, 1).item()
    if pred == 0:
        answer = "This is not a desaster tweet"
        meme = "https://pyxis.nymag.com/v1/imgs/9ef/336/775d89db9c8ffcd8589f3acdf37d0e323f-25-this-is-fine-lede-new.2x.rhorizontal.w700.jpg"  # noqa: E501
    elif pred == 1:
        answer = "This is a desaster tweet"
        meme = "https://wwwcache.wral.com/asset/news/local/2021/02/12/19524364/viral-raleigh-snow-glenwood-meme-DMID1-5putzq7om-640x360.jpg"  # noqa: E501
    else:
        answer = "error"
        meme = "error"

    resp = {"prediction": pred, "answer": answer, "picutre": meme}
    return resp


print("init done")
