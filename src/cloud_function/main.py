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
    indices = tokenizer.encode_plus(
        text,
        max_length=64,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        truncation=True,
    )
    return indices["input_ids"], indices["attention_mask"]


def predict(request):
    request_json = request.get_json()
    tweet = request_json["tweet"]
    tweet = clean_tweet(tweet)
    tweet, mask = encode(tweet)
    tweet = torch.IntTensor(tweet)
    mask = torch.LongTensor(mask)
    tweet.unsqueeze_(0)
    mask.unsqueeze_(0)
    (pred,) = model(tweet, mask)
    pred = torch.argmax(pred, 1).item()
    if pred == 0:
        answer = "This is not a disaster tweet"
        meme = "https://pyxis.nymag.com/v1/imgs/9ef/336/775d89db9c8ffcd8589f3acdf37d0e323f-25-this-is-fine-lede-new.2x.rhorizontal.w700.jpg"  # noqa: E501
    elif pred == 1:
        answer = "This is a disaster tweet"
        meme = "https://memegenerator.net/img/instances/57036215/this-is-a-fucking-disaster.jpg"  # noqa: E501
    else:
        answer = "error"
        meme = "error"

    resp = {"prediction": pred, "answer": answer, "picture": meme}
    return resp


print("init done")
