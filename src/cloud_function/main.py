import io

import torch
from google.cloud import storage
from transformers import AutoTokenizer

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


def encode(text: str) -> list[int]:
    tokens = tokenizer.encode(text)
    tokens = tokens[: 140 - 2]
    pad_len = 140 - len(tokens)
    tokens += [0] * pad_len
    return tokens


def predict(request):
    request_json = request.get_json()
    tweet = request_json["tweet"]
    tweet = encode(tweet)
    tweet = torch.IntTensor(tweet)
    tweet.unsqueeze_(0)
    (pred,) = model(tweet)
    pred = torch.argmax(pred, 1).item()
    if pred == 0:
        answer = "This is not a desaster tweet"
    elif pred == 1:
        answer = "This is a desaster tweet"
    else:
        answer = "error"
    return answer


print("init done")
