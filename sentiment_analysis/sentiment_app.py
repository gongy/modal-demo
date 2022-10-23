from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

web_app = FastAPI()

@web_app.get("/", response_class=HTMLResponse)
async def slash(request: Request):
    with open("/assets/index.html") as f:
        return HTMLResponse(status_code=200, content=f.read())

@web_app.post("/predict")
async def foo(request: Request):
    x = (await request.body()).decode()
    model = SentimentAnalysis()
    return model.predict(x)

import os
import modal

stub = modal.Stub(
    image=modal.Image.debian_slim().pip_install(["transformers", "torch"])
)

# make all the files in current directory is available on the Modal runner under /assets
my_file_directory = os.path.dirname(__file__)

@stub.asgi(
    mounts=[modal.Mount(local_dir=my_file_directory, remote_dir="/assets")]
)
def fastapi_app():
    return web_app

class SentimentAnalysis:
    def __enter__(self):
        from transformers import pipeline

        self.sentiment_pipeline = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")

    @stub.function(cpu=8, retries=3)
    def predict(self, phrase: str):
        pred = self.sentiment_pipeline(phrase, truncation=True, max_length=512, top_k=2)
        # pred will look like: [{'label': 'NEGATIVE', 'score': 0.99}, {'label': 'POSITIVE', 'score': 0.01}]
        probs = {p["label"]: p["score"] for p in pred}
        return probs["POSITIVE"]

if __name__ == "__main__":
    stub.serve()
