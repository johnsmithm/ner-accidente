import json
# import joblib
import os
from os import path
from sanic import Sanic
from sanic import response 
from predicter import predict_ent

app = Sanic("App Name")

from sanic_cors import CORS, cross_origin

# app = Sanic(__name__)
CORS(app) 
 
@app.route("/")
async def test(request):
    return response.json({"hello": "world"})


@app.route('/extract', methods=['GET'])
async def extract(request):
    #if key doesn't exist, returns None
    text_to_extract_from = request.args.get('text_to_extract_from')

    prediction = predict_ent(text_to_extract_from)

    result = 'Found these enteties: {}'.format(prediction)
    return response.json(result)

@app.route("/evaluate-model/<name>")
async def eval_model(request):
    return response.json({"TODO:": ""})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)