# Predict bbox model
This is a server build with **sanic**, which is a framework similar to flask, but with asynchronous functions. 

The syntax is very similar to flask, but the deployment is different.

## Run in docker

(use sudo if needed)

```$ docker build -t extractor .```

```$ docker run --name extractor -p 8001:8000 extractor```


## Routes and parameters:
- **GET /train-model**

  *parameters:*
  - field - name of the field to train model on (example "cod_fiscal_furnizor", "client" etc)
  - k_neighbors = nr of neighbors to consider when creating the dataset
  - model_name - (optional) name of the model to be saved in "<model_name>.pkl" file (optional), default model.pkl
  - model_path = (optional) path where to save model - default 'models/'
  

- **POST /predict**

  *parameters:*
   - ocr_data = json ocr data from invoice
   - k_neighbors = nr of neighbors to consider (should be same as in train)
   - field - name of the field (example "cod_fiscal_furnizor", "client" etc)
   - model_name - (optional) name of the model to be saved in "<model_name>.pkl" file (optional), default model.pkl
   - model_path = (optional) path where to save model - default 'models/'
    
## Deployment on heroku

Because it is developed in sanic, for deployment it uses **asgi** and **uvicorn** as a difference to Flask that uses *wsgi* and *gunicorn*.

The configuration is done with 'asgi.py' file and command "CMD uvicorn asgi:app --host 0.0.0.0 --port $PORT " from Dockerfile.

So the steps needed for heroku are:
1. ```$ heroku container:login```

2. **optional step, only if the app doesn't exist!!!**
```$ heroku create name-of-the-app```

3. **(optional step) if app is created from web:**
```$ heroku git:remote -a name-of-the-app```

4. ```$ heroku container:push web  -a predict-app-box```

5. ```$ heroku container:release web```


If app is already created and deployed on heroku, to update:
1. ```$ heroku container:push web```
2. ```$ heroku container:release web```


To view logs on heroku:
``` $heroku logs --tail```

https://devcenter.heroku.com/articles/container-registry-and-runtime#getting-started
