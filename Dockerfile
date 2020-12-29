FROM python:3.8
WORKDIR /app

EXPOSE 8000

COPY ./deployment_sanic/requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update

# copy just what you need
COPY ./deployment_sanic/* /app/
COPY ./src/models /models

CMD uvicorn asgi:app --host 0.0.0.0 --port 8000 
