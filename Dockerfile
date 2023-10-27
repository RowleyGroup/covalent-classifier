FROM python:3.10.12

WORKDIR /src

COPY requirements.txt .
COPY data/ /src/data
COPY saved_models/ /src/saved_models
COPY models/ /src/models/

RUN pip install --no-cache-dir -r requirements.txt