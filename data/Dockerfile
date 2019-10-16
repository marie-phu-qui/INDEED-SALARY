FROM python:3.7-alpine

COPY ./Scrapping ./scrappeur

WORKDIR /scrappeur

RUN apk add --update curl gcc g++ \
    && rm -rf /var/cache/apk/*

RUN pip install numpy cython pandas

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD python app.py