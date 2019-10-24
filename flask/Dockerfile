FROM python:3.7-alpine

COPY ./app ./app

WORKDIR /app

RUN apk add --update curl gcc g++ \
    && apk add python3-dev build-base linux-headers pcre-dev \
    && rm -rf /var/cache/apk/* \
    apk add zlib libjpeg-turbo-dev libpng-dev freetype-dev lcms2-dev libwebp-dev harfbuzz-dev fribidi-dev tcl-dev tk-dev

RUN pip install --upgrade pip

RUN pip install uwsgi

RUN pip install -r requirements.txt

CMD python .app/app.py