FROM ubuntu

FROM python:3.10.0-slim

ENV VIRTUAL_ENV "/venv"
RUN python -m venv $VIRTUAL_ENV
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

COPY . .
RUN pip install -r requirements.txt
RUN pip install .
#RUN py -m pip install --upgrade pip
#RUN pip install sigmaepsilon.math
