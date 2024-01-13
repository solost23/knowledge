FROM python:3.11

LABEL version="1.0.0"

WORKDIR /app
COPY . .
RUN pip install pipenv && \
    pipenv install --system

CMD ["python", "./main.py"]
