# First stage: build environment
FROM python:3.8-slim as build

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Second stage: runtime environment
FROM python:3.8-slim

WORKDIR /app

COPY --from=build /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY . .

EXPOSE 8080

CMD ["python", "predict.py"]
