FROM python:3.9-buster
RUN apt-get update -y
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]