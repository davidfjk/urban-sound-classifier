FROM python:3.7.12-slim

RUN PYTHONUNBUFFERED=TRUE

WORKDIR /app

COPY . /app

RUN apt-get update
RUN apt-get install libsndfile1-dev --yes
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 9696

#ENTRYPOINT

CMD ["python3", "app.py"]