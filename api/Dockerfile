FROM python:3.8.3

# make directories suited to your application
RUN mkdir -p /home/project/app
RUN mkdir -p /home/project/app/models
WORKDIR /home/project/app

# copy and install packages for flask
COPY ./requirements.txt /home/project/app
RUN pip install --no-cache-dir -r requirements.txt

# copy contents from your local to your docker container
COPY . /home/project/app

COPY ./models /home/project/app/models
