FROM gcr.io/tensorflow/tensorflow

MAINTAINER Aaron J. Masino

# UPDATE apt-get INSTALL vim

RUN apt-get update &&  apt-get install -y --no-install-recommends \
        vim \
    && apt-get clean

RUN yes | pip install gensim

RUN yes | pip install -U scikit-learn 

RUN yes | pip install pandas
