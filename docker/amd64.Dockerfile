FROM tensorflow/tensorflow
WORKDIR /server
COPY . /server
RUN pip install avro
RUN pip install keras
EXPOSE 8080