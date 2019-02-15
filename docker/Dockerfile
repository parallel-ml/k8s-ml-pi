FROM jiashenc/arm-keras
WORKDIR /server
COPY . /server
RUN pip install avro
EXPOSE 8080

CMD python -m service.generic_server alexnet.demo