from kafka import KafkaProducer
from time import sleep
from json import dumps
import json

from flask import Flask, render_template, request, jsonify, make_response

app = Flask(__name__)

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda m: 
                         json.dumps(m).encode('ascii'))

for f in range(100):
    data = {'number' : f}
    producer.send('json-topic', {'key': data})
    sleep(5)