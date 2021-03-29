'''
comsumer to get json-topic, and compute the angle, send the angle as 'angle-topic' to kafka

Consumer 1 will process the incoming stream (this is creating a new stream > see point 4, new producer)

Design stream processing on the incoming key points (continues queries), 
take 2 evaluation criteria e.g. angle and angular speed of specific limb. 
'''

from kafka import KafkaConsumer
import json
from datetime import datetime
import math
from kafka import KafkaProducer

# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer('json-topic',
                         bootstrap_servers=['localhost:9092'],
                         auto_offset_reset='latest')

producer_angle = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda m: 
                         json.dumps(m).encode('ascii'))

for message in consumer:  
    # read from json-topic
    whole_message = message.value.decode('ascii')
    dict_message = json.loads(whole_message)
    # time stamp is the milliseconds from January 1, 1970, 00:00:00 UTC
    time_stamp = dict_message['excercise']['timestamp'] 
    time_stamp2 = dict_message['time'][11:19]
    minutes = time_stamp2[-5:-3]
    seconds = time_stamp2[-2:]  
    keypoints = dict_message['excercise']['keypoints']
    leftShoulder = (keypoints[5]['position']['x'],keypoints[5]['position']['y'])
    leftElbow = (keypoints[7]['position']['x'],keypoints[7]['position']['y'])
    
    # computes the angle
    vector_1 = (leftElbow[0] - leftShoulder[0], leftElbow[1] - leftShoulder[1]) # vector point from leftShoulder to leftElbow
    vector_2 = (0, 1) # straight down (not up)
    # angel from vertical down to vector of left arm
    angle = (math.atan2(vector_2[1], vector_2[0]) - math.atan2(vector_1[1], vector_1[0])) / math.pi * 180.0    
    # below an alternative way to calculate angle, yields the same result
    # x = leftElbow[0] - leftShoulder[0]
    # y = leftElbow[1] - leftShoulder[1]
    # angle2 = np.arctan(x/y) * (180 / math.pi)
    # if angle2 < 0:
    #    angle2 = angle2 + 90 + 90 
     
    # sending angle with time_stamp to "angle-topic"
    producer_angle.send('angle-topic', {'mil_sec': time_stamp, 'angle': angle, 'min': minutes, 'sec': seconds}) # sent data like this: {"mil_sec": "xxxxxxxx", "angle": 19.947940899363747, "min": minutes, "sec": seconds}
    
    
    # print('leftShoulder', leftShoulder)
    # print('leftElbow', leftElbow)
    print('left arm angle', angle, 'degrees')
    print('timestamp', time_stamp)
    print()