from kafka import KafkaConsumer
import json
from datetime import datetime

# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer('json-topic',
                         bootstrap_servers=['localhost:9092'],
                         auto_offset_reset='latest') # earliest or latest

for message in consumer:
 
      
    whole_message = message.value.decode('ascii')
    dict_message = json.loads(whole_message)
    # time_stamp = dict_message['time'][11:]
    # print(time_stamp)
    # keypoints is a list
    keypoints = dict_message['excercise']['keypoints']
    time_stamp = dict_message['excercise']['timestamp']
    nose_position_x = keypoints[0]['position']['x']
    nose_position_y = keypoints[0]['position']['y']
    print('nose_x', nose_position_x)
    print('nose_y', nose_position_y)
    print('timestamp', time_stamp)
    print('  ')
    
        # leftEye = keypoints[1]['part']
        # leftEye_position_x = keypoints[1]['position']['x']
        # leftEye_position_y = keypoints[1]['position']['y']
        # print('leftEye_x', leftEye_position_x)
        # print('leftEye_y', leftEye_position_y)
        # print('  ')

    # print (message.value.decode('ascii'))


