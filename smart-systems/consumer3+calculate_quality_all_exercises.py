# comsumer 3
from kafka import KafkaConsumer
import json
from datetime import datetime
import math
from kafka import KafkaProducer
import datetime

negativeQuality1 = 0
positiveQuality1 = 0
qualityRatio1 = ""
qualityRatio2 = ""
qualityRatio3 = ""
startTime = 0
state = 0
interval = 0
filename = "C:/Users/Samir/Documents/smart-systems-lab-4-and-5-group-3-mei-tao-samir-kenny-master/LOGFILE.txt"
qualityString = ""
qualityString1 = ""
qualityString2 = ""
finalString = ""

# To consume latest messages and auto-commit offsets
# topic is "angle-topic (with time-stamp)"
consumer = KafkaConsumer('quality-topic',
                         bootstrap_servers=['localhost:9092'],
                         auto_offset_reset='latest')  # earliest or latest
producer_Quality_ratios = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda m: 
                         json.dumps(m).encode('ascii'))
    
def calcQualityRatio():
    global negativeQuality1
    global positiveQuality1
    global qualityRatio1
    global negativeQuality2
    global positiveQuality2
    global qualityRatio2
    global negativeQuality3
    global positiveQuality3
    global qualityRatio3
    global qualityString
    global qualityString1
    global qualityString2
    negquality1 = dict_message['negativeQuality1']
    posquality1 = dict_message['positiveQuality1']
    negquality2 = dict_message['negativeQuality2']
    posquality2 = dict_message['positiveQuality2']
    negquality3 = dict_message['negativeQuality3']
    posquality3 = dict_message['positiveQuality3']
    #print(posquality1)
    #print(negquality1)
    qualityRatio1 = str(posquality1) + ':' + str(negquality1)
    qualityRatio2 = str(posquality2) + ':' + str(negquality2)
    qualityRatio3 = str(negquality2) + ':' + str(posquality3)
    qualityString1 = str(datetime.datetime.now()) + " ex1 quality ratio 1: " + qualityRatio1 + " ex2 quality ratio 2: " 
    qualityString2 = qualityRatio2 + " ex3 quality ratio 3: " + qualityRatio3
    qualityString = qualityString1 + qualityString2
    return qualityString

def stateHandler():
    global state
    global filename
    global qualityString
    global finalString
    hours = 1
    minutes = 1
    seconds = 2
    milliseconds = 1000
    global interval
    # case 0: get start time
    if (state == 0):
        startTime = time_stamp
        interval = startTime + hours * minutes * seconds * milliseconds
        print(interval)
        state = 1
    # case 1: continuously get the ratio and check whether the logging interval has passed
    elif (state == 1):
        finalString = calcQualityRatio()
        if (time_stamp > interval):
            print("time to store data")
            state = 2
    # case 3: store data in file
    elif (state == 2):    
        print(finalString)
        file = open(filename, "a") 
        file.write(finalString)
        file.write("\n")
        file.close()
        print("data is logged!")
        state = 0
   
  
for message in consumer:
    whole_message = message.value.decode('ascii')
    dict_message = json.loads(whole_message)   
    time_stamp = dict_message['mil_sec']
    stateHandler()

# store quality data in .csv file
