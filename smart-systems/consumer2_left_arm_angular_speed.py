from kafka import KafkaConsumer
import json
from datetime import datetime
import math
from kafka import KafkaProducer
from collections import deque
window = deque(maxlen=10)

startExercise = 1
startTimeMinutes = 0
startTimeSeconds = 0
stopTimeMinutes = 0
stopTimeSeconds = 0
finalTime = 0
previousState = -1
goodExecerciseTime = 5
counter = 0
state = 0
negativeQuality1 = 0
positiveQuality1 = 0
negativeQuality2 = 0
positiveQuality2 = 0
negativeQuality3 = 0
positiveQuality3 = 0
finalTimeSec = 0
finalTimeMin = 0
angular_speed = 0
angle = 0


# To consume latest messages and auto-commit offsets
# topic is "angle-topic (with time-stamp)"
consumer = KafkaConsumer('angle-topic',
                         bootstrap_servers=['localhost:9092'],
                         auto_offset_reset='latest')  # earliest or latest
producer_quality = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda m: 
                         json.dumps(m).encode('ascii'))
angular_speed = 0



# Quality of exercise: time. Time to put arm up can be tracked. 
# If value falls within our predefined range, exercise is executed correctly
def checkExerciseLength():
    global state
    global previousState
    global startTimeMinutes
    global startTimeSeconds
    global stopTimeMinutes
    global stopTimeSeconds
    global finalTime
    global goodExecerciseTime 
    global negativeQuality1
    global positiveQuality1
    
    print('neg1:', negativeQuality1)
    print('pos1:', positiveQuality1)
    
    if (state == 0): # process duration of exercise
        if (previousState != state):
            previousState = state
            startTimeMinutes, startTimeSeconds = int(dict_message['min']), int(dict_message['sec'])         
            print('startTimeMinutes: ', startTimeMinutes)
            print('startTimeSeconds: ', startTimeSeconds)
        elif (angle > 85):
            stopTimeMinutes, stopTimeSeconds = int(dict_message['min']), int(dict_message['sec']) 
            print('stopTimeMinutes: ', stopTimeMinutes)
            print('stopTimeSeconds: ', stopTimeSeconds)
            state = 1
    if (state == 1): # calculate duration of exercise      
        if (previousState != state):
            previousState = state
            finalTime = stopTimeSeconds - startTimeSeconds
            if (finalTime < 0):
                finalTime = finalTime + 60
            print('finalTime: ', finalTime)
            if finalTime < goodExecerciseTime:
                print('exercise executed within time, well done!')
                positiveQuality1 += 1
                print('pos', positiveQuality1)
            else:
                print('Work harder!')
                negativeQuality1 += 1
                print('neg', negativeQuality1)
            state = 2
    if (state == 2):  # wait untill arm is low, to start next cycle
        if (angle < 30):
            state = 0
#     #return negativeQuality1, positiveQuality1

# This function checks whether the exercising person has done an exercise a number of times.
# If the exercise is done a predefined x number of times within a predefined x interval, the exercise quality 
# is high. If not, exercise quality is low and message is sent to participant to train harder!
def checkNumberOfExercises():
    global state
    global previousState
    global startTimeMinutes
    global startTimeSeconds
    global stopTimeMinutes
    global stopTimeSeconds
    global finalTimeSec
    global finalTimeMin
    global finalTime
    global previousAngle
    global goodExecerciseTime
    global counter
    global negativeQuality2
    global positiveQuality2
    
    
    print('counter', counter)
    print('angle', angle)
    print('neg2:', negativeQuality2)
    print('pos2:', positiveQuality2)
    
    if (state == 0):
        if (previousState != state):
            previousState = state
            startTimeMinutes, startTimeSeconds = int(dict_message['min']), int(dict_message['sec'])         
            print('startTimeMinutes: ', startTimeMinutes)
            print('startTimeSeconds: ', startTimeSeconds)
            state = 1
    
    if (state == 1):  
        if (previousState != state):
            previousState = state
        if (angle > 85):
            counter += 1
            previousAngle = 1
            state = 2
           
    if (state == 2): 
        if (previousState != state):
            previousState = state
        if (previousAngle == 1):   # detect state change (arm going from up to down again)
            if (angle < 30):
                previousAngle = 0
                state = 3
    
    if (state == 3): 
        if (previousState != state):
            previousState = state
        if (counter == 3):
            counter = 0
            stopTimeMinutes, stopTimeSeconds = int(dict_message['min']), int(dict_message['sec']) 
            print('stopTimeMinutes: ', stopTimeMinutes)
            print('stopTimeSeconds: ', stopTimeSeconds)
            finalTimeMin = (stopTimeMinutes - startTimeMinutes) * 60
            finalTimeSec = stopTimeSeconds - startTimeSeconds        
            if (finalTimeSec < 0):
                finalTime = finalTimeMin - finalTimeSec
                state = 4
            elif (finalTimeSec > 0):
                finalTime = finalTimeMin + finalTimeSec
                state = 4
        else:
            state = 1        
                         
    if (state == 4):   # increase quality count 
        if finalTime < 60:
            print(finalTime)
            print('exercise executed multiple times within time interval, well done!')
            positiveQuality2 += 1
            state = 0
        if finalTime > 60:
            print(finalTime)
            print('Work harder!')
            negativeQuality2 += 1              
            state = 0
          
        
# This function tests how whether the arm is moving fast enough, when going from horizontal to vertical position of the arm
# If speed is fast enough, positive quality score will be increased. If not, negative quality score will be increased
def checkSpeedUpmovement():  
    
    global negativeQuality3
    global positiveQuality3
    global angular_speed
    print('neg3:', negativeQuality3)
    print('pos3:', positiveQuality3)
    
    window.append(dict_message)
    time_stamp = dict_message['mil_sec']
    # computer average angle in the window
    angle_list = [window[i]['angle'] for i in range(len(window))]
    time_list = [int(window[i]['mil_sec']) for i in range(len(window))]
    for index, item in enumerate(time_list):
        if item < 0:
            time_list[index] = time_list[index] + 60

    # compute the angular speed:
    angle_change = max(angle_list) - min(angle_list)
    time_change = (time_list[-1] - time_list[0]) / 1000 # ms to seconds
    if time_change != 0:
        angular_speed = angle_change / time_change
    print('angular_speed', angular_speed)
    angle_sum = sum(angle_list)
    angle_average = angle_sum / len(angle_list)
    # print('average angle during 2 seconds is:       ', angle_average)

    # simple expert system: excerise quality depends on the angle of left arm
    # add angular_speed judgment
    if angle_average < 80:
        # print('angular_speed', angular_speed)
        print('Higher!')
        negativeQuality3 += 1
        

    else: # average angle during 2 seconds is larger than 80 degrees
        if angular_speed >=0: # only tracking the up movement
            if angular_speed < 30:
                print('Faster!')
                negativeQuality3 += 1
            else:
                print('Good excerise!')
                positiveQuality3 += 1
                # print('angular_speed', angular_speed)
                # print('Good! Keep doing!')

                # producer_quality.send('quality-topic', {'quality': quality, 'ang': angle_average}) # sent data like this: 
        else:
            print('only tracking upmovement!')
    #print('time interval:', time_change)
    #print()
       
         
 # read message from "angle-topic"
for message in consumer:
    whole_message = message.value.decode('ascii')
    dict_message = json.loads(whole_message)
    time_stamp = dict_message['mil_sec']
    angle = dict_message['angle']
    checkSpeedUpmovement()
    checkNumberOfExercises()
    checkExerciseLength()
    #print(state)
    producer_quality.send('quality-topic', {'mil_sec': time_stamp, 'negativeQuality1': negativeQuality1, 'positiveQuality1': positiveQuality1,'negativeQuality2': negativeQuality2, 'positiveQuality2': positiveQuality2, 'negativeQuality3': negativeQuality3, 'positiveQuality3': positiveQuality3}) 

