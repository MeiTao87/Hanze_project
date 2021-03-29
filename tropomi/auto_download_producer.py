import os
from shapely.geometry import Polygon
from urllib.request import Request, urlopen
from kafka import KafkaProducer
# from pytropomi.downs5p import downs5p
import json
from pytropomi.s5p import s5p
import datetime
import math
import csv

savepath = "C:/Users/Samir/Documents/output_processed_data/"
userInputPath = "C:/Users/Samir/Documents/data_fusion_temp/userInput.csv"
with open(userInputPath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        userInputDay = row[0]
        userInputmonth = row[1]
        userInputYear = row[2]
        userInputDeltaDay = row[3]
        userInputDaysBetween = row[4]
        userInputlon1= row[5]
        userInputlat1 = row[6]
        userInputlon2= row[7]
        userInputlat2 = row[8]

today = datetime.date.today()
# producer to send data
adress_producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                        value_serializer=lambda m:
                        json.dumps(m).encode('ascii'))
products = "L2__CH4___"
# coordinates of the polygon 
# top_right = (120, 45)
# bottom_left = (100, 30)
top_right = (int(userInputlon1), int(userInputlat1))
bottom_left = (int(userInputlon2), int(userInputlat2))
# the constraints of longitude and latitude
lon_constraints = [top_right[0], bottom_left[0]] # 115, 110
lat_constraints = [top_right[1], bottom_left[1]] # 40, 35
# Added this because Polygon objects cannot be serialized. This list can be turned into a polygon on the receiving end
polygon_list = [bottom_left, (top_right[0], bottom_left[1]), top_right, (bottom_left[0], top_right[1]), bottom_left]
polygon = Polygon(polygon_list)
# area is the minimum overlapping area to get a serach result
area = 20
# GUI will give input for day, month, year
# Day = '14'
# Month = '02'
# Year = '2020'
Day = int(userInputDay)
Month = int(userInputmonth)
Year = int(userInputYear)
# For some reason day must be +1
date = datetime.datetime(year=int(Year), month=int(Month), day=int(Day)+1) # 2020,02,15
# deltaday is the time interval,
# days_between is the days between two time intervals we want to compare
# deltaday = 2
# days_between = 60
deltaday = int(userInputDeltaDay)
days_between = int(userInputDaysBetween)
# time interval of the date we are interested in
beginPosition = date + datetime.timedelta(days=-(deltaday+1)) # 2020,02,13
endPosition = date # 2020,02, 15
# time interval of the date we will compare to (previous date)
beginPosition_history = beginPosition + datetime.timedelta(days=-(days_between)) # 2020,02,13 - 300 days
endPosition_history = endPosition + datetime.timedelta(days=-(days_between))# 2020,02, 15
# We are going to do two search: one using the given date
# another using the previous date
# add lat_constraints and lon_constraints to the data dict
data = {'lat_constraints': lat_constraints, 'lon_constraints': lon_constraints}
# search for the given date
sp = s5p(producttype=products, processinglevel='L2',
        beginPosition=beginPosition, endPosition=endPosition,  processingmode='Offline')
login = sp.login()
if login:
    print('login successfully!')
sfs = list(sp.search(polygon=polygon, area=area))
print('The total number of file indexed is {0}.'.format(sp.totalresults))
adress_list = []
# search the given day
for i in range(1, math.ceil(sp.totalresults/sp._limit + 1)):
    for sg in sfs:
        adress = savepath + str(sg[2])
        if(os.path.exists(adress)):
            print("Already downloaded")
        else:
            print('now, download {0}, the total size of file is {1}.'.format(sg[2], sg[3]))
            sp.download(sg[1], filename=sg[2], savepath=savepath)
            print('The file just downloaded is: ', adress)
        # append the search results to the list
        adress_list.append(adress)
    print('searching from page {0}...'.format(i+1))
    sfs = sp.next_page(offset=i*sp._limit)
data['adress_list_now'] = adress_list

# search for the previous date
sp_history = s5p(producttype=products, processinglevel='L2',
        beginPosition=beginPosition_history, endPosition=endPosition_history,  processingmode='Offline')
login_history = sp_history.login()
if login_history:
    print()
    print('login for history search successfully!')
sfs_history = list(sp_history.search(polygon=polygon, area=area))
adress_list_prev = []
print('The total number of file days before indexed is {0}.'.format(sp_history.totalresults))
for i in range(1, math.ceil(sp_history.totalresults/sp_history._limit + 1)):
    for sg in sfs_history:
        adress_history = savepath + str(sg[2])
        if(os.path.exists(adress_history)):
            print("Already downloaded")
        else:
            print('now, download {0}, the total size of file is {1}.'.format(sg[2], sg[3]))
            sp_history.download(sg[1], filename=sg[2], savepath=savepath)
            print('The file just downloaded is: ', adress_history)
        # append the search results to the list
        adress_list_prev.append(adress_history)
    print('searching from page {0}...'.format(i+1))
    sfs_history = sp_history.next_page(offset=i*sp_history._limit)
data['adress_list_prev'] = adress_list_prev
# After the data of both given date and previous date been downloaded, producer will send data 
adress_producer.send('adress_topic', value=data)
adress_producer.flush()
