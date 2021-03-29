from tkinter import *
from functools import partial
import csv
import os
window=Tk()


dayInput = []
monthInput = []
yearInput = []
deltaDayInput = []
daysBetweenInput = []
long1 = []
lat1= []
long2 = []
lat2 = []

userInputPath = "C:/Users/Samir/Documents/data_fusion_temp/userInput.csv"
producerPath = "C:/Users/Samir/Documents/tropomi-group-repo-s-group-legend-ab-c-d-k-m-s-master/auto_download_producer.py"
consumerPath = "C:/Users/Samir/Documents/tropomi-group-repo-s-group-legend-ab-c-d-k-m-s-master/auto_download_consumer_withGUI.py"
# add widgets here
def validateLogin(username, password):
    
    if((username.get()=='1')):
        if((password.get()=='1')):
            nlbl=Label(window, text="Login Successfull",fg='white',bg='black',font=("Helvetica",10))
            nlbl.place(x=440,y=220)
            
            inputfield1=Label(window, text= "Day", fg='black',font=("Helvetica",10))
            inputfield1.place(x=90,y=250)
            field1 = IntVar()
            inputfield1Entry=Entry(window, text=field1, fg='black',font=("Helvetica",10))
            inputfield1Entry.place(x=200,y=250)
            
            inputfield2=Label(window, text="Month", fg='black',font=("Helvetica",10))
            inputfield2.place(x=90,y=300)
            field2 = IntVar()
            inputfield2Entry=Entry(window, text=field2, fg='black',font=("Helvetica",10))
            inputfield2Entry.place(x=200,y=300)
            
            inputfield3=Label(window, text="Year", fg='black',font=("Helvetica",10))
            inputfield3.place(x=90,y=350)
            field3 = IntVar()
            inputfield3Entry=Entry(window, text=field3, fg='black',font=("Helvetica",10))
            inputfield3Entry.place(x=200,y=350)
            
            inputfield4=Label(window, text="Delta Day", fg='black',font=("Helvetica",10))
            inputfield4.place(x=90,y=400)
            field4 = IntVar()
            inputfield4Entry=Entry(window, text=field4, fg='black',font=("Helvetica",10))
            inputfield4Entry.place(x=200,y=400)
            
            inputfield5=Label(window, text="Days Between", fg='black',font=("Helvetica",10))
            inputfield5.place(x=90,y=450)
            field5 = IntVar()
            inputfield5Entry=Entry(window, text=field5, fg='black',font=("Helvetica",10))
            inputfield5Entry.place(x=200,y=450)
            
            n1labl=Label(window, text="Top Right", fg='black', font=("Helvetica",10))
            n1labl.place(x=600,y=250)
            
            inputfield6=Label(window, text="Longitude",fg='black',font=("Helvetica",10))
            inputfield6.place(x=590,y=300)
            field6=IntVar()
            inputfield6Entry=Entry(window, text=field6, fg='black',font=("Helvetica",10))
            inputfield6Entry.place(x=670,y=300)
            
            inputfield7=Label(window, text="Latitude",fg='black',font=("Helvetica",10))
            inputfield7.place(x=590,y=350)
            field7=IntVar()
            inputfield7Entry=Entry(window, text=field7, fg='black',font=("Helvetica",10))
            inputfield7Entry.place(x=670,y=350)
            
            n1labl=Label(window, text="Bottom Left", fg='black', font=("Helvetica",10))
            n1labl.place(x=600,y=400)
            
            inputfield8=Label(window, text="Longitude",fg='black',font=("Helvetica",10))
            inputfield8.place(x=590,y=450)
            field8=IntVar()
            inputfield8Entry=Entry(window, text=field8, fg='black',font=("Helvetica",10))
            inputfield8Entry.place(x=670,y=450)
            
            inputfield9=Label(window, text="Latitude",fg='black',font=("Helvetica",10))
            inputfield9.place(x=590,y=500)
            field9=IntVar()
            inputfield9Entry=Entry(window, text=field9, fg='black',font=("Helvetica",10))
            inputfield9Entry.place(x=670,y=500)
            #function to show the map with leakage data with its respective input fields:
            def apply():
                global dayInput
                global monthInput
                global yearInput
                global deltaDayInput
                global daysBetweenInput
                global long1
                global lat1
                global long2
                global lat2
                
                #this part will create the mapping with the producer and consumer to take the data and plot it using the basemap
                #for now its a print statement of your map is getting created
                daytemp = inputfield1Entry.get()
                dayInput.append(daytemp)
                
                monthtemp = inputfield2Entry.get()
                monthInput.append(monthtemp)
                
                yeartemp = inputfield3Entry.get()
                yearInput.append(yeartemp)
                
                deltaDaytemp = inputfield4Entry.get()
                deltaDayInput.append(deltaDaytemp)
                
                daysBetweentemp = inputfield5Entry.get()
                daysBetweenInput.append(daysBetweentemp)
                
                long1temp = inputfield6Entry.get()
                long1.append(long1temp)
                
                lat1temp = inputfield7Entry.get()
                lat1.append(lat1temp)
                
                long2temp = inputfield8Entry.get()
                long2.append(long2temp)
                
                lat2temp = inputfield9Entry.get()
                lat2.append(lat2temp)
              
                #userInput = str(dayInput) + "," + str(monthInput) + "," + str(yearInput) + "," + str(deltaDayInput) + "," + str(daysBetweenInput)
                userInput = [dayInput[0], monthInput[0], yearInput[0], deltaDayInput[0], daysBetweenInput[0], long1[0], lat1[0], long2[0], lat2[0]]
                print(userInput)
                # open file in append mode, with \n char
                with open(userInputPath, 'a', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerow(userInput)
                dayInput = []
                monthInput = []
                yearInput = []
                deltaDayInput = []
                daysBetweenInput = []
                long1 =[]
                lat1=[]
                long2=[]
                lat2=[]
                #window.destroy()
                # my_dir = os.path.dirname(sys.argv[0])
                # os.system('%s %s %s' % (sys.executable, 
                #                         os.path.join(my_dir, consumerPath),
                #                         os.path.join(my_dir, producerPath) ) )
                #os.system(consumerPath)
                os.system(producerPath)
                disp=Label(window, text="The map is being generated", fg='black',font=("Helvetica",16))
                disp.place(x=350,y=600)
                return
            
            nbtn=Button(window, text="Apply", fg='black', bg='white', font=("Helvetica",16), command=apply)
            nbtn.place(x=460, y=500)
            
        else:
            nelbl=Label(window, text="Incorrect Password or Username",fg='white',bg='black',font=("Helvetica",10))
            nelbl.place(x=380,y=220)
            window.destroy()
    else:
        elbl=Label(window, text="Incorrect Password or Username",fg='white',bg='black',font=("Helvetica",10))
        elbl.place(x=380,y=220)
        window.destroy()
        
    return 

window.title('TROPOMI Methane Leakage Monitoring')
window.geometry("1000x900+10+20")
lbl=Label(window, text="Welcome to Methane Leakage Tracker", fg='Black',font=("Helvetica", 26))
lbl.place(x=200, y=10)

usernameLabel = Label(window, text="User Name", fg ='Black', font=("Helvetica", 16))
usernameLabel.place(x=100,y=100)
username = StringVar()
usernameEntry = Entry(window, textvariable=username, fg='Black', bg='White', font=("Helvetica",16))
usernameEntry.place(x=250, y=100)

passwordLabel = Label(window,text="Password", fg='Black', font=("Helvetica",16))
passwordLabel.place(x=550,y=100)
password = StringVar()
passwordEntry = Entry(window, textvariable=password, show='*',font=("Helvetica",16))
passwordEntry.place(x=680,y=100)

validateLogin = partial(validateLogin, username, password)

btn=Button(window, text="Tropomi Login", fg='black', bg='white', font=("Helvetica",16), command=validateLogin)
btn.place(x=420, y=150)
window.mainloop()