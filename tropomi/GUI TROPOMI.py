# Authors: TROPOMI project
# Date: 10-06-2020


import PySimpleGUI as sg
import numpy as np

myfont = "Ariel 16"

sg.theme('DarkAmber')   # Add a touch of color
g1 = sg.Graph((200,200), (0,0), (200,200),background_color="blue")
g2 = sg.Graph((200,200), (500,500), (200,200),background_color="green")

# All the stuff inside your window.
layout = [  [sg.Text('TROPOMI GUI, please input your output data path:')],
            [sg.Text('After this, press OK.')],
            [sg.Text('Your data will be processed after pressing OK'), sg.InputText()],
            [sg.Button('Ok'), sg.Button('Cancel')], 
            [sg.Text('TROPOMI Methane leakage results',font=myfont)],
            [g1] ]

def graph():
    random_plot = np.random.normal(200000, 25000, 5000)
    plt.hist(random_plot)
    plt.show()

# Create the Window
window = sg.Window('TROPOMI GUI', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    print('You entered ', values[0])

window.close()