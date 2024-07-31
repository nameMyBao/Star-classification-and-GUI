import tensorflow as tf
import project as mod
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox, ttk
from sklearn.preprocessing import StandardScaler


direct='C:\\Users\\Admin\\Desktop\\final ml\\Stars.csv'
# load the model
train_feature, test_feature, train_label, test_label, label,mean,scale = mod.data_in(direct)

model = mod.build_model()
trained_epochs, accuracy, loss, val_acc, val_loss = mod.train_model(model, train_feature, train_label, test_feature, test_label)
mod.plot_loss_curve(trained_epochs,  accuracy, loss, val_acc, val_loss)


# input def
color_star=["blue", "bluewhite", "orange" ,"orangered" ,"paleyelloworange" ,"red",
 "white", "whiteyellow", "whitish" ,"yellowish" ,"yellowishwhite",
 "yellowwhite"]
spectral_star=["A" ,"B", "F" ,"G" ,"K", "M" ,"O"]
star_type=["Brown Dwarf","Red Dwarf","White Dwarf","Main Sequence","Super Giant","Hypergiant",]




# root window
root = tk.Tk()
root.geometry('900x700')
root.resizable(True, True)
root.title('Star classification GUI')


root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=2)


# slider current value
temp_current_value = tk.DoubleVar()
lumi_current_value = tk.DoubleVar()
radi_current_value = tk.DoubleVar()
weight_current_value=tk.DoubleVar()



def get_temp_value():
    return '{: .2f}'.format(temp_current_value.get())
def get_lumi_value():
    return '{: .6f}'.format(lumi_current_value.get())
def get_radi_value():
    return '{: .2f}'.format(radi_current_value.get())
def get_weight_value():
    return '{: .2f}'.format(weight_current_value.get())


def display_selection():
    count=0
    # Get the selected value.
    star_color_selection = star_color.get()
    star_spectral_class_selection=star_spectral_class.get()
    
    # get the entry value
    tempval1=get_temp_value()
    lumival1=get_lumi_value()
    radival1=get_radi_value()
    weightval1=get_weight_value()
    colorofstar=color_star.index(star_color.get())
    spectralofstar=spectral_star.index(star_spectral_class.get())
    
    
    # make dataset from userinput
    dataset =np.array([[tempval1, lumival1, radival1, weightval1,colorofstar , spectralofstar]])
    dataset=dataset.astype(float)
    print(dataset)
    datafinal= mod.custom_data(dataset,mean,scale)
    print(datafinal)
    
    # preditction
    prediction=model.predict(datafinal)
    print("Prediction : ", prediction) 
    
    percent=np.reshape(prediction,6)
    percent=100*percent
    for x in percent:
         print(percent[count],star_type[count])
         count+=1
         
    predicted_label = np.argmax(prediction)
    print('predicted label:',predicted_label)
    # load the percent for each star
    mod.star_percent(root,percent)
    
    messagebox.showinfo(
        message=f"The suitable star is: {star_type[predicted_label]}",
        title="Star classification")
       


def temp_slider_changed(event):
    tempval_label.configure(text= get_temp_value())
    # temperature=get_temp_value()
    
    
def lumi_slider_changed(event):
    lumival_label.configure(text=get_lumi_value())
    # luminousity=get_lumi_value()
    
    
    
def radi_slider_changed(event):
    radival_label.configure(text=get_radi_value())
    # radius=get_radi_value()
    
    
def weight_slider_changed(event):
    weightval_label.configure(text=get_weight_value())
    # absoluteweight=get_weight_value()



# label for the slider
Temp_label = ttk.Label(
    root,
    text='Temperature:'
)

Lumi_label = ttk.Label(
    root,
    text='Luminosity:'
)

Radi_label = ttk.Label(
    root,
    text='Radius:'
)
Weight_label = ttk.Label(
    root,
    text='Abs Weight:'
)

# Label grid
Temp_label.grid(
    column=0,
    row=0,
    sticky='w'
)
Lumi_label.grid(
    column=0,
    row=4,
    sticky='w'
)
Radi_label.grid(
    column=0,
    row=8,
    sticky='w'
)
Weight_label.grid(
    column=0,
    row=12,
    sticky='w'
)

# precisive input
temp_sp = tk.Entry(root,textvariable = temp_current_value, font=('calibre',8,'normal'))
temp_sp.grid(column=2,
    row=0,
    sticky='w')

lumi_sp = tk.Entry(root,textvariable = lumi_current_value, font=('calibre',8,'normal'))
lumi_sp.grid(column=2,
    row=4,
    sticky='w')

radi_sp = tk.Entry(root,textvariable = radi_current_value, font=('calibre',8,'normal'))
radi_sp.grid(column=2,
    row=8,
    sticky='w')

weight_sp = tk.Entry(root,textvariable = weight_current_value, font=('calibre',8,'normal'))
weight_sp.grid(column=2,
    row=12,
    sticky='w')

# slider
temp = Scale(
    root,
    from_=0,
    to=40000,
    orient='horizontal',  # vertical
    command=temp_slider_changed,
    variable=temp_current_value
)

lumi = Scale(
    root,
    from_= 0,
    to = 849420,
    resolution=0.000001,
    length = 400,
    orient='horizontal',  # vertical
    command=lumi_slider_changed,
    variable=lumi_current_value
)
radi = Scale(
    root,
    from_=0,
    to=1948.5,
    resolution=0.0001,
    orient='horizontal',  # vertical
    command=radi_slider_changed,
    variable=radi_current_value
)
weight = Scale(
    root,
    from_=-11.92,
    to=20.06,
    resolution=0.01,
    orient='horizontal',  # vertical
    command=weight_slider_changed,
    variable=weight_current_value
)

# slider grid
temp.grid(
    column=1,
    row=0,
    sticky='we'
)
lumi.grid(
    column=1,
    row=4,
    sticky='we'
)
radi.grid(
    column=1,
    row=8,
    sticky='we'
)
weight.grid(
    column=1,
    row=12,
    sticky='we'
)

# current value label

temp_value_label = ttk.Label(
    root,
    text='temperature Value:'
)
lumi_value_label = ttk.Label(
    root,
    text='luminous Value:'
)
radi_value_label = ttk.Label(
    root,
    text='radius Value:'
)
weight_value_label = ttk.Label(
    root,
    text='weight Value:'
)
# current value grid
temp_value_label.grid(
    row=1,
    columnspan=2,
    sticky='n',
    ipadx=10,
    ipady=10
)
lumi_value_label.grid(
    row=5,
    columnspan=2,
    sticky='n',
    ipadx=10,
    ipady=10
)
radi_value_label.grid(
    row=9,
    columnspan=2,
    sticky='n',
    ipadx=10,
    ipady=10
)
weight_value_label.grid(
    row=13,
    columnspan=2,
    sticky='n',
    ipadx=10,
    ipady=10
)
# value label
tempval_label = ttk.Label(
    root,
    text=get_temp_value()
)
lumival_label = ttk.Label(
    root,
    text=get_lumi_value()
)
radival_label = ttk.Label(
    root,
    text=get_radi_value()
)
weightval_label = ttk.Label(
    root,
    text=get_weight_value()
)
# value grid
tempval_label.grid(
    row=2,
    columnspan=2,
    sticky='n'
)
lumival_label.grid(
    row=6,
    columnspan=2,
    sticky='n'
)
radival_label.grid(
    row=10,
    columnspan=2,
    sticky='n'
)
weightval_label.grid(
    row=14,
    columnspan=2,
    sticky='n'
)


# box display
star_color = ttk.Combobox(
    state="readonly",
    values=["blue", "bluewhite", "orange" ,"orangered" ,"paleyelloworange" ,"red",
 "white", "whiteyellow", "whitish" ,"yellowish" ,"yellowishwhite",
 "yellowwhite"]
    
)
star_color_label= ttk.Label(
    root,
    text='Star color'
)
star_color_label.grid(
    column=0,
    row=15,
    sticky='we')
star_color.grid(
    column=0,
    row=16,
    sticky='w')


star_spectral_class = ttk.Combobox(
    state="readonly",
    values=["A" ,"B", "F" ,"G" ,"K", "M" ,"O"]
    
)
star_spectral_class_label= ttk.Label(
    root,
    text='Star spectral class'
)
star_spectral_class_label.grid(
    column=1,
    row=15,
    sticky='we')
star_spectral_class.grid(
    column=1,
    row=16,
    sticky='w')
# final button
button = ttk.Button(text="Display selection", command=display_selection)
button.grid(column=1,
    row=16,
    sticky='n')



root.mainloop()


 


