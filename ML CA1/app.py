import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle
import logging as log



app = Flask(__name__)

model = pickle.load(open('Dt.pkl','rb'))


@app.route('/',methods=['GET'])
def home():
    log.warning('<--------------------Inside home -------------------->')
    return render_template('home.html')




@app.route('/predict',methods=['POST'])

def predict():
    log.warning('<--------------------Inside Predict ------------------------->')
    df = pd.DataFrame(columns=['duration', 'days_left', 'x0_AirAsia', 'x0_Air_India', 'x0_GO_FIRST',
       'x0_Indigo', 'x0_SpiceJet', 'x0_Vistara', 'x0_Bangalore', 'x0_Chennai',
       'x0_Delhi', 'x0_Hyderabad', 'x0_Kolkata', 'x0_Mumbai', 'd0_Bangalore',
       'd0_Chennai', 'd0_Delhi', 'd0_Hyderabad', 'd0_Kolkata', 'd0_Mumbai',
       'departure_time', 'arrival_time', 'stops', 'class'])

    template = "An exception of type {0} occurred."
    lst = []

    dict_Airline = {"SpiceJet": [0.0,0.0,0.0,0.0,1.0,0.0],
                    "AirAsia":  [1.0,0.0,0.0,0.0,0.0,0.0],
                    "Vistara":  [0.0,0.0,0.0,0.0,0.0,1.0],
                    "GO_FIRST": [0.0,0.0,1.0,0.0,0.0,0.0],
                    "Indigo":   [0.0,0.0,0.0,1.0,0.0,0.0],
                    "Air_India":[0.0,1.0,0.0,0.0,0.0,0.0]
                    }

    dict_City = {"Bangalore": [1.0,0.0,0.0,0.0,0.0,0.0],
                "Chennai":    [0.0,1.0,0.0,0.0,0.0,0.0],
                "Kolkata":    [0.0,0.0,0.0,0.0,1.0,0.0],
                "Mumbai":     [0.0,0.0,0.0,0.0,0.0,1.0],
                "Delhi":      [0.0,0.0,1.0,0.0,0.0,0.0],
                "Hyderabad":  [0.0,0.0,0.0,1.0,0.0,0.0]
                }


    dict_time = {"Evening":0,"Early_Morning":1,"Morning":2,"Afternoon":3,"Night":4,"Late_Night":5}

    dict_stops = {"zero":0,"one":1,"two_or_more":2}

    dict_class = {"Economy":0,"Business":1}


    features =[x for x in request.form.values()]
    airline = features[0]
    Dtime = features[1]
    stops = features[2]
    Atime = features[3]
    DCity = features[4]
    SCity = features[5]
    classType = features[6]
    Duration = features[7]
    Dleft = features[8]

    try:
        lst.append(float(Duration))
        lst.append(float(Dleft))

        for i in dict_Airline[airline]:
            lst.append(i)

        for i in dict_City[DCity]:
            lst.append(i)

        for i in dict_City[SCity]:
            lst.append(i)
        
        lst.append(dict_time[Dtime])
        lst.append(dict_time[Atime])
        lst.append(dict_stops[stops])
        lst.append(dict_class[classType])


    except  Exception as ex :
            print(template.format(type(ex).__name__)) 


    if DCity == SCity:
        return render_template('home.html',prediction_text='Source And Destination Cannot be same')
    log.warning('<-----Values of List are------->',lst)
    df.loc[len(df)] = lst    
    prediction = model.predict(df)



    log.warning('<----------------Prediction value-------->',prediction)
    return render_template('home.html',prediction_text='Price of Ticket Should be: {}'.format(prediction))
    

if __name__ == "__main__":
    app.run()

