__author__ = 'church-father'
# python.jobble.com/81215/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets,linear_model

def get_data(file_name):
    data=pd.read_csv(file_name)
    x_parameter=[]
    y_parameter=[]
    for sigle_square_feet,single_price_value in zip(data['square_feet'],data['price']):
        x_parameter.append([float(sigle_square_feet)])
        y_parameter.append([float(single_price_value)])
    # print(x_parameter)
    # print(y_parameter)
    return x_parameter,y_parameter

def linear_model_main(x_parameter,y_parameter,predict_value):
    regr=linear_model.LinearRegression()
    regr.fit(x_parameter,y_parameter)
    predict_outcome=regr.predict(predict_value)
    predicts={}
    predicts['intercept']=regr.intercept_
    predicts['coefficient']=regr.coef_
    predicts['predicted_value']=predict_outcome
    return predicts

def show_linear_line(x_parameter,y_parameter):
    regr=linear_model.LinearRegression()
    regr.fit(x_parameter,y_parameter)
    plt.scatter(x_parameter,y_parameter,color='blue')
    plt.plot(x_parameter,regr.predict(x_parameter),color='red',linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.show()

x_parameter,y_parameter=get_data("./import_data.csv")
result=linear_model_main(x_parameter,y_parameter,700)
print result
show_linear_line(x_parameter,y_parameter)
