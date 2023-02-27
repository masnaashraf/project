import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")


DATA_PATH = os.path.join(dir_of_interest, "data", "iris.csv")

st.title("Iris Flower Prediction APP")



df_file= pd.read_csv(DATA_PATH)

st.write("""
Simple Prediction - Iris flower type
This app predicts the iris flower type

Give it a try""")

st.header("User Input Parameters")

def user_input():
    SepalLengthCm=st.slider("sepal length",4.3,7.9,5.4)
    SepalWidthCm=st.slider("sepal width",2.0,4.4,3.4)
    PetalLengthCm=st.slider("petal length",1.0,6.9,1.3)
    PetalWidthCm=st.slider("petal width",0.1,2.5,0.2)

    data={"SepalLengthCm":SepalLengthCm,
          "SepalWidthCm":SepalWidthCm,
          "PetalLengthCm":PetalLengthCm,
          "PetalWidthCm":PetalWidthCm
          }
    features=pd.DataFrame(data,index=[0])
    return features

df=user_input()
st.header("Features you have selected are")
st.write(df)


X=df_file.iloc[:,:-1]
Y=df_file.iloc[:,-1]

clf=RandomForestClassifier()
clf.fit(X,Y)

y_res=clf.predict(df)
r=y_res[0]
st.header("The flower type is")
st.write(r)
st.balloons()




