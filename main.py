import pandas as pd
import streamlit as st
import pickle

st.title('Home Loan Simulation')

#Collect client profil
st.sidebar.header('Characteristics')

def client_characteristics():

    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))

    Married = st.sidebar.selectbox('Married', ('Yes', 'No'))

    Dependents = st.sidebar.selectbox('Dependents', ('0', '1', '2', '3+'))

    Education = st.sidebar.selectbox('Education', ('Graduate', 'Not Graduate'))

    Self_Employed = st.sidebar.selectbox('Self_Employed', ('Yes', 'No'))

    ApplicantIncome = st.sidebar.slider('Applicant income', min_value=0, max_value=100000)

    CoapplicantIncome = st.sidebar.slider('Co Applicant income', min_value=0, max_value=100000)

    LoanAmount = st.sidebar.slider('Loan amount in K$', min_value=0, max_value=5000)

    Loan_Amount_Term = st.sidebar.selectbox('Loan amount term (Months) ', (12, 24, 36, 48, 60, 72, 84,
                                                                           96, 108, 120, 132, 144, 156,
                                                                           168, 180, 192, 204, 216, 228,
                                                                           240, 252, 264, 276, 288, 300,
                                                                           312, 324, 336, 348, 360, 372,
                                                                           384,396, 408, 420, 432, 444,
                                                                           456, 468, 480))

    Credit_History = st.sidebar.selectbox('Credit History', (1, 0))
    Property_Area = st.sidebar.selectbox('Property Area', ('Urban', 'Rural', 'Semiurban'))


    client = {
        'Gender' : Gender,
        'Married' : Married,
        'Dependents' : Dependents,
        'Education' : Education,
        'Self_Employed' : Self_Employed,
        'ApplicantIncome' : ApplicantIncome,
        'CoapplicantIncome' : CoapplicantIncome,
        'LoanAmount' : LoanAmount,
        'Loan_Amount_Term' : Loan_Amount_Term,
        'Credit_History' : Credit_History,
        'Property_Area' : Property_Area
    }

    df_client = pd.DataFrame(client, index=[0])

    return df_client

input_client = client_characteristics()

#Load initial data
df = pd.read_csv('home_loan.csv')
credit_input = df.drop(['Loan_ID', 'Loan_Status'], axis=1)

input_data = pd.concat([input_client, credit_input], axis=0)

var_cat = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Property_Area']

for col in var_cat:
    dummy= pd.get_dummies(input_data[col], drop_first=True)
    input_data = pd.concat([dummy, input_data], axis=1)
    del input_data[col]

input_data = input_data[:1]

loan_predictor = pickle.load(open('home_loan.pkl', 'rb'))

result = loan_predictor.predict(input_data)

st.subheader('Results : ')

if result == 0:
 st.write("Sorry, we can't give you the credit")
else:
 st.write("Congrats, you get the credit !")