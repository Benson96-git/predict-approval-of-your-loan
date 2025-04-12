# import pickle
import joblib
import streamlit as st
import pandas as pd
import numpy as np
# import warnings
# warnings.filterwarnings("ignore")

model_columns = joblib.load('./models/model_columns.pkl')
model = joblib.load('./models/feature_model.pkl')

def run():
    st.title('Predict Approval of your loan')
   
    col1, col2, col3 = st.columns(3)
    #First row
    with col1:
        no_of_dependents = st.number_input('Number of dependents', value=2)
    with col2:
        education = st.selectbox('Education', options=['Not Graduate', 'Graduate'])
    with col3:       
        self_employed = st.selectbox('Self Employed', options=['No', 'Yes'])
    #Second row
    with col1:
        income_annum = st.number_input("Applicant's Annual Income", value=200000)
    with col2:
        loan_amount = st.number_input("Loan Amount", value=10000)   
    with col3:
        loan_term = st.number_input('Tenure(In Years)', value=0)
    #Third row
    with col1:
        cibil_score =  st.number_input('Cibil score', value=410)
    with col2:
        residential_assets_value = st.number_input('Residential Assets Value', value=15000)
    with col3:
        commercial_assets_value = st.number_input('Commercial Assests Value', value=16000)
    #$th row
    with col1:
        luxury_assets_value = st.number_input('Luxury Assets Value', value=15600)
    with col2:
        bank_asset_value = st.number_input('Bank Assets Value', value=18900)

    print('income_annum', income_annum)
    print('no_of_dependents', no_of_dependents)
    print('education', education)
    print('self_employed', self_employed)
    print('loan_amount', loan_amount)
    print('loan_term', loan_term)
    print('cibil_score', cibil_score)
    print('residential_assets_value', residential_assets_value)
    print('commercial_assets_value', commercial_assets_value)
    print('luxury_assets_value', luxury_assets_value)
    print('bank_asset_value', bank_asset_value)

    if st.button('Submit'):
        user_input = {
    'no_of_dependents': no_of_dependents, 
    'education': education,
    'self_employed':self_employed, 
    'income_annum':income_annum,
    'loan_amount':loan_amount, 
    'loan_term':loan_term, 
    'cibil_score':cibil_score, 
    'residential_assets_value':residential_assets_value,
    'commercial_assets_value':commercial_assets_value, 
    'luxury_assets_value':luxury_assets_value, 
    'bank_asset_value':bank_asset_value,
        }
        input_df = pd.DataFrame(user_input,index=range(0,1))
        input_encoded = pd.get_dummies(input_df)

        # Reindex to match training columns (fill missing columns with 0)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        prediction=model.predict(input_encoded)
        pred_prob = model.predict_proba(input_encoded.values.reshape(1, -1))
        success_container = st.container()
        with success_container:
             if prediction[0]:
              st.markdown(
        """
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; color: #155724;">
            <strong>There is a {:.2f}% chance for your loan to get approved.</strong><br>
        </div><br>
        """.format(pred_prob[0][1] * 100),
        unsafe_allow_html=True,
    )
             else:
              st.markdown(
        """
        <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; color: #721c24;">
            <strong>There is a  {:.2f}% chance for your loan to get rejected.<br>
        </div><br>
        """.format(pred_prob[0][0] * 100),
        unsafe_allow_html=True,
    )
        if st.button("Close", key="close_success"):
         success_container.empty() 
run()