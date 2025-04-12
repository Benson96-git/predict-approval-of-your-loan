import joblib
import streamlit as st
import pandas as pd
import numpy as np


model_columns = joblib.load('./models/model_columns.pkl')
model = joblib.load('./models/feature_model.pkl')
def add_custom_title_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Lobster&display=swap');

        .title-stylish{
            font-family: 'Lobster', cursive; 
            font-size: 40px; 
            color: #2E86C1; 
            text-align: center; 
        }

        .title-formal{
            font-family: 'Arial', sans-serif;
            font-size: 40px;
            color: #2E86C1; 
            text-align: center;
            margin-bottom: 60px
        }
        </style><br><br>
        """,
        unsafe_allow_html=True
    )
def add_sidebar_footer_email():
    st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .sidebar-footer {
            margin-top: auto; 
            text-align: center; 
            font-size: 14px; 
            color: #555;
        }
        </style>
        <div class="sidebar-footer">
            <p>Need help? Contact us at:</p>
            <a href="mailto:support@bankingapp.com">support@bankingapp.com</a>
        </div>
        """,
        unsafe_allow_html=True
    )

def run():
    add_custom_title_style()
    st.markdown('<h1 class="title-formal">Predict Approval of your Loan</h1>', unsafe_allow_html=True)
    # Sidebar
    st.sidebar.markdown('<h2 class="title-stylish">Your Banking Partner!</h2>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image('bank.png')

    st.sidebar.write("* Use the form on the main page to predict loan approval.")
    st.sidebar.write("* Ensure all fields are filled correctly.")
    add_sidebar_footer_email()
    # Main Page
    col1, col2, col3 = st.columns(3)
    #First row
    with col1:
        no_of_dependents = st.number_input('Number of dependents',min_value=0, value=2)
    with col2:
        education = st.selectbox('Education', options=['Not Graduate', 'Graduate'])
    with col3:       
        self_employed = st.selectbox('Self Employed', options=['No', 'Yes'])
    #Second row
    with col1:
        income_annum = st.number_input("Applicant's Annual Income",min_value=0, value=200000)
    with col2:
        loan_amount = st.number_input("Loan Amount",min_value=0 ,value=10000)   
    with col3:
        loan_term = st.number_input('Tenure(In Years)',min_value=0 ,value=1)
    #Third row
    with col1:
        cibil_score =  st.number_input('Cibil score',min_value=0,max_value=900, value=410)
    with col2:
        residential_assets_value = st.number_input('Residential Assets Value',min_value=0, value=15000)
    with col3:
        commercial_assets_value = st.number_input('Commercial Assests Value',min_value=0, value=16000)
    #$th row
    with col1:
        luxury_assets_value = st.number_input('Luxury Assets Value',min_value=0, value=15600)
    with col2:
        bank_asset_value = st.number_input('Bank Assets Value',min_value=0, value=18900)

    if st.button('Predict Loan Status'):
        #validations
        if loan_amount == 0:
          st.error("Please fill out loan_amount field.")
          return
        if loan_amount > bank_asset_value:
          st.error("Loan Amount cannot exceed Bank Asset Value.")
          return
        if loan_term == 0 and loan_amount > 0:
          st.error("Loan Term must be greater than 0 if a Loan Amount is added.")
          return
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
        print(f"user_input: {user_input}")
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
