import streamlit as st
import pickle

st.title('Fraud Email Detection')
feature_transformer = pickle.load(open("count_vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

# Mock Email Format

mail_from = st.text_input('From')
mail_to = st.text_input('To')
mail_subject = st.text_input('Subject')
mail_body = st.text_area('Body')

data = [mail_from + mail_to + mail_subject + mail_body]
res = feature_transformer.transform(data).toarray()


if st.button('Is fraud'):
    result = model.predict(res)
    prob = model.predict_proba(res) 
    st.write(f"class is {result[0]}, with probabilities  : Not Fraudulent {round(prob.T[0][0],3)} and Fraudulent {round(prob.T[1][0],3)}")
