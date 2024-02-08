import streamlit as st
import os
import http.client
import json

"""
# Welcome to DocVerify!

"""

# Call the function to process files
# process_files_in_folder(folder_path)

with st.form(key='my_form'):
    docid = st.text_input('Aadhaar number👇', '111111111111')
    agree = st.checkbox('I agree that this data is shared with the informed consent of owner / user for the purpose of verification and processing.')
    submit_button = st.form_submit_button(label='Validate')

if agree and submit_button:
    st.write('Thank you for accepting the consent! :sunglasses:')

    with st.spinner('Fetching details...'):
        conn = http.client.HTTPSConnection("api.gridlines.io")
        # payload = "{\n  \"pan_number\": \"" + docid + "\",\n  \"consent\": \"Y\"\n}"
        payload = "{\n  \"aadhaar_number\": \"" + docid + "\",\n  \"consent\": \"Y\"\n}"

        headers = {
            'X-Auth-Type': "API-Key",
            'Content-Type': "application/json",
            'Accept': "application/json",
            'X-API-Key': st.secrets["API_KEY"]
        }
        # conn.request("POST", "/pan-api/fetch", payload, headers)
        conn.request("POST", "/aadhaar-api/verify", payload, headers)

        res = conn.getresponse()
        data = res.read()
        dataj = json.loads(data)

        if "status" not in dataj or ("status" in dataj and dataj["status"] != 200):
            st.error(data.decode("utf-8"), icon="🔥")
        else:
            st.snow()
            st.success(data.decode("utf-8"))
else:
    st.error('Sorry, cannot proceed without your consent!', icon="🚨")



