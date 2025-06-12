import streamlit as st
import requests
import datetime

# --- Configuration ---
# This is the URL where your FastAPI application is running.
# If you are running both Streamlit and FastAPI on your local machine,
# this default URL should work.
FASTAPI_URL = "http://127.0.0.1:8000"

# --- Streamlit UI ---

st.title("Hacker News Post Scorer")

st.header("Predict Directly from Input Data")
st.write("Fill in the fields below to get a prediction for your post.")

# Create input fields for the user
by_input = st.text_input("Author (by)", "testuser")
title_input = st.text_input("Post Title", "My awesome new project")
url_input = st.text_input("URL", "http://example.com")

# The 'time' field is an integer (Unix timestamp), so we'll use a date and time picker
# and convert it for the user's convenience.
date_input = st.date_input("Post Date", datetime.date.today())
time_input = st.time_input("Post Time", datetime.datetime.now().time())

# Convert date and time to a Unix timestamp
datetime_obj = datetime.datetime.combine(date_input, time_input)
time_stamp = int(datetime_obj.timestamp())

st.write(f"_Converted Unix Timestamp: `{time_stamp}`_")


# Create a button to trigger the prediction
if st.button("Predict Score"):
    # 1. Prepare the data payload for the API request
    payload = {
        "by": by_input,
        "title": title_input,
        "url": url_input,
        "time": time_stamp
    }

    # 2. Make the POST request to the FastAPI endpoint
    try:
        response = requests.post(f"{FASTAPI_URL}/predict/direct", json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # 3. Display the results
        prediction_data = response.json()
        st.success("Prediction successful!")
        st.json(prediction_data) # Display the full JSON response nicely

        # Or display a more formatted output
        st.write(f"**Predicted Score:** `{prediction_data['prediction']}`")

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while connecting to the API: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

