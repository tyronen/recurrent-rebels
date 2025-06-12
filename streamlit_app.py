import streamlit as st
import requests
import datetime

# --- Configuration ---
FASTAPI_URL = "http://127.0.0.1:8000"
HN_API_BASE_URL = "https://hacker-news.firebaseio.com/v0"

# --- Initialize Session State ---
# Session state is used to store information across app reruns.
# We initialize the form fields here to prevent them from resetting.
if 'by' not in st.session_state:
    st.session_state.by = "testuser"
if 'title' not in st.session_state:
    st.session_state.title = "My awesome new project"
if 'url' not in st.session_state:
    st.session_state.url = "http://example.com"
if 'time_obj' not in st.session_state:
    # We store the datetime object to easily manage date and time inputs
    st.session_state.time_obj = datetime.datetime.now()

# --- Streamlit UI ---

st.title("Hacker News Post Scorer")

# --- Section 1: Fetching Data ---
st.header("1. Fetch Post Data (Optional)")
st.write("Enter a Hacker News Post ID to automatically populate the fields below.")

col1, col2 = st.columns([3, 2])
with col1:
    item_id_input = st.number_input("Hacker News Post ID", min_value=1, value=1171783, step=1, label_visibility="collapsed")

with col2:
    if st.button("Fetch and Populate Form"):
        try:
            with st.spinner(f"Fetching data for item {item_id_input}..."):
                # We can call the public HN API directly from Streamlit
                response = requests.get(f"{HN_API_BASE_URL}/item/{item_id_input}.json")
                response.raise_for_status()
                item_data = response.json()

                if item_data and item_data.get("type") == "story":
                    # Update session state with the fetched data
                    st.session_state.by = item_data.get("by", "")
                    st.session_state.title = item_data.get("title", "")
                    st.session_state.url = item_data.get("url", "")
                    st.session_state.time_obj = datetime.datetime.fromtimestamp(item_data.get("time", 0))
                    st.success(f"Successfully populated form with data for ID {item_id_input}.")
                else:
                    st.warning(f"Item {item_id_input} is not a story or was not found.")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch data from Hacker News API: {e}")

st.divider()

# --- Section 2: Prediction ---
st.header("2. Review Data and Predict Score")
st.write("Review or edit the populated data, then click predict.")


# The input fields now use the values stored in session_state.
# This means they will update automatically when the "Fetch" button is clicked.
by_input = st.text_input("Author (by)", key="by")
title_input = st.text_input("Post Title", key="title")
url_input = st.text_input("URL", key="url")

# Split the datetime object for the UI widgets
date_input = st.date_input("Post Date", st.session_state.time_obj.date())
time_input = st.time_input("Post Time", st.session_state.time_obj.time())

# Convert the final date and time back to a Unix timestamp for the API
datetime_obj = datetime.datetime.combine(date_input, time_input)
time_stamp = int(datetime_obj.timestamp())

st.write(f"_Final Unix Timestamp for API: `{time_stamp}`_")

if st.button("Predict Score"):
    payload = {
        "by": by_input,
        "title": title_input,
        "url": url_input,
        "time": time_stamp
    }

    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(f"{FASTAPI_URL}/predict/direct", json=payload)
            response.raise_for_status()

            prediction_data = response.json()
            st.success("Prediction successful!")
            st.write(f"**Predicted Class:** `{prediction_data['prediction']}`")
            st.bar_chart(prediction_data['probabilities'])
            with st.expander("See Raw JSON Response"):
                st.json(prediction_data)

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while connecting to the API: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
