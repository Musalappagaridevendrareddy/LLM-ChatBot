import streamlit as st
import sqlite3
from main import main_fun

# Create a SQLite connection
conn = sqlite3.connect('user_data.db')

# Create a table for the user data
conn.execute('''
CREATE TABLE IF NOT EXISTS user_data (
    username TEXT PRIMARY KEY,
    password TEXT,
    name TEXT,
    path TEXT
)
''')

# Function to verify user credentials
def verify_credentials(username, password):
    # Query the database for the user
    user = conn.execute('''
    SELECT password FROM user_data WHERE username = ?
    ''', (username,)).fetchone()

    # If the user exists and the password is correct, return True
    if user is not None and user[0] == password:
        return True

    # If the user doesn't exist or the password is incorrect, return False
    return False

# Function to create a new user account
def create_account(username, password):
    # Insert the new user into the database
    conn.execute('''
    INSERT INTO user_data (username, password)
    VALUES (?, ?)
    ''', (username, password))
    conn.commit()

# Function to update the user name
def update_name(username, name):
    # Update the name in the database
    conn.execute('''
    UPDATE user_data SET name = ? WHERE username = ?
    ''', (name, username))
    conn.commit()

def login_signup():
    option = st.selectbox("Choose an option", ("Login", "Sign Up"))
    if option == "Login":
        with st.form("login_form"):
            st.header("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button or st.session_state.get("login_submitted", False):
                st.session_state["login_submitted"] = True
                if verify_credentials(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.session_state["login_submitted"] = False
                else:
                    st.error("Invalid username or password.")
    else:
        with st.form("signup_form"):
            st.header("Sign Up")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Sign Up")

            if submit_button:
                create_account(username, password)
                st.success("Account created. You can now log in.")

def profile():
    st.header("Profile")

    # Get the user data from the database
    user_data = conn.execute('''
    SELECT name FROM user_data WHERE username = ?
    ''', (st.session_state["username"],)).fetchone()

    # Display the user data
    st.write("Username:", st.session_state["username"])
    st.write("Name:", user_data[0] if user_data[0] else "No name provided")

    # Edit name form
    with st.form("edit_name_form"):
        new_name = st.text_input("New name")
        submit_button = st.form_submit_button("Update Name")

        if submit_button:
            update_name(st.session_state["username"], new_name)
            st.success("Name updated.")

def main():
    if st.session_state.get("logged_in", False):
        main_fun()
    else:
        login_signup()

main()