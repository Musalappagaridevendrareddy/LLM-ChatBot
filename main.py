import streamlit as st
from app import ConvertToText
import sqlite3

conn = sqlite3.connect('user_data.db')
conn.execute('''
CREATE TABLE IF NOT EXISTS user_data (
    username TEXT PRIMARY KEY,
    password TEXT,
    path TEXT
)
''')

app = ConvertToText()

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

def create_account(username, password):
    # Insert the new user into the database
    conn.execute('''
    INSERT INTO user_data (username, password)
    VALUES (?, ?)
    ''', (username, password))
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



def main_fun():
    st.set_page_config("Chat PDF")
    col1, col2 = st.tabs(["Locally", "OutOfTheBox"])
    with col1:
        st.header("Chat with Your Data Locally using DEV Bot 🤖")

        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            st.write("Reply: \n", app.user_input(user_question, st.session_state["username"]))

        with st.sidebar:
            # say hello to the user
            st.write(f"Hello, {st.session_state['username']}")
            st.title("Menu:")
            # provide radio button
            option = st.radio("Choose an Option", ["Files", "Link"])
            
            if option == "Link":
                link = st.text_input("Enter the Link")
                val = st.slider("Select the number of links to be extracted", 1, 10)
                # check box
                all_links = st.checkbox("Extract All Links")
                if st.button("Submit"):
                    with st.spinner("Processing..."):
                        app.find_links(link, link, link, val if not all_links else None)
                        raw_text = app.get_link_text()
                        text_chunks = app.get_text_chunks(raw_text)
                        # print(text_chunks)
                        if text_chunks:
                            app.get_vector_store(text_chunks, st.session_state["username"])
                        st.success("Done")

            else:
                st.write("Upload Your Files")
                pdf_docs = st.file_uploader(f"Upload your {option} Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf", "docx", "txt", "rtf", "odt", "pptx", "epub", "py", "wav"])
            
                if st.button("Submit & Process"):
                    with st.spinner("Processing..."):
                        raw_text = app.get_text(pdf_docs)
                        text_chunks = app.get_text_chunks(raw_text)
                        # print(text_chunks)
                        app.get_vector_store(text_chunks, st.session_state["username"])
                        st.success("Done")
            if st.button("Delete My Data"):
                app.delete_user_data(st.session_state["username"])
                st.success("Data Deleted Successfully")
            # provide a button to logout
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.session_state["username"] = None
                st.session_state["login_submitted"] = False
                st.rerun()

    with col2:

        st.header("Chat Out of the box")
        input=st.text_input("Input: ",key="input")
        submit=st.button("Ask the question")
        if submit and input:
            response=app.get_gemini_response(input)
            st.subheader("The Response is")
            for chunk in response:
                st.write(chunk.text)



if __name__ == "__main__":
    if st.session_state.get("logged_in", False):
        main_fun()
    else:
        login_signup()