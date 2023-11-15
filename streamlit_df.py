import glob
import streamlit as st
from streamlit_chat import message
from streamlit_image_select import image_select
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from functions import *

file_formats = {
        "csv": pd.read_csv,
        "xls": pd.read_excel,
        "xlsx": pd.read_excel,
        "xlsm": pd.read_excel,
        "xlsb": pd.read_excel,
    }
def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False

def get_text(n):
    # Streamlit apps are reactive, meaning that components are re-rendered when the app's state changes
    # Use a unique key for each input box to ensure uniqueness and independent behavior.
    input_text = st.text_input('How are you today?', '', key="input{}".format(n))
    return input_text

def show_data(tabs, df):
    for i, df_ in enumerate(df):
        print(i, len(df_))
        with tabs[i]:
            st.dataframe(df_)
def main():

    st.set_page_config(page_title="LangChain: Chat with pandas", page_icon='🦜')
    st.title("🦜 LangChain: Chat with pandas  🦜")

    OPENAI_API_KEY = st.sidebar.text_input("OPENAI API KEY", type="password")
    if st.sidebar.button("Enter key"):
        setopenai(OPENAI_API_KEY)

    st.sidebar.title("Pandas AI Agent ")
    st.sidebar.write("""
                  ###### This project uses LangChain library utilizing Pandas AI and OpenAI to act as a Data Analyst AI assistant.
                  ###### All :red[conversations are stored] in a JSON file including the question, steps to answer (including code written by AI), and answer for tracking and monitoring of the tool usage.
                  ###### All Charts/Graphs/Plots :red[generated by AI] are saved as well.
                  ###### - If the agent :red[fails to locate the dataframe] for any reason, try specifying it in the prompt (i.e. for dataframe is df1).
                  ###### [My Github](https://github.com/sxaxmz/)
                  ###### [Docs](https://github.com/sxaxmz/PandasGPTAgent)
                  """)

    st.sidebar.title("process")


    uploaded_file = st.file_uploader(
        "Upload a Data file",
        type=list(file_formats.keys()),
        help="Various File formats are Support",
        on_change=clear_submit,
        accept_multiple_files=True
    )
    if not uploaded_file:
        st.warning(
            "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
        )
    agent=""

    if uploaded_file:
        for file in uploaded_file:
            agent, df, df_names = save_uploaded_file(file)
            st.session_state["tabs"].clear()
            for df_name in df_names:
                st.session_state.tabs.append(df_name)
            tabs = st.tabs([s.center(9, "\u2001") for s in st.session_state["tabs"]])
            show_data(tabs, df)

    st.header("AI Agent Output Directory")
    if st.button('Open Directory'):
        os.startfile(os.getcwd())  # Current Working Directory

    imgs_png = glob.glob('*.png')
    imgs_jpg = glob.glob('*.jpg')
    imgs_jpeeg = glob.glob('*.jpeg')
    imgs_ = imgs_png + imgs_jpg + imgs_jpeeg
    if len(imgs_) > 0:
        img = image_select("Generated Charts", imgs_, captions=imgs_, return_value='index')
        st.write(img)

    st.header("Query The Dataframes")
    x=0
    user_input = get_text(x)
    if st.button('Query'):
        x += 1
        # st.write("You:", user_input)
        print(user_input, len(user_input))
        response, thought, action, action_input, observation = run_query(agent, user_input)
        # st.write("Pandas Agent: ")
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        # Loop through the generated messages in reverse order
        for i in range(len(st.session_state['generated']) - 1, -1, -1): # -1:index=0,-1:stop in -1 not include-1,-1:reverse order
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        for i in range(0, len(thought)):
            st.sidebar.write(thought[i])
            st.sidebar.write(action[i])
            st.sidebar.write(action_input[i])
            st.sidebar.write(observation[i])
            st.sidebar.write('====')

if __name__ == "__main__":
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'tabs' not in st.session_state:
        st.session_state['tabs'] = []

    load_dotenv() # Import enviornmental variables
    main()