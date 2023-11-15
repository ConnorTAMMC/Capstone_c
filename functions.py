from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, AgentType, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import glob
import json
from datetime import datetime
from dotenv import load_dotenv
import os
from typing import NamedTuple

load_dotenv()

def setopenai(key):

    os.environ['OPENAI_API_KEY'] = key
def save_chart(query):
    q_s= "If any charts or graphs or plots were created save them localy and include the save file names in your response."
    query += "."+q_s
    return query

def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())  #short term memory
    df_arr, df_arr_name = load_data([uploaded_file])

    llm = ChatOpenAI(
        temperature=0.0,
        model_name="gpt-3.5-turbo-1106",
        streaming=True, ) # ! important
        # callbacks=[StreamingStdOutCallbackHandler()]
        # ! important ,every generate new token will be taken to the standard ourput
        # agent.agent.llm_chain.prompt.template see the prompt of agent
    agent = create_pandas_dataframe_agent(
        llm,
        df_arr,
        return_intermediate_steps=True,
        save_charts=True,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent, df_arr, df_arr_name


def load_data(uploaded_files):
    file_formats = {
        "csv": pd.read_csv,
        "xls": pd.read_excel,
        "xlsx": pd.read_excel,
        "xlsm": pd.read_excel,
        "xlsb": pd.read_excel,
    }
    df=[]
    df_names = []
    for file in uploaded_files:
        try:
            ext = os.path.splitext(file.name)[1][1:].lower() #get the format name
        except:
            ext = file.name.split(".")[-1]  #split by .
        if ext in file_formats:
            df.append(file_formats[ext](file))  # e.g=pd.read_csv(xxx)
            df_names.append(file.name+"."+ext)
        else:
            st.error(f"Unsupported file format: {ext}")
    return df, df_names



def run_query(agent, query_):
    if 'chart' in query_ or 'charts' in query_ or 'graph' in query_ or 'graphs' in query_ or 'plot' in query_ or 'plt' in query_:
        query_ = save_chart(query_)
    output = agent(query_)
    response, intermediate_steps = output['output'], output["intermediate_steps"]
    thought, action, action_input, observation, steps = decode_intermediate_steps(intermediate_steps)
    store_convo(query_, steps, response)
    return response, thought, action, action_input, observation

from typing import NamedTuple

class AgentAction(NamedTuple):
    tool: str
    tool_input: str
    log: str

def decode_intermediate_steps(steps):
    thought_ = []
    action_ = []
    action_input_ = []
    observation_ = []
    log = []
    text = ""

    for step in steps:
        if isinstance(step[0], AgentAction):
            thought_.append(':green[{}]'.format(step[0].log.split('Action:')[0]))
            action_.append(':green[Action:] {}'.format(step[0].log.split('Action:')[1].split('Action Input:')[0]))
            action_input_.append(':green[Action Input:] {}'.format(step[0].log.split('Action:')[1].split('Action Input:')[1]))
        else:
            thought_.append('')  # Placeholder if step[0] is not an AgentAction
            action_.append('')
            action_input_.append('')

        observation_.append(':green[Observation:] {}'.format(step[1]))

        log.append([thought_[-1], action_[-1], action_input_[-1]])

        text = step[0].log + ' Observation: {}'.format(step[1])

    return thought_, action_, action_input_, observation_, text

def get_convo():
    convo_file = 'Conversation_history.json'
    with open(convo_file, 'r',encoding='utf-8') as f:
        data = json.load(f)
    return data, convo_file
def store_convo(query, steps, response):
    data, convo_file = get_convo()
    current_dateTime = datetime.now()
    data['{}'.format(current_dateTime)] = []
    data['{}'.format(current_dateTime)].append({'Question': query, 'Answer': response, 'Steps': steps})

    with open(convo_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

