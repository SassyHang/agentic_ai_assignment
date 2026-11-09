import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, load_tools
import os

# Set page config
st.set_page_config(page_title="Agentic AI", page_icon="🧠")
st.title("🧠 Agentic AI Assistant")
st.write("I am an Agentic AI. I can autonomously use tools like Wikipedia for research and a Calculator for math.")

# Configure API Key securely
API_KEY = os.getenv("GEMINI_API_KEY") 
if not API_KEY:
    st.warning("Please set your GEMINI_API_KEY environment variable.")
    st.stop()

# Initialize the Agent (Cached so it doesn't reload on every interaction)
@st.cache_resource
def get_agent():
    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
    
    # Load the tools the agent is allowed to use
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    
    # Initialize a ReAct (Reasoning + Acting) Agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, # Allows us to see the agent's thought process in the terminal
        handle_parsing_errors=True
    )
    return agent

agent = get_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# React to user input
if prompt := st.chat_input("Ask a complex math question or search for a topic..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from the Agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking and deciding which tool to use..."):
            try:
                # The agent decides autonomously how to answer
                response = agent.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error executing agent logic: {e}")