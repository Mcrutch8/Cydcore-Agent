# app.py

import streamlit as st
import agent
import os
from agent import initialize_model, get_answer
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Retrieve the API key from Streamlit Secrets
api_key = st.secrets["openai_key"]

# Pass the API key to a function in agent.py
result = agent.initialize_model(api_key)


def main():

    st.title("Customer Support Assistant")
    st.write("Interact with our AI-powered customer support assistant.")

    # Initialize the QA chain and chat history
    if 'qa_chain' not in st.session_state:
        st.session_state['qa_chain'] = initialize_model(api_key)

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display previous messages
    for message in st.session_state['chat_history']:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, SystemMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Accept user input
    if user_input := st.chat_input("Type your message"):
        # Add user message to chat history
        user_message = HumanMessage(content=user_input)
        st.session_state['chat_history'].append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.spinner("Generating response..."):
            assistant_response = get_answer(
                st.session_state['qa_chain'],
                user_input,
                st.session_state['chat_history'][:-1]  # Exclude the latest user message
            )

        # Add assistant response to chat history
        assistant_message = SystemMessage(content=assistant_response)
        st.session_state['chat_history'].append(assistant_message)

        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

if __name__ == "__main__":
    main()