import streamlit as st
from openai import OpenAI
from backend import get_vectorstore, augment_prompt
import constants
API_KEY=constants.OPENAI_API_KEY

st.title("Knowledge-specific Chatbot")

@st.cache_resource
def create_vectordb(file, filename):
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database"):
        vectordb = get_vectorstore(
            file.getvalue(), filename, API_KEY
        )
    return vectordb


# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=API_KEY)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if("hasPDF" not in st.session_state):
    st.session_state.hasPDF=False
# Upload PDF file
pdf_file = st.file_uploader("", type="pdf", accept_multiple_files=False)

if pdf_file:
    st.session_state["vectordb"] = create_vectordb(pdf_file, pdf_file.name)
    message= "PDF file accepted. Please ask anything related to the PDF content."
    if(st.session_state.hasPDF==False):
        st.session_state.messages.append({"role": "assistant", "content": message})
        st.session_state.hasPDF = True
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# React to user input
if prompt := st.chat_input("Ask a question"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    vectordb = st.session_state.get("vectordb", None)
    augmented_prompt = prompt
    if(vectordb!=None):
        augmented_prompt=augment_prompt(prompt,vectordb)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if(not pdf_file):
            full_response="There is no PDF file. Please upload a PDF file."
        else:
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": "user", "content": augmented_prompt}
                    # {"role": m["role"], "content": m["content"]}
                    # for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})