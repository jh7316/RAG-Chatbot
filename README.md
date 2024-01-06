# Question and Answer Chatbot

Chatbot that answers questions related to a user-provided pdf file. 
Implemented with RAG model using Langchain.
- streamlit_main.py : chatbot UI via streamlit
- backend.py : creating index from pdf file, creating augmented prompt with context
- answer.txt : contains answers from chatbot for provided questions


Demonstration video:
[![Demo Video](https://img.youtube.com/vi/H67801b5EQ4/0.jpg)](https://www.youtube.com/watch?v=H67801b5EQ4)

- Prompts the user to upload pdf file if user tries to ask question without uploading any file, as demonstrated at the start of the video
- RateLimitError may occur due to limit in usage of openAI API if too many quetions are asked in a short amount of time


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required modules in requirements.txt.

```bash
pip install -r requirements.txt
```

## Usage
Run the following command to execute streamlit app on your local environment.
```bash
streamlit run streamlit_main.py
```

