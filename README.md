# RVI-LexyAI
This Chatbot is based on Chainlit and LlamaIndex. It loads the EU AI-Act as an input into the RAG-system.

## Installation
To install LexyAI, follow these steps:

0. Get source files.
   
1. rename chainlit folder to .chainlit.
      
2. Install python virtual environment:
 
~~~~Terminal
    python3 -m venv .venv
~~~~
 
3. Activate the virtual environment:
 
~~~~Terminal
    .venv/Scripts/activate
~~~~
 
4. Install packages via requirements.txt:
 
~~~~Terminal
    pip install -r requirements.txt
~~~~

5. Create a .env file.

6. Set OPENAI_API_KEY = "YOUR_API_KEY".
 
7. Make sure the OpenAI API key in .env is valid.
 
8. Run the chainlit application locally:
 
~~~~Terminal
    chainlit run main.py -w
~~~~
 
9. A new window will open in your browser.
   
10. If not, navigate to:
 
~~~~Terminal
http://localhost:8000
~~~~

11. To terminate the chainlit application use *CTRL+C*.
