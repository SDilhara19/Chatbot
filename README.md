# Convogrid_Assessment

# Project Setup Instructions
Navigate to the Project Directory
Open a terminal and go to the project directory.
Create and Activate a Virtual Environment
To create a virtual environment, run the command:
python -m venv .venv
To activate the virtual environment:
On Windows (Command Prompt), use: .\venv\Scripts\activate
On Windows (PowerShell), use: .\venv\Scripts\Activate.ps1
On macOS/Linux, use: source venv/bin/activate

# Install Required Dependencies
Install all the necessary Python dependencies by running:
    pip install -r requirements.txt
Ensure Ollama Server is Running
The ollama server should be up and running. The "llama3.2:1b" model must be pulled to the server by using the command to pull the model.

Create a .env File
Generate an API key from: https://sapling.ai/api_settings
Create a .env file in the project root. Add the API key in the following format:
SAPLING_PRIVATE_API_KEY=your_api_key

# Run the Application
To start the app, run the following command in the terminal:
streamlit run main.py
