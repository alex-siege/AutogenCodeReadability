# Project Setup Instructions

## 1. Installing Required Packages

To install all the custom packages needed for this project, first create a new project with python 3.11 
and run the following command in the terminal within your IDE:
`pip install -r requirements.txt`

## 2. Setting Up Your OpenAI API Key

### 2.1 Open System Properties
- Right-click on 'This PC' or 'My Computer' on your desktop or in File Explorer.
- Select 'Properties'.
- Click on 'Advanced system settings' to open the System Properties window.

### 2.2 Edit Environment Variables
- In the System Properties window, go to the 'Advanced' tab.
- Click on the 'Environment Variables' button.

### 2.3 Set the Environment Variable
- Under 'System variables' (for all users) or 'User variables' (for your user only), click 'New'.
- Create a new environment variable named `OPENAI_API_KEY`.
- Enter your actual OpenAI API key as the variable value.
- Click 'OK' to close each dialog.

### 2.4 Note on OpenAI API Key
- Ensure you have set up your Open AI API Key. Log into your OpenAI account and visit: [OpenAI API Keys](https://platform.openai.com/account/api-keys).
- As of 2023, the usage of API Services is not free. Make sure to deposit funds (at least $1) into your OpenAI account.

### 2.5 Optional: Restart Your System

## Important Security Note
- Keep your API keys and other sensitive information secure.
- Avoid exposing them in places where unauthorized users might access them.
- When distributing your code, ensure you're not including these keys directly in the code or in any files that might be shared publicly.
