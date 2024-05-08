# Python Readability Enhancer with Test Validation

This Python-based tool is part of a Master Thesis and designed to automate the enhancement of readability in Python-Files, 
optionally incorporating unit tests to ensure code integrity post-enhancement.
By applying readability improvements with the help of a LLM using Autogen as an Orchestrator, it refines Python scripts to make them cleaner and more maintainable. 
The tool (optionally) not only processes individual Python files but also pairs them with their respective unit tests 
to verify that readability optimizations do not introduce errors or alter the intended functionality of the code. 

<table>
  <tr>
    <td align="center" valign="middle"><img src="images/logo.png" alt="Project Logo" width="60"></td>
    <td><p style="font-size:17px; margin-left:30px; margin-right:30px;"><strong><em>
    For a quick overview please check out the simplified Flowchart at the end of this File.
    </em></strong></p></td>
  </tr>
</table>


## Cloning Repository and setting up Environment:
- clone this repository to your local drive
- create a new .venv with python (Version: >= 3.8, <3.13) as the base interpreter. Here's a short [Video Tutorial](https://youtu.be/Ohu_MP8fpqM)
- run the following command in the terminal within your IDE: `python -m pip install -r requirements.txt`

## Project Set-Up

### Chose Project Name and set verbosity of the script (amount of output in the console) - `config_user.py`
Inside this configuration-file only edit the variables that are listed in `1. Project Set-Up (Settings for the User)`
- set the `use_gui` variable to `True` in order to have the GUI displayed (if you decide to use the GUI, all the mandatory variables listed below 
can be set there)
- select a `project_name` (it will later be used to create a directory inside `your_projects_edit` - folder to store the edited files as well 
as some temporary ones)
- set the `suppress_autogen_conversation` and `suppress_miscellaneous_info` to `True` in order to have less output in the console
- the seed is used by AutoGen in order to have a reference to store previous conversations in its `.cache` folder
- set the temperature for the LLM
- set the `oai_api_key` variable either directly in code or leave it empty and instead set it up as described in Chapter 2.2
- select a LLM-Model with the `user_specified_llm` variable by setting it to either 0, 1 or 2
- provide the directories to the .py-files you want to be processed by this script inside the 
`directories_that_contain_files_to_be_modified` list
- (optionally) provide the directories to the test-.py-files you want to be used as unit tests for the verification 
of the changes made inside the 'directories_that_contain_test_files' list (Note: the order has to be the same as in the previous list)

### Optional Step: Set 'oai_api_key' as environment variable - `Windows Operating System`
This step is only necessary if the API Key has not been set directly in code (left blank). 
The benefit of setting the API Key as an environment variable is that you will not accidentally commit to GitHub with the API Key still
inside the Code for others to see.
- Right-Click on `This PC` or `My Computer` on your desktop or inside File Explorer
- Select `Properties`
- Click on `Advanced system settings` to open the System Properties window
- In the System Properties window, go to the `Advanced` tab
- Click on the `Environment Variables` button
- Under `System variables` (for all users) or `User variables` (for your user only), click `New`
- Create a new environment variable named `OPENAI_API_KEY`
- Enter your actual OpenAI API key as the variable value
- Click `OK`

## Some Notes and Troubleshooting
### How to get an OpenAI API Key?
- Log into your OpenAI account and visit: [OpenAI API Keys](https://platform.openai.com/account/api-keys)
- As of 2024, the usage of API Services is not free. Make sure to deposit funds (at least $1) into your OpenAI account

### Important Security Note
- Keep your API keys and other sensitive information secure.
- Avoid exposing them in places where unauthorized users might access them.
- When distributing your code, ensure you're not including these keys directly in the source-code or in any files that might be shared publicly.

### Unit tests are not executing
- Currently only the `pytest` environment is supported.
- Make sure that the unit test is executable in its set up environment (it is possible to run the unit test file and get proper results)

# Simplified Flowchart of the main.py-file
<p align="left" style="margin-left: 20px;">
  <img src="images/Flowchart.png" alt="Project Logo" width="850">
</p>
