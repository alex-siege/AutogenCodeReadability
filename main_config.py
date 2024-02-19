from autogen import config_list_from_json
import os
from helper_config import *

# 1. Project Set-Up (Settings for the User) =====================================================================================
# 1.1 Chose Project Name and set verbosity of the script (amount of output in the console)
project_name = format_project_name('Chose_a_Project_Name')  # (avoid using blank lines and special characters)
use_gui = False  # TODO: GUI not fully working yet
suppress_autogen_conversation = False  # select True if you are not interested in seeing the conversation between the agents
suppress_miscellaneous_info = False  # select True if you want to see less cluttering in the console output

# 1.2 Chose a LLM-Model ()
oai_api_key = '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'  # Your OpenAI API Key goes inside these brackets (in case you did not set it up as an environment variable)
available_llm_models = ['', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-preview']
user_specified_llm = 1  # select the specific ChatGPT version to use from the 'available_llm_models' list where:
# 0 = local-llm(with LM-Studio),  1 = gpt-3.5-turbo-0125,  2 = gpt-4-turbo-preview

# 1.3 Chose the path of the files that are subject for change
# 1.3.1 Chose the files that are subject for change e.g:
directories_that_contain_files_to_be_modified = \
    ['E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/pandas/core/sorting.py',
     'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/torch/fx/passes/pass_manager.py']

# 1.3.2 Chose the files that are used as test-files in order to make sure that the changes in 1.3.1 are valid
directories_that_contain_test_files = []  # You can leave the list empty (to skip the verification after each alteration) ...

# ... or provide the corresponding test-file paths in the same order as the files to be altered in 1.3.1 e.g:
directories_that_contain_test_files = \
    ['E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/pandas/tests/indexes/multi/test_sorting.py',
     'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/torch/fx/passes/tests/test_pass_manager.py']
# Note: The test files should already be set up in the corresponding .venv such that when you run them in their location
# they are able to execute the pytest properly with no errors


# 2. Project Set-Up (Variables for the environment - do not change unless you know what you do) =================================
# 2.1 Destination Directory where a copy of the Base Project is stored and also the refactored Project with its metrics
base_path = os.getcwd()  # Retrieve current Working Directory
base_project = os.path.join(base_path, project_name)  # Set base path
edit_project_directory = os.path.join(base_path, 'your_project_edit', project_name)
print(edit_project_directory)
edit_unreadable_project = os.path.join(edit_project_directory, 'unreadable_code/')
edit_readable_project = os.path.join(edit_project_directory, 'readable_code/')
edit_original_project_bu = os.path.join(edit_project_directory, 'original_code_bu/')
edit_readability_rating_project = os.path.join(edit_project_directory, 'readability_rating/')
edit_replies_project = os.path.join(edit_project_directory, 'editor_replies/')

# 2.2 Set-Up the running configuration for the LLM
if oai_api_key is None or user_specified_llm == 0:
    # either retrieving from the 'OAI_CONFIG_LIST' in the 'llm_configs' folder ...
    config_list_all = config_list_from_json(env_or_file='llm_configs/OAI_CONFIG_LIST_ALL')
    config_list_llm_for_autogen = config_list_all[user_specified_llm]
else:
    # ... or setting it up locally by using the specified parameters from 1.2
    llm_model = available_llm_models[user_specified_llm]
    config_list_llm_for_autogen = {
        "seed": 42,
        "cache_seed": 42,
        "config_list": [
            {
                "model": llm_model,
                "api_key": oai_api_key
            }
        ],
        "temperature": 0}
