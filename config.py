from autogen import config_list_from_json
import os

# 1. Define a Custom Name for your Project
# project_name = 'Pandas_Select'
project_name = 'Pandas_Select_GPT3_guided'
use_gui = False  # TODO: GUI not fully working yet
suppress_autogen_conversation = True
suppress_misc_info = True

# 2. Location where to paste your Project for enhancing readability (don't change)
# Define the base path as the current working directory
base_path = os.getcwd()  # Retrieve current Working Directory
base_project = os.path.join(base_path, project_name)

# 3. Destination Directory where a copy of the Base Project is stored and also the refactored Project with its metrics (don't change)
edit_project_directory = os.path.join(base_path, 'your_project_edit', project_name)
print(edit_project_directory)
edit_unreadable_project = os.path.join(edit_project_directory, 'unreadable_code/')
edit_readable_project = os.path.join(edit_project_directory, 'readable_code/')
edit_original_project_bu = os.path.join(edit_project_directory, 'original_code_bu/')
edit_readability_rating_project = os.path.join(edit_project_directory, 'readability_rating/')
edit_replies_project = os.path.join(edit_project_directory, 'editor_replies/')

# 1.1 Choose a custom name for the current run ==================================================================================
readability_check_and_show_improvements = True
run_through_autogen = True

# 1.2 LLM-Model Definition ======================================================================================================
# 1.2.1 Load the whole dictionary with all Configurations
oai_api_key = ''
llm_model = 'gpt-3.5-turbo-0125'

if oai_api_key is None:
    config_list_all = config_list_from_json(env_or_file='llm_configs/OAI_CONFIG_LIST_ALL')
    # 1.2.2 Assign individual variables to each LLM
    # [0]  # Local LLM (for use with LM Studio on your PC)
    # [1]  # GPT 3.5
    # [2]  # GPT 3.5 (with functions)
    # [3]  # GPT 4
    # [4]  # GPT 4 (with functions)
    config_list_llm_for_autogen = config_list_all[1]  # Chose the number inside the '[]' for the corresponding LLM
else:
    config_list_llm_for_autogen = {
            "seed": 42,
            "config_list": [
                {
                    "model": llm_model,
                    "api_key": oai_api_key
                }
            ],
            "temperature": 0}

# 2.1 Directories of files to be modified =======================================================================================
# For all these definitions below the order is important
directories_that_contain_files_to_be_modified = ['E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/pandas/core/sorting.py',
                                                 'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/torch/fx/passes/pass_manager.py']
# directories_that_contain_test_files = []
directories_that_contain_test_files = ['E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/pandas/tests/indexes/multi/test_sorting.py',
                                       'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/torch/fx/passes/tests/test_pass_manager.py']
