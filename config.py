from autogen import config_list_from_json

# 1. Define a Custom Name for your Project
# project_name = 'Pandas_Select'
project_name = 'Pandas_Select_GPT3_guided'
use_gui = False

# 2. Location where to paste your Project for enhancing readability (don't change)
base_project = 'E:/AutogenCodeReadability/your_project/'

# 3. Destination Directory where a copy of the Base Project is stored and also the refactored Project with its metrics (don't change)
edit_project_directory = 'E:/AutogenCodeReadability/your_project_edit/' + project_name
edit_unreadable_project = 'E:/AutogenCodeReadability/your_project_edit/' + project_name + '/unreadable_code/'
edit_readable_project = 'E:/AutogenCodeReadability/your_project_edit/' + project_name + '/readable_code/'
edit_original_project_bu = 'E:/AutogenCodeReadability/your_project_edit/' + project_name + '/original_code_bu/'
edit_readability_rating_project = 'E:/AutogenCodeReadability/your_project_edit/' + project_name + '/readability_rating/'
edit_replies_project = 'E:/AutogenCodeReadability/your_project_edit/' + project_name + '/editor_replies/'

# 1.1 Choose a custom name for the current run ==================================================================================
readability_check_and_show_improvements = True
run_through_autogen = True

# 1.2 LLM-Model Definition ======================================================================================================
# 1.2.1 Load the whole dictionary with all Configurations
oai_api_key = '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'
llm_model = 'gpt-3.5-turbo-0125'

if oai_api_key is None:
    config_list_gpt3_all = config_list_from_json(env_or_file='llm_configs/OAI_CONFIG_LIST_ALL')
    # 1.2.2 Assign individual variables to each LLM
    config_list_custom_llm = config_list_gpt3_all[0]  # Local LLM (for use with LM Studio)
    config_list_gpt3_turbo = config_list_gpt3_all[1]  # GPT 3.5
    config_list_gpt3_tools = config_list_gpt3_all[2]  # GPT 3.5 (with functions)
    config_list_gpt4_turbo = config_list_gpt3_all[3]  # GPT 4
    config_list_gpt4_tools = config_list_gpt3_all[4]  # GPT 4 (with functions)
else:
    config_list = [
        {
            "model": llm_model,
            "api_key": oai_api_key
        }
    ]
    config_list_gpt3_turbo = [
        {
            "seed": 42,
            "config_list": config_list,
            "temperature": 0
        }
    ]


# 2.1 Directories of files to be modified =======================================================================================
test_file_pandas_dir = 'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/pandas/tests/indexes/multi/test_sorting.py'
file_pandas_dir = 'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/pandas/core/sorting.py'  # 748 LOC

test_file_torch_dir = 'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/torch/fx/passes/tests/test_pass_manager.py'
file_torch_dir = 'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/torch/fx/passes/pass_manager.py'  # 257 LOC

test_file_flask_dir = 'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/flask/tests/test_helpers.py'
file_flask_dir = 'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/flask/helpers.py'  # 621 LOC

test_file_matplotlib_dir = 'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/matplotlib/tests/test_agg.py'
file_matplotlib_dir = 'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/matplotlib/backends/backend_agg.py'  # 544 LOC

test_file_requests_dir = 'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/requests/tests/test_utils.py'
file_requests_dir = 'E:/AutoGenCodeReadabilityProjects/pandas-main/pythonProject/.venv/Lib/site-packages/requests/utils.py'  # 1024 LOC

# For all these definitions below the order is important
directories_that_contain_files_to_be_modified = [file_pandas_dir, file_torch_dir]
directories_that_contain_test_files = []
# directories_that_contain_test_files = [test_file_pandas_dir, test_file_torch_dir]

