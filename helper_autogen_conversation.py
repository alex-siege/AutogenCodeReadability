from helper_global import *
from config_user import *
import itertools
from helper_autogen_conversable_agent import CustomConversableAgent


def autogen_conversation(reply_save_path, iterations_counter, unreadable_code, partial_script):
    # AutoGen Network for Operation ==============================================================================================

    # Creating Editor
    if partial_script:
        system_message_editor = read_txt('system_messages/system_message_editor_partial.txt')
    else:
        system_message_editor = read_txt('system_messages/system_message_editor.txt')

    editor = CustomConversableAgent(
        name='Editor',
        system_message=system_message_editor,
        code_execution_config=False,
        is_termination_msg=is_termination_msg,
        llm_config=config_list_llm_for_autogen,
        function_map={"refactor_method_name": refactor_method_name},
        reply_save_path=reply_save_path,
        max_consecutive_auto_reply=5,
        human_input_mode='NEVER'
    )

    # Creating Reviewer
    reviewer = CustomConversableAgent(
        name='Reviewer',
        system_message=read_txt('system_messages/system_message_reviewer.txt'),
        code_execution_config=False,
        is_termination_msg=is_termination_msg,
        llm_config=config_list_llm_for_autogen,
        function_map={"refactor_method_name": refactor_method_name},
        max_consecutive_auto_reply=5,
        human_input_mode='NEVER'
    )

    # In case of subsequent attempts to make the code more readable such that the LLM does not 'feel' forced to change up the code in case its already been changed
    system_message_initiate_chat = read_txt('system_messages/system_message_reviewer_initiate_chat.txt') \
        if iterations_counter > 1 else ''

    reviewer.initiate_chat(
        recipient=editor,
        message=system_message_initiate_chat + 'Here is the code to be made more readable:\n' + unreadable_code,
        silent=suppress_autogen_conversation
    )

    time.sleep(1)  # to be kind to the API
