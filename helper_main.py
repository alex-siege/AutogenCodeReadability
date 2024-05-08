from helper_autogen_conversation import *

UStutt_Dark_Blue = '#00519e'
UStutt_Light_Blue = '#00beff'
UStutt_Dark_Green = '#3b8c7a'
UStutt_Light_Green = '#7d9b65'
UStutt_Dark_Orange = '#e4af34'
UStutt_Light_Orange = '#ecda91'
UStutt_Dark_Red = '#950003'
UStutt_Light_Red = '#ff0005'


def long_operation(unreadable_code, file_name, iterations_counter, file_dir, test_file_dir, test_file_name, file_path):
    # Splitting the py-file up into more manageable chunks
    functions, classes, intermediate_code = split_in_segments(unreadable_code)
    previously_unreadable_code = ''

    # Iterating over the individual functions and classes:
    for code_segment in functions + classes + intermediate_code:
        # Only dealing with code segments that are reasonably sized
        if 1200 > len(code_segment.split('\n')) > 6:

            # Processing each segment with the LLM (function or class) and saving it temporarily into a separate file
            autogen_conversation(
                reply_save_path=config_user.edit_replies_project + file_name,
                iterations_counter=iterations_counter,
                unreadable_code=code_segment,
                partial_script=True
            )

            # Check if the altered code-segment is compilable
            if os.path.exists(config_user.edit_replies_project + file_name):
                with open(config_user.edit_replies_project + file_name, 'r') as altered_file:
                    altered_code_segment = altered_file.read()
                try:
                    # Try to compile the code to check for syntax errors
                    compile(altered_code_segment, file_name, 'exec')
                    print_colored(f"Compilation success in file {file_name}", 'green') if not suppress_miscellaneous_info else None
                except SyntaxError as e:
                    print_colored(f"Syntax error in file {file_name}: {e}", 'magenta') if not suppress_miscellaneous_info else None
                    print_colored('The following Code Segment did not pass the unit_test:\n\n'
                                  + str(altered_code_segment), 'red') if not suppress_miscellaneous_info else None
                else:
                    # If no compilation errors - replace the original segment inside unreadable_code
                    if len(altered_code_segment) > 1:
                        # Making a copy of the currently unedited full code prior to replacing the edited code segment for later fall-back
                        previously_unreadable_code = unreadable_code
                        # Replacing code
                        unreadable_code = replace_code_segment(unreadable_code, code_segment, altered_code_segment)
                        # Saving the currently modified file to eventually test it further (if test files are provided)
                        with open(file_path, 'w') as file:
                            file.write(unreadable_code)

                # Making sure to delete used code from temporary replies in order to not use it accidentally for anything else
                delete_directory(config_user.edit_replies_project)
                altered_code_segment = ''

                # Running unit_test now to validate code (at this stage we already replaced the code segment of the file we're modifying):
                if test_file_name is not None:
                    test_result = run_pytest_and_report(
                        test_file_directory=os.path.dirname(test_file_dir),
                        test_file_name=test_file_name,
                        mod_file_directory=config_user.edit_readable_project,
                        mod_file_name=file_name,
                        original_file_directory=config_user.edit_original_project_bu,
                        file_to_be_tested_directory=os.path.dirname(file_dir))

                    if test_result == 'All tests passed.':
                        # Do nothing basically
                        print(test_result) if not suppress_miscellaneous_info else None
                    elif test_result == 'Some tests failed.':
                        # Revert changes (take the code before we replaced the modified code segment and save it in place of the currently modified file)
                        print(test_result) if not suppress_miscellaneous_info else None
                        with open(file_path, 'w') as file:
                            file.write(previously_unreadable_code)
                        unreadable_code = previously_unreadable_code
                    else:
                        # Something went wrong very unexpectedly
                        raise ValueError(f"Unexpected test result: {test_result}")


def short_operation(unreadable_code, file_name, iterations_counter, file_dir, test_file_dir, test_file_name, file_path):
    print('The current .py-file does not contain multiple methods or classes.')
    previously_unreadable_code = ''
    code_segment = unreadable_code
    autogen_conversation(
        reply_save_path=os.path.join(config_user.edit_readable_project, file_name),
        iterations_counter=iterations_counter,
        unreadable_code=code_segment,
        partial_script=False
    ) if len(unreadable_code.split('\n')) < 1200 else None

    # Check if the altered code-segment is compilable
    if os.path.exists(config_user.edit_replies_project + file_name):
        with open(config_user.edit_replies_project + file_name, 'r') as altered_file:
            altered_code = altered_file.read()
        try:
            # Try to compile the code to check for syntax errors
            compile(altered_code, file_name, 'exec')
            print_colored(f"Compilation success in file {file_name}", 'green')
        except SyntaxError as e:
            print_colored(f"Syntax error in file {file_name}: {e}", 'magenta')
            print_colored('The following Code Segment did not pass the unit_test:\n\n'
                          + str(altered_code), 'red')
        else:
            # If no compilation errors - replace the original segment inside unreadable_code
            if len(altered_code) > 1:
                # Making a copy of the currently unedited full code prior to replacing the edited code segment for later fall-back
                previously_unreadable_code = unreadable_code
                # Replacing code
                unreadable_code = altered_code
                # Saving the currently modified file to eventually test it further (if test files are provided)
                with open(file_path, 'w') as file:
                    file.write(unreadable_code)

        # Making sure to delete used code from temporary replies in order to not use it accidentally for anything else
        delete_directory(config_user.edit_replies_project)
        altered_code = ''

        # Running unit_test now to validate code (at this stage we already replaced the code segment of the file we're modifying):
        if test_file_name is not None:
            test_result = run_pytest_and_report(
                test_file_directory=os.path.dirname(test_file_dir),
                test_file_name=test_file_name,
                mod_file_directory=config_user.edit_readable_project,
                mod_file_name=file_name,
                original_file_directory=config_user.edit_original_project_bu,
                file_to_be_tested_directory=os.path.dirname(file_dir))

            if test_result == 'All tests passed.':
                # Do nothing basically
                print(test_result) if not suppress_miscellaneous_info else None
            elif test_result == 'Some tests failed.':
                # Revert changes (take the code before we replaced the modified code segment and save it in place of the currently modified file)
                print(test_result) if not suppress_miscellaneous_info else None
                with open(file_path, 'w') as file:
                    file.write(previously_unreadable_code)
                unreadable_code = previously_unreadable_code
            else:
                # Something went wrong very unexpectedly
                raise ValueError(f"Unexpected test result: {test_result}")
