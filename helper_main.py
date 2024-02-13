from autogen_framework import *


def long_method_operation(unreadable_code, file_name, iterations_counter, file_dir, test_file_dir, test_file_name, file_path):
    # Splitting the py-file up into more manageable chunks
    functions, classes, intermediate_code = extract_functions_classes_and_intermediate_code(unreadable_code)
    previously_unreadable_code = ''

    # Iterating over the individual functions and classes:
    for code_segment in functions + classes:
        # Only dealing with code segments that are reasonably sized
        if 150 > len(code_segment.split('\n')) > 6:

            # Processing each segment with the LLM (function or class) and saving it temporarily into a separate file
            run_autogen_conversation(
                reply_save_path=edit_replies_project + file_name,
                iterations_counter=iterations_counter,
                unreadable_code=code_segment,
                partial_script=True
            )

            # Check if the altered code-segment is compilable
            if os.path.exists(edit_replies_project + file_name):
                with open(edit_replies_project + file_name, 'r') as altered_file:
                    altered_code_segment = altered_file.read()
                try:
                    # Try to compile the code to check for syntax errors
                    compile(altered_code_segment, file_name, 'exec')
                    print_colored(f"Compilation success in file {file_name}", 'green')
                except SyntaxError as e:
                    print_colored(f"Syntax error in file {file_name}: {e}", 'magenta')
                    print_colored('The following Code Segment did not pass the unit_test:\n\n'
                                  + str(altered_code_segment), 'red')
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
                delete_directory(edit_replies_project)
                altered_code_segment = ''

                # Running unit_test now to validate code (at this stage we already replaced the code segment of the file we're modifying):
                if test_file_name is not None:
                    test_result = run_pytest_and_report(
                        test_file_directory=os.path.dirname(test_file_dir),
                        test_file_name=test_file_name,
                        mod_file_directory=edit_readable_project,
                        mod_file_name=file_name,
                        original_file_directory=edit_original_project_bu,
                        file_to_be_tested_directory=os.path.dirname(file_dir))

                    if test_result == 'All tests passed.':
                        # Do nothing basically
                        print(test_result)
                    elif test_result == 'Some tests failed.':
                        # Revert changes (take the code before we replaced the modified code segment and save it in place of the currently modified file)
                        print(test_result)
                        with open(file_path, 'w') as file:
                            file.write(previously_unreadable_code)
                        unreadable_code = previously_unreadable_code
                    else:
                        # Something went wrong very unexpectedly
                        raise ValueError(f"Unexpected test result: {test_result}")


def short_method_operation(unreadable_code, file_name, iterations_counter, file_dir, test_file_dir, test_file_name, file_path):
    print('The current .py-file does not contain multiple methods or classes.')
    previously_unreadable_code = ''
    code_segment = unreadable_code
    run_autogen_conversation(
        reply_save_path=os.path.join(edit_readable_project, file_name),
        iterations_counter=iterations_counter,
        unreadable_code=code_segment,
        partial_script=False
    ) if len(unreadable_code.split('\n')) < 200 else None

    # Check if the altered code-segment is compilable
    if os.path.exists(edit_replies_project + file_name):
        with open(edit_replies_project + file_name, 'r') as altered_file:
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
        delete_directory(edit_replies_project)
        altered_code = ''

        # Running unit_test now to validate code (at this stage we already replaced the code segment of the file we're modifying):
        if test_file_name is not None:
            test_result = run_pytest_and_report(
                test_file_directory=os.path.dirname(test_file_dir),
                test_file_name=test_file_name,
                mod_file_directory=edit_readable_project,
                mod_file_name=file_name,
                original_file_directory=edit_original_project_bu,
                file_to_be_tested_directory=os.path.dirname(file_dir))

            if test_result == 'All tests passed.':
                # Do nothing basically
                print(test_result)
            elif test_result == 'Some tests failed.':
                # Revert changes (take the code before we replaced the modified code segment and save it in place of the currently modified file)
                print(test_result)
                with open(file_path, 'w') as file:
                    file.write(previously_unreadable_code)
                unreadable_code = previously_unreadable_code
            else:
                # Something went wrong very unexpectedly
                raise ValueError(f"Unexpected test result: {test_result}")