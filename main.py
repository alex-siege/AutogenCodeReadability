from helper_main import *
from gui import *


def main_loop():
    # 1. Setup ==================================================================================================================
    # 1.1 Clean Up (Removing all existing files associated with the current project from previous runs)
    delete_directory(edit_project_directory)

    # 1.2 Choose if the previous replies from the LLM shall be reused or new ones generated
    keep_or_clear_cache() if not use_gui else None

    # 2. Iterating over provided files (and test-files including the unit-tests) ================================================
    for (file_dir, test_file_dir) in itertools.zip_longest(
            directories_that_contain_files_to_be_modified,
            directories_that_contain_test_files):
        file_name = os.path.basename(file_dir)
        test_file_name = os.path.basename(test_file_dir) if test_file_dir is not None else None

        # 2.1 Make sure we're dealing with py-files:
        if not file_name.endswith('.py') and (not test_file_name.endswith('.py') or test_file_name is not None):
            raise ValueError(f"One of the files does not have a '.py' extension: {file_name}, {test_file_name}")

        # 2.2 In case a test-file is provided, use it to verify if it is testing properly:
        if test_file_name is not None:
            test_result = run_pytest_and_report(os.path.dirname(test_file_dir), test_file_name)

            # 2.2.1 Check results from the unit-tests:
            if test_result == "Some tests failed.":
                raise ValueError(f"The test: {test_file_name} could not be executed in {test_file_dir}. "
                                 f"Make sure that the .venv where the pytest is located at, is set up properly.")
            else:
                print_colored('Initial Pytest has passed. Processing ' + str(file_name), 'green')

        # 2.3 Create the working environment with all the necessary directories:
        copy_file_to(os.path.dirname(file_dir), edit_unreadable_project, file_name)
        copy_file_to(os.path.dirname(file_dir), edit_readable_project, file_name)
        if test_file_name is not None:
            copy_file_to(os.path.dirname(file_dir), edit_original_project_bu, file_name)

        # 2.4 Perform an initial Readability Check:
        current_mi = 0.0
        check_readability_single(
            edit_readable_project, edit_readability_rating_project, 'Base', file_name)

        # 2.5 Initialize an array with current readability-values for checking on improvement inside while-loop later on
        previous_mi = retrieve_readability_index_from_csv(
            edit_readability_rating_project, file_name, 'MI')

        print('Previous and Current MI: ' + str(previous_mi) + ' ' + str(current_mi)) if not suppress_misc_info else None

        # 2.6 Set up the while-loop that iterates over the
        # Runs at least once and as long as there's Improvement of the Maintainability Index in any of the files
        iterations_counter = 0
        while iterations_counter <= 2 or is_improvement(previous_mi, current_mi):
            iterations_counter += 1

            file_path = edit_readable_project + file_name

            # 2.6.1
            previously_unreadable_code = ''
            with open(file_path, 'r') as file:
                unreadable_code = file.read()

            # 2.6.2 Distinguishing between too long py-files and those that can be fed into the LLM as whole
            if has_multiple_methods_or_classes(unreadable_code) and len(unreadable_code.split('\n')) > 100:
                long_method_operation(
                    unreadable_code, file_name, iterations_counter, file_dir, test_file_dir, test_file_name, file_path)

            # 2.6.3 Feeding whole py-file into LLM as its length does not exceed the LLMs capacity
            else:
                short_method_operation(
                    unreadable_code, file_name, iterations_counter, file_dir, test_file_dir, test_file_name, file_path)

            # 2.6.4 Perform Readability Check after Operation
            autoformat_py_files(edit_readable_project)
            if iterations_counter > 1:
                previous_mi = current_mi
            check_readability_single(
                edit_readable_project, edit_readability_rating_project, 'Iter. ' + str(iterations_counter), file_name)

            # 2.6.5 Update the readability measurement in order to decide if to continue running the while-loop
            current_mi = retrieve_readability_index_from_csv(
                edit_readability_rating_project, file_name, 'MI')
            print('Previous and Current MI After: ' + str(previous_mi) + ' ' + str(current_mi)) if not suppress_misc_info else None

    # 3. Visualizing Results ====================================================================================================
    plot_readability_metrics(
        edit_readability_rating_project, ['CC', 'MI'], True)


if __name__ == '__main__':
    if use_gui:
        launch_gui()
    else:
        main_loop()
