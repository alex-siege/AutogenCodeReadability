import openai
import keyboard
import autogen
import glob, os, csv
from rope.base.project import Project
from radon.complexity import cc_visit
from radon.metrics import mi_visit, h_visit
from radon.raw import analyze as raw_analyze
from pathlib import Path
import pandas as pd
from config import *
import shutil
import time
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import ast
import re
import subprocess
import nbformat
import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider
import tiktoken
import subprocess
import os
import sys


def run_pytest_and_report(test_file_directory, test_file_name, mod_file_directory=None, mod_file_name=None, original_file_directory=None, file_to_be_tested_directory=None):
    """
    Runs pytest on a specified test file against a modified file, if provided, and provides a report on the test results.

    This function performs the following steps:
    1. If provided, backs up the original file to be tested.
    2. If provided, replaces the original file with a modified version.
    3. Runs pytest on the specified test file.
    4. Outputs the test results.
    5. If steps 1 and 2 were performed, restores the original file from backup, regardless of test outcomes or errors.

    Only 'test_file_directory' and 'test_file_name' are mandatory. Other parameters are optional.

    Parameters:
    test_file_directory (str): Directory of the pytest file.
    test_file_name (str): Name of the pytest file.
    mod_file_directory (str, optional): Directory of the modified file to be tested.
    mod_file_name (str, optional): Name of the modified file.
    original_file_directory (str, optional): Directory where the backup of the original file will be stored.
    file_to_be_tested_directory (str, optional): Directory of the original file to be tested.

    Returns:
    str: A message indicating whether all tests passed or some failed.
    """

    # Construct the full path to the test file
    test_file_path = test_file_directory + '/' + test_file_name

    # Check if the optional parameters are provided
    if all([mod_file_directory, mod_file_name, original_file_directory, file_to_be_tested_directory]):
        original_file_path = file_to_be_tested_directory + '/' + mod_file_name
        backup_file_path = original_file_directory + '/' + mod_file_name
        modified_file_path = mod_file_directory + '/' + mod_file_name

        # Ensure the backup directory exists, create it if not
        Path(backup_file_path).parent.mkdir(parents=True, exist_ok=True)

        # Backup the original file
        shutil.copy(original_file_path, backup_file_path)

        try:
            # Replace the original file with the modified file
            shutil.copy(modified_file_path, original_file_path)

            # Run pytest
            return run_and_report_pytest(test_file_path)

        finally:
            # Restore the original file from backup
            shutil.copy(backup_file_path, original_file_path)
    else:
        # Run pytest without file operations
        return run_and_report_pytest(test_file_path)


def run_and_report_pytest(test_file_path):
    """ Run pytest and report results. """
    result = subprocess.run(['pytest', str(test_file_path)], capture_output=True, text=True)
    print(result.stdout) if not suppress_misc_info else None
    print(result.stderr) if not suppress_misc_info else None
    return "All tests passed." if result.returncode == 0 else "Some tests failed."


def analyze_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Get Cyclomatic Complexity results
    cc_results = cc_visit(content)
    # Get raw metrics (including LOC)
    raw_results = raw_analyze(content)
    # Get Maintainability Index results
    mi_results = mi_visit(content, multi=True)
    # Halstead metrics
    halstead_results = h_visit(content)

    return cc_results, raw_results, mi_results, halstead_results


def print_readability_of_given_folder(folder_path, path_for_save, time):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                cc_results, raw_results, mi_results, halstead_results = analyze_file(file_path)

                # Specify the directory and the file name (without using 'time' in the file name)
                file_name = file[:-3] + '.csv'
                full_path = os.path.join(path_for_save, file_name)

                # Create directory if it doesn't exist
                directory_path = os.path.dirname(full_path)
                os.makedirs(directory_path, exist_ok=True)

                # Collect metrics for this file, prepend 'time' at the beginning
                metrics = [time] + [
                    cc_results[0].complexity if cc_results else 1,  # Cyclomatic Complexity (CC)
                    # Add raw metrics
                    raw_results.loc,
                    raw_results.lloc,
                    raw_results.sloc,
                    raw_results.comments,
                    raw_results.multi,
                    raw_results.blank,
                    raw_results.single_comments,
                    # Add Halstead metrics
                    halstead_results.total.h1,
                    halstead_results.total.h2,
                    halstead_results.total.N1,
                    halstead_results.total.N2,
                    halstead_results.total.vocabulary,
                    halstead_results.total.length,
                    round(halstead_results.total.calculated_length, 2),
                    round(halstead_results.total.volume, 2),
                    round(halstead_results.total.difficulty, 2),
                    round(halstead_results.total.effort, 2),
                    round(halstead_results.total.time, 2),
                    round(halstead_results.total.bugs, 2),
                    # Add Maintainability Index (MI)
                    round(mi_results, 2)
                ]

                # Define headers, including 'Time' as the first header
                headers = ["Time", "CC", "LOC", "LLOC", "SLOC", "Comments", "Comment Blocks", "Blank",
                           "Single Comments", "H1", "H2", "N1", "N2", "Vocabulary", "Length",
                           "Calculated Length", "Volume", "Difficulty", "Effort", "Time", "Bugs", "MI"]

                # Create DataFrame for this file's metrics
                df = pd.DataFrame([metrics], columns=headers)

                # Check if the file exists to determine if headers should be written
                file_exists = os.path.isfile(full_path)

                # Save DataFrame to CSV (append if file exists, write headers only if creating new file)
                df.to_csv(full_path, mode='a', index=False, header=not file_exists)
                print(f"Metrics {'appended to' if file_exists else 'saved to'} {full_path}")


def check_readability(path_for_check, path_for_save, time_point):
    if path_for_check:
        folder_path = path_for_check
    else:
        folder_path = '/your_project/'
    # print("Press 'Enter' to check readability, or 'Space' to skip.")

    print("Checking readability...\n")
    print_readability_of_given_folder(folder_path, path_for_save, time_point)


def print_readability_of_given_folder_single(file_path, path_for_save, time):
    if file_path.endswith('.py'):
        cc_results, raw_results, mi_results, halstead_results = analyze_file(file_path)

        # Specify the directory and the file name (without using 'time' in the file name)
        file_name = os.path.basename(file_path)
        file_name_csv = file_name[:-3] + '.csv'
        full_path = os.path.join(path_for_save, file_name_csv)

        # Create directory if it doesn't exist
        directory_path = os.path.dirname(full_path)
        os.makedirs(directory_path, exist_ok=True)

        # Collect metrics for this file, prepend 'time' at the beginning
        metrics = [time] + [
            cc_results[0].complexity if cc_results else 1,  # Cyclomatic Complexity (CC)
            # Add raw metrics
            raw_results.loc,
            raw_results.lloc,
            raw_results.sloc,
            raw_results.comments,
            raw_results.multi,
            raw_results.blank,
            raw_results.single_comments,
            # Add Halstead metrics
            halstead_results.total.h1,
            halstead_results.total.h2,
            halstead_results.total.N1,
            halstead_results.total.N2,
            halstead_results.total.vocabulary,
            halstead_results.total.length,
            round(halstead_results.total.calculated_length, 2),
            round(halstead_results.total.volume, 2),
            round(halstead_results.total.difficulty, 2),
            round(halstead_results.total.effort, 2),
            round(halstead_results.total.time, 2),
            round(halstead_results.total.bugs, 2),
            # Add Maintainability Index (MI)
            round(mi_results, 2)
        ]

        # Define headers, including 'Time' as the first header
        headers = ["Time", "CC", "LOC", "LLOC", "SLOC", "Comments", "Comment Blocks", "Blank",
                   "Single Comments", "H1", "H2", "N1", "N2", "Vocabulary", "Length",
                   "Calculated Length", "Volume", "Difficulty", "Effort", "Time", "Bugs", "MI"]

        # Create DataFrame for this file's metrics
        df = pd.DataFrame([metrics], columns=headers)

        # Check if the file exists to determine if headers should be written
        file_exists = os.path.isfile(full_path)

        # Save DataFrame to CSV (append if file exists, write headers only if creating new file)
        df.to_csv(full_path, mode='a', index=False, header=not file_exists)
        print(f"Metrics {'appended to' if file_exists else 'saved to'} {full_path}")


def check_readability_single(path_for_check, path_for_save, time_point, file_name):
    if path_for_check:
        file_path = path_for_check + file_name
    else:
        file_path = '/your_project/' + file_name

    print("Checking readability...\n")
    print_readability_of_given_folder_single(file_path, path_for_save, time_point)


def print_selected_models(config_lists):
    valid_models = []

    for config_list in config_lists:
        if config_list and "model" in config_list[0]:
            valid_models.append(str(config_list[0]["model"]))

    if len(valid_models) == len(config_lists):
        print('\nSelected Models: ' + ', '.join(valid_models) + '\n')
    else:
        print('\nSelected Models: ' + ', '.join(valid_models) + '\nMake sure your models are selected properly...')

    return 0


def read_txt(directory):
    with open(directory, 'r') as file:
        file_read = file.read()
    return file_read


def keep_or_clear_cache():
    cache_path = ".\\.cache"

    # Check if the ".cache" folder exists
    if os.path.exists(cache_path):
        print_colored("Press 'Enter' to keep the cache, or 'Space' to clear it.", 'green')

        while True:
            if keyboard.is_pressed('enter'):
                print("Keeping the cache.\n")
                break
            elif keyboard.is_pressed('space'):  # Detect any key press
                # Clear the contents of the ".\\cache" folder
                for entry in os.listdir(cache_path):
                    entry_path = os.path.join(cache_path, entry)
                    if os.path.isfile(entry_path):
                        os.remove(entry_path)
                    elif os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
                print("Cache cleared.\n")
                break


def set_openai_api_key():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        openai.api_key = openai_api_key
    else:
        print("Error: OPENAI_API_KEY environment variable not set.")


def select_llm_version(llm_model):
    return autogen.config_list_from_json(
        env_or_file="configuration.json",
        filter_dict={"model": [llm_model],
                     },
    )


def strip_off_python_code(text):
    python_code = ""
    no_python_code = text
    start_marker = "```python"
    end_marker = "```"

    while start_marker in no_python_code:
        start = no_python_code.find(start_marker)
        end = no_python_code.find(end_marker, start + len(start_marker))

        if end != -1:
            python_code += no_python_code[start:end + len(end_marker)] + "\n"
            no_python_code = no_python_code[:start] + no_python_code[end + len(end_marker):]
        else:
            # Handle the case of unmatched start without a corresponding end
            break

    return python_code, no_python_code


def is_termination_msg(message):
    # Extract the content from the message
    content = message.get("content", "")

    # Return False immediately if content is None or empty
    if not content:
        return False

    # Strip off Python code blocks
    _, no_python_code = strip_off_python_code(content)

    # Check for termination strings in the remaining content
    termination_strings = ('TERMINATE', 'TERMINATE.', 'terminate', 'terminate.')
    return any(termination_string in no_python_code for termination_string in termination_strings)


def get_all_py_files(dict_path):
    """
    Reads all Python (.py) files in a given directory and stores their contents in a dictionary.

    Args:
    dict_path (str): The directory path where the .py files are located.

    Returns:
    dict: A dictionary where each key is the name of a .py file (without extension)
          and its corresponding value is the content of that file.
    """
    # Dictionary to store the contents of each file
    file_contents = {}

    # Iterate over all .py files in the directory
    for filepath in glob.glob(os.path.join(dict_path, '*.py')):
        # Extract the filename without extension
        filename = os.path.basename(filepath).split('.')[0]

        # Open and read the content of the file
        with open(filepath, 'r') as file:
            content = file.read()

        # Store the content in the dictionary with the filename as the key
        file_contents[filename] = content

    # Return the dictionary containing file contents
    return file_contents


def refactor_method_name(old_method_name, new_method_name):
    project = None
    try:
        directory = os.path.normpath(edit_readable_project)
        project = Project(directory)
        for file_name in os.listdir(directory):
            if file_name.endswith('.py'):
                file_path = os.path.normpath(os.path.join(directory, file_name))
                print(f"Processing file: {file_name}")

                with open(file_path, 'r') as file:
                    content = file.read()

                if old_method_name in content:
                    print(f"Found '{old_method_name}' in {file_name}")

                    # Refactor the method name
                    content = content.replace(old_method_name, new_method_name)
                    with open(file_path, 'w') as file:
                        file.write(content)

                    print(f"Refactored '{old_method_name}' to '{new_method_name}' in {file_name}")

        print("Refactoring complete.")
        return True
    except Exception as e:
        print(f"Refactoring failed: {e}")
        return False
    finally:
        if project is not None:
            project.close()


def copy_all_files_from_to(copy_from, copy_to, filetype=None):
    source_dir = Path(copy_from)
    copy_to = Path(copy_to)

    if not copy_to.exists():
        copy_to.mkdir(parents=True)

    for file in source_dir.iterdir():
        if file.is_file():
            # Check if a filetype is specified and matches the file's extension
            if filetype is None or file.suffix == '.' + filetype:
                shutil.copy(str(file), str(copy_to / file.name))


def copy_file_to(copy_from, copy_to, file_name, filetype=None):
    """
    Copies a specified file from one directory to another, with an optional filter for filetype.

    This function copies a file from the 'copy_from' directory to the 'copy_to' directory.
    If the 'filetype' parameter is provided, the function will copy the file only if it matches
    the specified filetype (extension). If the target directory does not exist, it will be created.

    Parameters:
    copy_from (str): The path of the source directory.
    copy_to (str): The path of the target directory.
    file_name (str): The name of the file to be copied.
    filetype (str, optional): The filetype (extension) to filter the file. Defaults to None.

    Returns:
    None

    Note: The function does not return any value or confirmation of success.
    """

    # Convert the source and target directory paths to Path objects
    source_dir = Path(copy_from)
    target_dir = Path(copy_to)

    # Ensure the target directory exists, create it if it doesn't
    if not target_dir.exists():
        target_dir.mkdir(parents=True)

    # Construct the full path of the source file
    source_file_path = source_dir / file_name

    # Check if the file exists and if a filetype is specified, whether it matches the file's extension
    if source_file_path.is_file() and (filetype is None or source_file_path.suffix == '.' + filetype):
        # Copy the file to the target directory
        shutil.copy(str(source_file_path), str(target_dir / file_name))


def retrieve_readability_index_from_csv(directory_for_retrieval, file_name, selected_indicator):
    """
    Retrieves a specific readability indicator from a CSV file.

    This function reads a CSV file specified by 'directory_for_retrieval' and a modified 'file_name',
    then returns the last value of the column named 'selected_indicator'. The 'file_name' is adjusted
    to ensure its extension is '.csv'. If the file is not found, or the indicator column is not in the CSV,
    the function raises an exception.

    Parameters:
    directory_for_retrieval (str): Directory where the CSV file is located.
    file_name (str): Initial name of the file, which will be modified to a '.csv' extension.
    selected_indicator (str): The indicator column to retrieve from the CSV file.

    Returns:
    float: The last value of the selected indicator in the CSV file.

    Raises:
    FileNotFoundError: If the CSV file is not found.
    ValueError: If the file is not a CSV or the selected indicator column is not in the CSV.
    """

    # Adjust the file name to have a .csv extension
    root, _ = os.path.splitext(file_name)
    file_name = root + '.csv'

    # Construct the full path to the CSV file
    file_path = os.path.join(directory_for_retrieval, file_name)

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_name} was not found.")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Check if the DataFrame contains the selected indicator
    if df.empty or selected_indicator not in df.columns:
        raise ValueError(f"The column '{selected_indicator}' was not found in the file {file_name}.")

    # Extract and return the last value from the selected indicator column
    last_value = df[selected_indicator].iloc[-1]
    return last_value


def retrieve_readability_index_from_csv_in_directory(directory_for_retrieval):
    previous_mi = []

    for filename in os.listdir(directory_for_retrieval):
        # Check if the file is a .csv file
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_for_retrieval, filename)

            # Read the CSV file with headers
            df = pd.read_csv(file_path)

            # Check if the DataFrame is not empty and has the 'MI' column
            if not df.empty and 'MI' in df.columns:
                # Extract the last value from the 'MI' column
                last_value = df['MI'].iloc[-1]

                # Append the value to the array
                previous_mi.append(last_value)

    return previous_mi


def is_improvement(previous_mi, current_mi):
    # Ensure both previous_mi and current_mi are lists
    if not isinstance(previous_mi, list):
        previous_mi = [previous_mi]
    if not isinstance(current_mi, list):
        current_mi = [current_mi]

    # Check if there's an improvement in any of the elements
    for prev, curr in zip(previous_mi, current_mi):
        if curr > prev:
            return True
    return False


def delete_directory(directory):
    """
    Deletes a specified directory along with all its contents.

    This function removes the directory specified by the `directory` parameter,
    including all files and subdirectories contained within it. If the directory
    does not exist, it prints a message indicating that the directory was not found.

    Parameters:
    directory (str): The path of the directory to be deleted.

    Returns:
    None

    Note: Use this function with caution, as the deletion is irreversible.
    """

    # Check if the directory exists
    if os.path.exists(directory):
        # Remove the directory and all its contents
        shutil.rmtree(directory)
        # Inform the user of successful removal
        print("Directory removed successfully.")
    else:
        # Inform the user if the directory does not exist
        print("Directory does not exist.")


# def plot_readability_metrics(directory, columns_to_plot=None, split_axes=False):
#     if not os.path.exists(directory):
#         print(f"Directory not found: {directory}")
#         return None
#
#     files = [f for f in os.listdir(directory) if f.endswith('.csv')]
#     plots_per_figure = 8  # Max number of plots per figure
#     num_files = len(files)
#     num_figures = (num_files + plots_per_figure - 1) // plots_per_figure
#
#     for fig_index in range(num_figures):
#         plt.figure(figsize=(12, 10))  # Adjust figure size as needed
#
#         for subplot_index in range(plots_per_figure):
#             file_index = fig_index * plots_per_figure + subplot_index
#             if file_index < num_files:
#                 file = files[file_index]
#                 df = pd.read_csv(os.path.join(directory, file))
#
#                 if 'Time' not in df.columns or df['Time'].empty:
#                     continue  # Skip if 'Time' column is missing or empty
#
#                 df['Time'] = pd.factorize(df['Time'])[0]
#
#                 if columns_to_plot:
#                     df = df[['Time'] + columns_to_plot]
#
#                 ax1 = plt.subplot(4, 2, subplot_index + 1)
#
#                 if df['Time'].notnull().sum() > 1:
#                     xnew = np.linspace(df['Time'].min(), df['Time'].max(), 300)
#                     initial_values = {}
#                     max_deviations = {}
#
#                     for col in columns_to_plot:
#                         initial_values[col] = df[col].iloc[0]
#                         max_deviation = max(df[col].max() - initial_values[col], initial_values[col] - df[col].min())
#                         max_deviations[col] = max_deviation
#
#                     overall_max_deviation = max(max_deviations.values())
#
#                     for i, col in enumerate(columns_to_plot):
#                         interp = PchipInterpolator(df['Time'], df[col])
#                         y_smooth = interp(xnew)
#
#                         # Make the second line dashed
#                         line_style = "--" if i == 1 else "-"
#
#                         ax1.plot(xnew, y_smooth, line_style, label=col)
#
#                         middle_value = initial_values[col]
#                         ax1.set_ylim([middle_value - overall_max_deviation, middle_value + overall_max_deviation])
#
#                     ax1.legend()
#
#                 plt.title(file[:-4])
#                 ax1.set_xlabel('Iterations')
#
#                 if fig_index == 0 and subplot_index == 0 and not split_axes:
#                     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=5)
#
#         plt.tight_layout()
#
#     plt.show()

def plot_readability_metrics(directory, columns_to_plot=None, split_axes=False):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return None
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    plots_per_figure = 8  # Max number of plots per figure
    num_files = len(files)

    # Determine the number of figures needed
    num_figures = (num_files + plots_per_figure - 1) // plots_per_figure

    for fig_index in range(num_figures):
        plt.figure(figsize=(12, 10))  # Adjust figure size as needed

        # Plot each file in the current figure
        for subplot_index in range(plots_per_figure):
            file_index = fig_index * plots_per_figure + subplot_index
            if file_index < num_files:
                file = files[file_index]
                df = pd.read_csv(os.path.join(directory, file))

                if 'Time' not in df.columns or df['Time'].empty:
                    continue  # Skip if 'Time' column is missing or empty

                # Map 'Time' labels to numeric values
                df['Time'] = pd.factorize(df['Time'])[0]

                if columns_to_plot:
                    # Select only the specified columns if provided
                    df = df[['Time'] + columns_to_plot]

                ax1 = plt.subplot(4, 2, subplot_index + 1)

                # Ensure there's enough data to plot
                if df['Time'].notnull().sum() > 1:
                    xnew = np.linspace(df['Time'].min(), df['Time'].max(), 300)

                    if split_axes and len(columns_to_plot) > 1:
                        # Pchip interpolation for the first column
                        interp = PchipInterpolator(df['Time'], df[columns_to_plot[0]])
                        y_smooth = interp(xnew)

                        # Plot the first group of columns on the left Y-axis
                        ax1.plot(xnew, y_smooth, color='b', label=columns_to_plot[0])
                        ax1.set_ylabel(columns_to_plot[0], color='b')
                        ax1.tick_params(axis='y', labelcolor='b')
                        ax1.legend(loc='upper left')

                        # Plot the second group of columns on the right Y-axis
                        ax2 = ax1.twinx()
                        for col in columns_to_plot[1:]:
                            interp = PchipInterpolator(df['Time'], df[col])
                            y_smooth = interp(xnew)
                            ax2.plot(xnew, y_smooth, label=col, color='g')
                        ax2.set_ylabel(', '.join(columns_to_plot[1:]), color='g')
                        ax2.tick_params(axis='y', labelcolor='g')
                        ax2.legend(loc='upper right')

                    else:
                        # Plot all columns on the same Y-axis using Pchip interpolation
                        for col in columns_to_plot:
                            interp = PchipInterpolator(df['Time'], df[col])
                            y_smooth = interp(xnew)
                            ax1.plot(xnew, y_smooth, label=col)
                        ax1.legend()

                plt.title(file[:-4])
                ax1.set_xlabel('Iterations')

                # Add a legend to the first subplot of the first figure
                if fig_index == 0 and subplot_index == 0 and not split_axes:
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=5)

        plt.tight_layout()

    plt.show()


def has_multiple_methods_or_classes(source_code):
    parsed_code = ast.parse(source_code)
    method_count = 0

    for node in parsed_code.body:
        if isinstance(node, ast.FunctionDef):
            # Count top-level functions
            method_count += 1
        elif isinstance(node, ast.ClassDef):
            # Count methods inside classes
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef):
                    method_count += 1

    return method_count > 1


def extract_functions_classes_and_intermediate_code(source_code):
    cst_tree = cst.parse_module(source_code)
    wrapper = MetadataWrapper(cst_tree)
    wrapper.resolve(PositionProvider)

    functions = []
    classes = []
    intermediate_code = []
    last_line = 0
    function_depth = 0  # Track the depth of nested functions
    current_indentation = 0  # Track the current indentation level

    class FunctionClassVisitor(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def visit_IndentedBlock(self, node: cst.IndentedBlock) -> bool:
            nonlocal current_indentation
            current_indentation += 1
            return True  # Continue visiting children

        def leave_IndentedBlock(self, original_node: cst.IndentedBlock) -> None:
            nonlocal current_indentation
            current_indentation -= 1

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            nonlocal last_line, function_depth
            function_depth += 1

            # Process only top-level functions (not inside classes)
            if function_depth == 1 and current_indentation == 0:
                start_line = self.get_metadata(PositionProvider, node).start.line
                end_line = self.get_metadata(PositionProvider, node).end.line

                # Adjust start_line to include decorators
                while start_line > 1 and source_code.splitlines()[start_line - 2].strip().startswith('@'):
                    start_line -= 1

                if start_line > last_line + 1:
                    intermediate_snippet = '\n'.join(source_code.splitlines()[last_line:start_line - 1]).strip()
                    if intermediate_snippet:
                        intermediate_code.append(intermediate_snippet)

                function_code = '\n'.join(source_code.splitlines()[start_line - 1:end_line]).strip()
                functions.append(function_code)
                last_line = end_line

            # Do not descend into nested functions
            return function_depth == 1

        def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
            nonlocal function_depth
            function_depth -= 1

        def visit_ClassDef(self, node: cst.ClassDef) -> None:
            nonlocal last_line
            start_line = self.get_metadata(PositionProvider, node).start.line
            end_line = self.get_metadata(PositionProvider, node).end.line

            # Adjust start_line to include decorators
            while start_line > 1 and source_code.splitlines()[start_line - 2].strip().startswith('@'):
                start_line -= 1

            if start_line > last_line + 1:
                intermediate_snippet = '\n'.join(source_code.splitlines()[last_line:start_line - 1]).strip()
                if intermediate_snippet:
                    intermediate_code.append(intermediate_snippet)

            class_code = '\n'.join(source_code.splitlines()[start_line - 1:end_line]).strip()
            classes.append(class_code)
            last_line = end_line

    visitor = FunctionClassVisitor()
    wrapper.visit(visitor)

    # Capture any remaining code after the last function/class
    lines = source_code.splitlines()
    if last_line < len(lines):
        remaining_code = '\n'.join(lines[last_line:]).strip()
        if remaining_code:
            intermediate_code.append(remaining_code)

    return functions, classes, intermediate_code


def remove_extracted_code(source_code, functions, classes):
    removed_sections = set()
    for func in functions:
        start_index = source_code.find(func)
        end_index = start_index + len(func)
        removed_sections.add((start_index, end_index))

    for cls in classes:
        start_index = source_code.find(cls)
        end_index = start_index + len(cls)
        removed_sections.add((start_index, end_index))

    removed_sections = sorted(list(removed_sections), reverse=True)
    for start, end in removed_sections:
        source_code = source_code[:start] + source_code[end:]

    return source_code


def create_regex_pattern_from_code(code_segment):
    """
    Create a regex pattern from a code segment allowing for variations in whitespace.
    """
    # Split the code segment into words and escape each word
    words = code_segment.split()
    escaped_words = [re.escape(word) for word in words]

    # Join the escaped words with a regex pattern that allows for any whitespace
    pattern = r'\s*'.join(escaped_words)

    return pattern


def replace_code_segment(original_code, old_segment, new_segment):
    # Generate a regex pattern from the old code segment
    pattern = create_regex_pattern_from_code(old_segment)

    # Replace the old segment with the new segment using regex
    replaced_code = re.sub(pattern, new_segment, original_code, flags=re.DOTALL)
    return replaced_code


def autoformat_py_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            file_path = os.path.join(directory, filename)
            print(f"Auto-formatting {file_path}...")
            subprocess.run(["autopep8", "--in-place", "--aggressive", "--aggressive", file_path])


def convert_ipynb_to_py(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".ipynb"):
            full_path = os.path.join(directory, filename)
            # Reading the notebook
            with open(full_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            py_filename = os.path.splitext(full_path)[0] + '.py'
            with open(py_filename, 'w', encoding='utf-8') as f:
                for cell in nb.cells:
                    if cell.cell_type == 'code':
                        # Writing the code cell's source into the Python file
                        f.write("#" + '-' * 20 + "Cell" + '-' * 20 + "\n")
                        f.write(cell.source + "\n\n")

            print(f"Converted {filename} to {py_filename}")


def print_colored(text, color=None):
    """
    Print text in the specified color, or in the default color if no color is provided.

    Args:
    text (str): The text to print.
    color (str, optional): The color to print the text in. Supported colors: 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'. Defaults to None.
    """
    colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m"
    }

    # If a color is provided, use it; otherwise, use the default terminal color.
    color_code = colors.get(color.lower(), "") if color else ""
    reset_code = "\033[0m" if color else ""

    print(f"{color_code}{text}{reset_code}")


def estimate_cost_per_iteration(model_name: str, unreadable_files_dict: dict, cost_per_1k_tokens: float):
    """
    Estimates the cost per iteration for tokenizing and processing the contents of Python files.

    Args:
    model_name (str): The name of the LLM model (e.g., 'gpt-3.5-turbo').
    unreadable_files_dict (dict): Dictionary with filenames as keys and their contents as values.
    cost_per_1k_tokens (float): Cost per 1000 tokens.

    Prints:
    The total number of tokens and the estimated cost for processing all files.
    """
    total_num_tokens = 0

    # Iterate over each file and count the tokens
    for filename, content in unreadable_files_dict.items():
        num_tokens = num_tokens_from_string(content, model_name)
        total_num_tokens += num_tokens

    # Calculate the total cost
    total_cost = (total_num_tokens / 1000) * cost_per_1k_tokens

    # Print the result
    print_colored(f'\nThe whole project contains {total_num_tokens} tokens and will cost around {total_cost:.2f} $ per Iteration.', "red")


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Counts the number of tokens in a string using the specified encoding model.

    Args:
    string (str): The string to tokenize.
    encoding_name (str): The name of the encoding model.

    Returns:
    int: The number of tokens in the string.
    """
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
