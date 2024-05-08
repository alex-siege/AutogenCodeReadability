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
import config_user
# from config_system import *
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
import os
import matplotlib
from matplotlib.ticker import FuncFormatter
from itertools import cycle  # For cycling through colors if there are more files than colors


# Set global font properties to match TeX Gyre Heros/Helvetica/Arial
matplotlib.rcParams['font.sans-serif'] = "Arial" # Fallback to Arial if Helvetica is not available
matplotlib.rcParams['font.family'] = "sans-serif"


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
    print(result.stdout) if not config_user.suppress_miscellaneous_info else None
    print(result.stderr) if not config_user.suppress_miscellaneous_info else None
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
        headers = ["Iter", "Cyclomatic Complexity (CC)", "Lines of Code (LOC)", "Logical Lines of Code (LLOC)", "Source Lines of Code (SLOC)", "Comments", "Comment Blocks", "Blank Lines",
                   "Single Comments", "Distinct Operators (H1)", "Distinct Operands (H2)", "Total Number of Operators (N1)", "Total Number of Operands (N2)", "Vocabulary", "Length",
                   "Calculated Length", "Volume", "Difficulty", "Effort", "Time", "Bugs", "Maintainability Index (MI)"]

        # Create DataFrame for this file's metrics
        df = pd.DataFrame([metrics], columns=headers)

        # Check if the file exists to determine if headers should be written
        file_exists = os.path.isfile(full_path)

        # Save DataFrame to CSV (append if file exists, write headers only if creating new file)
        df.to_csv(full_path, mode='a', index=False, header=not file_exists)
        print(f"Metrics {'appended to' if file_exists else 'saved to'} {full_path}")


def document_readability(path_for_check, path_for_save, time_point, file_name):
    if path_for_check:
        file_path = path_for_check + file_name
    else:
        file_path = '/your_project/' + file_name

    print("Checking readability...\n")
    print_readability_of_given_folder_single(file_path, path_for_save, time_point)


def read_txt(directory):
    with open(directory, 'r') as file:
        file_read = file.read()
    return file_read


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


def refactor_method_name(old_method_name, new_method_name):
    project = None
    try:
        directory = os.path.normpath(config_user.edit_readable_project)
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


def check_readability_index(directory_for_retrieval, file_name, selected_indicator):
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


def has_multiple_methods(source_code):
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


def split_in_segments(source_code):
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


def replace_code_segment(original_code, old_segment, new_segment):
    # Escape special characters in the old code segment to create a safe regex pattern
    pattern = re.escape(old_segment)

    # Prepare the new segment for replacement, escaping backslashes and other potential escape sequences
    # This step might need adjustment based on how new_segment is provided
    safe_new_segment = new_segment.replace('\\', '\\\\')

    # Replace the old segment with the new, safe segment using regex
    replaced_code = re.sub(pattern, safe_new_segment, original_code, flags=re.DOTALL)
    return replaced_code


def autoformat_py_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            file_path = os.path.join(directory, filename)
            print(f"Auto-formatting {file_path}...")
            subprocess.run(["autopep8", "--in-place", "--aggressive", "--aggressive", file_path])


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


def plot_readability_metrics_all(directory, columns_to_plot=None, file_colors=None,
                                        arrow_height=0.05, arrow_tip_height=0.2, group_spacing=0.1,
                                        font_size=25, font_size_legend=22, dpi=100, figure_width_px=1400, show_legend=True):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not files:
        print("No CSV files found in the directory.")
        return

    if not file_colors:
        file_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    color_cycle = cycle(file_colors)

    figure_height_in = int(0.98 * figure_width_px / dpi)
    figure_width_in = figure_width_px / dpi

    plt.figure(figsize=(figure_width_in, figure_height_in), dpi=dpi)
    ax = plt.gca()

    min_percent_change, max_percent_change = float('inf'), float('-inf')

    for file in files:
        df = pd.read_csv(os.path.join(directory, file))
        if columns_to_plot is None:
            columns_to_plot = [col for col in df.columns if col != 'Time']

        for col in columns_to_plot:
            if col not in df.columns:
                continue
            first_value = df[col].iloc[0]
            last_value = df[col].iloc[-1]
            percent_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
            min_percent_change = min(min_percent_change, percent_change)
            max_percent_change = max(max_percent_change, percent_change)

    ax.set_xlim([min_percent_change - 10, max_percent_change + 10])
    ax.set_xscale('symlog', linthresh=1, linscale=1)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}%'.format(x)))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='#d3d3d3', zorder=0)

    ax.spines['left'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0, labelleft=False)
    ax.tick_params(axis='x', labelsize=font_size)

    legend_handles = []

    for file_index, file in enumerate(files):
        df = pd.read_csv(os.path.join(directory, file))
        current_color = next(color_cycle)

        for col_index, col in enumerate(columns_to_plot):
            if col not in df.columns:
                continue
            first_value = df[col].iloc[0]
            last_value = df[col].iloc[-1]
            percent_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0

            # Adjust position calculation to align with labels ordered from top to bottom
            position = (len(columns_to_plot) - col_index - 1) * (1 + group_spacing) + (len(files) - file_index - 1) * 0.2

            if percent_change > 0.1 or percent_change < -0.1:
                arrow_head_length = abs(percent_change) / 10
                plt.arrow(0, position, percent_change, 0, width=arrow_height,
                          head_width=arrow_tip_height, head_length=arrow_head_length,
                          length_includes_head=True, fc=current_color, ec=current_color, zorder=3)
            else:
                plt.plot(0, position, 'o', color=current_color, zorder=3, markersize=4)

        legend_handles.append(plt.Line2D([0], [0], color=current_color, lw=4, label=file.replace('.csv', '')))

    if show_legend:
        plt.legend(handles=list(reversed(legend_handles)), loc='upper center', bbox_to_anchor=(0.2, 1.07),
                   frameon=False, fontsize=font_size_legend, ncol=4)

    for col_index, col in enumerate(columns_to_plot):
        # Calculate position from the top instead of the bottom by reversing the order
        position = (len(columns_to_plot) - col_index - 1) * (1 + group_spacing) + 0.3
        plt.text(min_percent_change - 15, position, col, ha='right', va='center', fontsize=font_size)

    y_offset_factor = 1  # Adjust this factor as needed to move the label further down.

    y_pos_for_xlabel = ax.get_ylim()[0] - (y_offset_factor * abs(ax.get_ylim()[0]))
    ax.text(0, y_pos_for_xlabel, 'Relative Change (%)', ha='center', va='top', fontsize=font_size_legend)
    # Adjust ylim and subplots_adjust if necessary to accommodate the new layout
    plt.ylim(-1, len(columns_to_plot) * (1 + group_spacing))
    plt.subplots_adjust(left=0.15, right=0.85, top=0.75)
    plt.tight_layout()

    # Adjust the margins if necessary
    plt.subplots_adjust(bottom=0.1)  # Increase the bottom margin

    plt.show()
