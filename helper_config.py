import re
import glob, os


def format_project_name(project_name):
    """
    Formats a project name to be safe for use in file paths.

    This function replaces spaces with underscores and removes characters that are not
    allowed in file names for most operating systems.

    Args:
        project_name (str): The original project name.

    Returns:
        str: The formatted project name.
    """
    # Replace spaces with underscores
    formatted_name = project_name.replace(" ", "_")

    # Remove characters that are not allowed in file names
    # This regex will keep letters, numbers, underscores, hyphens, and periods
    formatted_name = re.sub(r"[^a-zA-Z0-9_.-]", "", formatted_name)

    return formatted_name


def get_all_py_files_paths(dict_path):
    """
    Lists all Python (.py) files in a given directory with their full or relative paths.

    Args:
    dict_path (str): The directory path where the .py files are located.

    Returns:
    list: A list containing the full or relative paths of all .py files in the directory.
    """
    # List to store the paths of each file
    file_paths = []

    # Iterate over all .py files in the directory
    for filepath in glob.glob(os.path.join(dict_path, '*.py')):
        # Add the full (or relative) path to the list
        file_paths.append(filepath)

    # Return the list containing paths
    return file_paths
