import re


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