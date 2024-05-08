from helper_main import *
from gui import *

"""
With this function you can visualize the results which should be located inside the 'csv_results' folder. 
The plots here are the ones that have been used in the Master Thesis document.
"""

# Visualizing Results ====================================================================================================
plot_readability_metrics_all(
    directory='csv_results/',
    columns_to_plot=["Cyclomatic Complexity (CC)",
                     "Lines of Code (LOC)", "Logical Lines of Code (LLOC)", "Source Lines of Code (SLOC)",
                     "Comments", "Comment Blocks", "Blank Lines", "Single Comments",
                     "Distinct Operators (H1)", "Distinct Operands (H2)", "Total Number of Operators (N1)", "Total Number of Operands (N2)",
                     "Vocabulary", "Length", "Calculated Length", "Volume", "Difficulty", "Effort", "Time", "Bugs",
                     "Maintainability Index (MI)"],
    file_colors=['#00519e', '#11c2ff', '#ea6322', '#eebf69']) if not use_gui else None
