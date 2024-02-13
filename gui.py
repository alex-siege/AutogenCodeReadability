import sys, os, re
from PyQt5.QtWidgets import QMessageBox, QApplication, QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit, QVBoxLayout, QHBoxLayout, QWidget, QMainWindow, QFileDialog
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl
import config

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialize lists to keep track of selected files
        self.selectedTestFiles = []
        self.selectedFilesToEdit = []
        self.setWindowTitle("OpenAI Configuration GUI")
        self.setGeometry(100, 100, 600, 400)  # Adjust size and position as needed

        # Create a central widget and set the main layout
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        mainLayout = QVBoxLayout(centralWidget)


        # API Key Input Layout
        apiKeyLayout = QHBoxLayout()
        self.labelApiKey = QLabel('Paste the OpenAI API Key here:')
        apiKeyLayout.addWidget(self.labelApiKey)
        self.textApiKey = QLineEdit()
        apiKeyLayout.addWidget(self.textApiKey)
        self.textApiKey.textChanged.connect(self.updateApiKey)
        mainLayout.addLayout(apiKeyLayout)

        # Project Name Input Layout
        projectNameLayout = QHBoxLayout()
        self.labelProjectName = QLabel('Define Project Name:')
        projectNameLayout.addWidget(self.labelProjectName)
        self.textProjectName = QLineEdit()
        projectNameLayout.addWidget(self.textProjectName)
        self.textProjectName.textChanged.connect(self.updateProjectName)
        mainLayout.addLayout(projectNameLayout)

        # Model Selection Dropdown Layout
        modelSelectionLayout = QHBoxLayout()
        self.modelSelection = QComboBox()
        self.modelSelection.addItems(['local-llm', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-preview'])
        self.modelSelection.setCurrentText('gpt-3.5-turbo-0125')  # Default value
        self.modelSelection.currentTextChanged.connect(self.updateModel)
        modelSelectionLayout.addWidget(self.modelSelection)
        mainLayout.addLayout(modelSelectionLayout)

        # Selecting Test Files Layout
        testFilesLayout = QHBoxLayout()
        self.btnSelectTestFiles = QPushButton('Select test files (optional)')
        self.btnSelectTestFiles.clicked.connect(self.selectTestFiles)
        testFilesLayout.addWidget(self.btnSelectTestFiles)
        self.testFilesPreview = QTextEdit()
        self.testFilesPreview.setReadOnly(True)
        self.testFilesPreview.setFixedHeight(40)
        testFilesLayout.addWidget(self.testFilesPreview)
        mainLayout.addLayout(testFilesLayout)

        # Selecting Files to Edit Layout
        filesToEditLayout = QHBoxLayout()
        self.btnSelectFilesToEdit = QPushButton('Select Files to edit')
        self.btnSelectFilesToEdit.clicked.connect(self.selectFilesToEdit)
        filesToEditLayout.addWidget(self.btnSelectFilesToEdit)
        self.editFilesPreview = QTextEdit()
        self.editFilesPreview.setReadOnly(True)
        self.editFilesPreview.setFixedHeight(40)
        filesToEditLayout.addWidget(self.editFilesPreview)
        mainLayout.addLayout(filesToEditLayout)

        # Clear Test Files Button
        self.btnClearTestFiles = QPushButton('Clear Test Files', self)
        self.btnClearTestFiles.clicked.connect(self.clearTestFiles)
        testFilesLayout.addWidget(self.btnClearTestFiles)  # Add it to the testFilesLayout

        # Clear Files to Edit Button
        self.btnClearFilesToEdit = QPushButton('Clear Files to Edit', self)
        self.btnClearFilesToEdit.clicked.connect(self.clearFilesToEdit)
        filesToEditLayout.addWidget(self.btnClearFilesToEdit)  # Add it to the filesToEditLayout

        # Run Button
        self.runButton = QPushButton('Run')
        self.runButton.clicked.connect(self.runMainLoop)
        mainLayout.addWidget(self.runButton)

        # Open Directory with Enhanced Files Button
        self.btnOpenEnhancedFilesDir = QPushButton('Open Directory with Enhanced Files', self)
        self.btnOpenEnhancedFilesDir.clicked.connect(self.openEnhancedFilesDir)
        # Add it to your layout where you want the button to appear
        mainLayout.addWidget(self.btnOpenEnhancedFilesDir)

    def updateProjectName(self):
        # Refactor the text input
        text = self.textProjectName.text()
        sanitized_text = re.sub(r'\W+', '', text.replace(' ', '_'))  # Replace spaces with underscores and remove special chars
        # global config.project_name
        config.project_name = sanitized_text
        # Reflect changes in the QLineEdit, if necessary
        self.textProjectName.setText(sanitized_text)
        print(config.project_name)
        # Consider writing the change back to config.py if needed

    def openEnhancedFilesDir(self):
        # Get the directory path from config.py
        directory_path = config.edit_readable_project
        # Check if the directory exists
        if os.path.exists(directory_path):
            # Open the directory in the system's file explorer
            QDesktopServices.openUrl(QUrl.fromLocalFile(directory_path))
        else:
            # Show a message box if the directory does not exist
            QMessageBox.warning(self, 'Directory Not Found', f"The directory '{directory_path}' does not exist.")

    def clearTestFiles(self):
        # Clear the list and the preview
        self.selectedTestFiles.clear()
        self.testFilesPreview.clear()
        global directories_that_contain_test_files
        directories_that_contain_test_files = self.selectedTestFiles
        # Consider writing the change back to config.py if needed

    def clearFilesToEdit(self):
        # Clear the list and the preview
        self.selectedFilesToEdit.clear()
        self.editFilesPreview.clear()
        global directories_that_contain_files_to_be_modified
        directories_that_contain_files_to_be_modified = self.selectedFilesToEdit
        # Consider writing the change back to config.py if needed


    def updateApiKey(self):
        # Strip potential quotes from the input
        cleaned_key = self.textApiKey.text().strip("\"'")
        # Wrap the cleaned key in single quotes
        config.oai_api_key = f"'{cleaned_key}'"
        print(config.oai_api_key)

    def updateModel(self, model):
        config.llm_model = model
        print('Chosen LLM Model: ' + str(config.llm_model))
        # Consider writing the change back to config.py if needed

    def selectTestFiles(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select test files", "", "Python Files (*.py)")
        if files:
            # Append new files to the existing list
            self.selectedTestFiles.extend(files)
            # Remove duplicates by converting to a set and back to a list
            self.selectedTestFiles = list(set(self.selectedTestFiles))
            # Update the preview text and the global variable
            self.testFilesPreview.setText(', '.join(self.selectedTestFiles))
            global directories_that_contain_test_files
            directories_that_contain_test_files = self.selectedTestFiles
            print("directories_that_contain_test_files:\n" + str(directories_that_contain_test_files))
            # Consider writing the change back to config.py if needed

    def selectFilesToEdit(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files to Edit", "", "Python Files (*.py)")
        if files:
            # Append new files to the existing list
            self.selectedFilesToEdit.extend(files)
            # Remove duplicates by converting to a set and back to a list
            self.selectedFilesToEdit = list(set(self.selectedFilesToEdit))
            # Update the preview text and the global variable
            self.editFilesPreview.setText(', '.join(self.selectedFilesToEdit))
            global directories_that_contain_files_to_be_modified
            directories_that_contain_files_to_be_modified = self.selectedFilesToEdit
            print("directories_that_contain_files_to_be_modified:\n" + str(directories_that_contain_files_to_be_modified))
            # Consider writing the change back to config.py if needed

    def runMainLoop(self):
        from main import main_loop  # Import the function here to avoid circular imports
        main_loop()


def launch_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
