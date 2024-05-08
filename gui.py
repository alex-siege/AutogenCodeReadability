import sys, os, re
from PyQt5.QtWidgets import QMessageBox, QApplication, QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit, QVBoxLayout, QHBoxLayout, QWidget, QMainWindow, QFileDialog
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtCore import QUrl, QThread, pyqtSignal

import config_user
from config_user import *
import json


class MainLoopThread(QThread):
    finished = pyqtSignal()  # Signal to indicate the thread has finished

    def run(self):
        from main import main_loop  # Import the function here to avoid circular imports
        main_loop()
        self.finished.emit()  # Emit the finished signal when the loop is done

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialize lists to keep track of selected files
        self.selectedTestFiles = []
        self.selectedFilesToEdit = []
        # Adjust font size for the entire window
        self.setFont(QFont('Arial', 11))  # Increase font size here
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
        self.modelSelection.addItems(config_user.available_llm_models)
        # Set the current model based on the configuration, if it's in the list
        self.modelSelection.setCurrentText(config_user.available_llm_models[config_user.user_specified_llm])
        modelSelectionLayout.addWidget(self.modelSelection)
        mainLayout.addLayout(modelSelectionLayout)
        self.modelSelection.currentTextChanged.connect(self.updateModel)

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
        self.btnClearTestFiles = QPushButton('Clear Files', self)
        self.btnClearTestFiles.clicked.connect(self.clearTestFiles)
        testFilesLayout.addWidget(self.btnClearTestFiles)  # Add it to the testFilesLayout

        # Clear Files to Edit Button
        self.btnClearFilesToEdit = QPushButton('Clear Files', self)
        self.btnClearFilesToEdit.clicked.connect(self.clearFilesToEdit)
        filesToEditLayout.addWidget(self.btnClearFilesToEdit)  # Add it to the filesToEditLayout

        # Run Button
        self.runButton = QPushButton('Run')
        # self.runButton.clicked.connect(self.runMainLoop)
        self.runButton.clicked.connect(self.startMainLoopThread)
        mainLayout.addWidget(self.runButton)

        # Open Directory with Enhanced Files Button
        self.btnOpenEnhancedFilesDir = QPushButton('Open Directory with Enhanced Files', self)
        self.btnOpenEnhancedFilesDir.clicked.connect(self.openEnhancedFilesDir)
        # Add it to your layout where you want the button to appear
        mainLayout.addWidget(self.btnOpenEnhancedFilesDir)

    def updateProjectName(self):
        text = self.textProjectName.text()
        sanitized_text = re.sub(r'\W+', '', text.replace(' ', '_'))

        # Reflect changes in the QLineEdit
        self.textProjectName.setText(sanitized_text)
        config_user.project_name = sanitized_text
        print('Project Users Sanitized Text: ' + str(config_user.project_name))

    def openEnhancedFilesDir(self):
        # Get the directory path from config_user.py
        directory_path = config_user.edit_readable_project
        # Check if the directory exists
        if os.path.exists(directory_path):
            # Open the directory in the system's file explorer
            QDesktopServices.openUrl(QUrl.fromLocalFile(directory_path))
        else:
            # Show a message box if the directory does not exist
            QMessageBox.warning(self, 'Directory Not Found', f"The directory '{directory_path}' does not exist.")

    def clearTestFiles(self):
        self.selectedTestFiles.clear()
        self.testFilesPreview.clear()
        config_user.test_files = []
        print('File List to modify is cleared: ' + str(config_user.test_files))

    def clearFilesToEdit(self):
        self.selectedFilesToEdit.clear()
        self.editFilesPreview.clear()
        config_user.files_to_be_modified = []
        print('File List to edit is cleared: ' + str(config_user.files_to_be_modified))

    def updateApiKey(self):
        cleaned_key = self.textApiKey.text().strip("\"'")

        # Reflect changes in the QLineEdit with the cleaned key
        self.textApiKey.setText(cleaned_key)
        config_user.oai_api_key = cleaned_key
        print(config_user.oai_api_key)

    def updateModel(self, model):
        # Find the index of the current selection in the QComboBox
        currentIndex = self.modelSelection.currentIndex()

        # Update the config_user.user_specified_llm with the index
        config_user.user_specified_llm = currentIndex

        # For debugging or confirmation, print the new selection index
        print('Chosen LLM Model Index: ' + str(config_user.user_specified_llm))

    def selectTestFiles(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select test files", "", "Python Files (*.py)")
        if files:
            # Append new files to the existing list
            self.selectedTestFiles.extend(files)
            # Remove duplicates by converting to a set and back to a list
            self.selectedTestFiles = list(set(self.selectedTestFiles))
            # Update the preview text
            self.testFilesPreview.setText(', '.join(self.selectedTestFiles))
            # Update config_user with the current list of selected test files
            config_user.test_files = self.selectedTestFiles

            # print("directories_that_contain_test_files:\n" + str(self.selectedTestFiles))
            print("directories_that_contain_test_files:\n" + str(config_user.test_files))

    def selectFilesToEdit(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files to Edit", "", "Python Files (*.py)")
        if files:
            # Append new files to the existing list
            self.selectedFilesToEdit.extend(files)
            # Remove duplicates by converting to a set and back to a list
            self.selectedFilesToEdit = list(set(self.selectedFilesToEdit))
            # Update the preview text
            self.editFilesPreview.setText(', '.join(self.selectedFilesToEdit))
            # Update config_user with the current list of selected files to edit
            config_user.files_to_be_modified = self.selectedFilesToEdit

            # print("directories_that_contain_files_to_be_modified:\n" + str(self.selectedFilesToEdit))
            print("directories_that_contain_files_to_be_modified:\n" + str(config_user.files_to_be_modified))

    def startMainLoopThread(self):

        # SET VARIABLES START #############################################################################################
        # config_user.project_name = self.sanitized_text
        base_path = os.getcwd()  # Retrieve current Working Directory
        edit_project_directory = os.path.join(base_path, 'your_project_edit', config_user.project_name)
        config_user.edit_project_directory = edit_project_directory
        config_user.edit_unreadable_project = os.path.join(edit_project_directory, 'unreadable_code/')
        config_user.edit_readable_project = os.path.join(edit_project_directory, 'readable_code/')
        config_user.edit_original_project_bu = os.path.join(edit_project_directory, 'original_code_bu/')
        config_user.edit_readability_rating_project = os.path.join(edit_project_directory, 'readability_rating/')
        config_user.edit_replies_project = os.path.join(edit_project_directory, 'editor_replies/')

        # 2.2 Set-Up the running configuration for the LLM
        if config_user.oai_api_key is None or config_user.user_specified_llm == 0:
            # either retrieving from the 'OAI_CONFIG_LIST' in the 'llm_configs' folder ...
            config_user.config_list_all = config_user.config_list_from_json(env_or_file='llm_configs/OAI_CONFIG_LIST_ALL')
            config_user.config_list_llm_for_autogen = config_user.config_list_all[user_specified_llm]
        else:
            # ... or setting it up locally by using the specified parameters from 1.2
            config_user.llm_model = config_user.available_llm_models[user_specified_llm]
            config_user.config_list_llm_for_autogen = {
                "seed": 42,
                "cache_seed": 42,
                "config_list": [
                    {
                        "model": config_user.llm_model,
                        "api_key": config_user.oai_api_key
                    }
                ],
                "temperature": 0}

        # SET VARIABLES END #################################################################################################

        self.thread = MainLoopThread()  # Create a thread instance
        self.thread.finished.connect(self.onMainLoopFinished)  # Connect the finished signal to a method
        self.thread.start()  # Start the thread

    def onMainLoopFinished(self):
        # This method is called when the main_loop function finishes
        # You can update the GUI or notify the user that the process is complete
        QMessageBox.information(self, "Process Complete", "The main loop has finished executing.")


def launch_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
