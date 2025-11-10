##  HQNN Environment Setup Guide

This guide provides detailed instructions for setting up the Hybrid Quantum Neural Network (HQNN) environment for the Catalyst Performance Prediction under Data Consistency Variation project.

# 1. Python Version Requirement
- Recommended Python Version:
- Python 3.11.x (3.10 also works — avoid 3.13 for now becuase it may not work with the package)
 - Python 3.13 was released in 2024, and some packages such as TensorFlow and MLflow are not yet officially aded to the version

- How to Check if Python Is Installed
 - Open Command Prompt or PowerShell, and run:
 - python --version
- If you see: Python 3.11.x
 - Great — you’re ready.
- If not, download and install it from:
- https://www.python.org/downloads/

- During installation:
- Check “Add Python to PATH”
- Click Customize Installation → Next → Install
- To verify after installation:
 - python --version

# 2. Create and Activate a Virtual Environment
- Step 1: Navigate to the Project Folder
 for example: cd "E:\Computational Engineering\HQNN-for-Catalyst-Prediction"

Step 2: Create the Environment
<python -m venv hqnn_env>
 the above will create a virtual environment called hqnn_env

Step 3: Activate the Environment
<.\hqnn_env\Scripts\Activate.ps1>

Step3: install kernel in the path you are running the code from to be able to use ipynb
<Installed kernelspec hqnn_env in C:\Users\sanya\AppData\Roaming\jupyter\kernels\hqnn_env>


Once active, your terminal should display:

(hqnn_env) PS E:\Computational Engineering\HQNN-for-Catalyst-Prediction>

3. Install Project Dependencies
Option A – Using the requirements file

<pip install -r requirements.txt>



4. Set-up fairchem-core see link for more information: https://fair-chem.github.io/core/install.html
from fairchem.data import PyChemDataset

# This will download or load from cache
dataset = PyChemDataset(root="data/pychem")

print("Number of catalyst samples:", len(dataset))
print("Sample data structure:", dataset[0])




🔍 4. Verify Your Setup
Check Key Package Imports

Run this in Python:

import pennylane, qiskit, torch, sklearn, mlflow
print("✅ Environment setup successful!")

Test PennyLane Quantum Circuit
import pennylane as qml
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliZ(1))

print(circuit([0.3, 0.7]))


If it prints a number (e.g., 0.88) ✅ everything works!

📊 5. Launch MLflow for Experiment Tracking

Run:

mlflow ui


Then open your browser and go to:
👉 http://127.0.0.1:5000

You should see the MLflow dashboard (this will later display your model runs, metrics, and dataset versions).

🧠 6. Running the Project
Start JupyterLab
jupyter lab


Then open:

notebooks/Week1_Environment_Setup_and_Verification.ipynb

Running Python Scripts

From the terminal:

python scripts/train_hqnn.py

🔄 7. Save Your Environment (Optional)

After successful setup, export your dependency list for sharing or reproducing:

pip freeze > requirements.txt

🧹 8. Deactivate the Environment

When done working:

deactivate

✅ 9. Troubleshooting
Problem	Solution
- Command not found: python	Reinstall Python and check “Add to PATH” during setup
- MLflow UI not launching	Ensure you are in the activated environment before running mlflow ui
- PowerShell blocked script execution	Run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
- Import errors for PennyLane or Qiskit	Run pip install --upgrade pip then reinstall the packages