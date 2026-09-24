
# Running the project
Highly suggested to create a python virtual environment
```
python -m venv myenv
```

Then activate it:
Windows powershell:
```
.\myenv\Scripts\Activate.ps1
```
Windows CMD:
```
myenv\Scripts\activate.bat
```
MacOS/Linux:
```
source myenv/bin/activate
```

After it's activated, it should says (myenv) at the start of your terminal.
All pip packages will be installed inside this virtual environment, and won't affect your global python installation.

### To install all requirements
```
pip install -r requirements.txt
```

- If you've added or removed packages in pip, save it to the requirements.txt via this command
```
pip freeze > requirements.txt
```