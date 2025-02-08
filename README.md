# Price Prediction App - README

## Setup and Installation

### 1. Create and Activate a Virtual Environment

Before installing dependencies, it is recommended to create a virtual environment to manage packages efficiently.

#### Windows:
```sh
python -m venv env
env\Scripts\activate
```

#### macOS/Linux:
```sh
python3 -m venv env
source env/bin/activate
```

### 2. Install Required Packages
Ensure you have an active internet connection, then install the required dependencies.
```sh
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the necessary libraries manually:
```sh
pip install streamlit pandas numpy scikit-learn
```

### 3. Running the Streamlit App
Once all dependencies are installed, you can start the application using the following command:
```sh
streamlit run app.py
```

## Additional Notes
- Ensure you are connected to the internet before installing dependencies.
- If any package installation fails, try running the installation command with `--upgrade`.
- If you face issues with Streamlit, check for updates using:
  ```sh
  pip install --upgrade streamlit
  ```

Now, you are ready to use the Price Prediction App!

