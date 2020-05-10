# papermill-mlflow
Simple data science experimentation with `jupyter`, `papermill`, and `mlflow`

#### Associated blog post: [A simpler experimentation workflow with Jupyter, Papermill, and MLflow](https://eugeneyan.com/2020/03/15/experimentation-workflow-with-jupyter-papermill-mlflow)
---

# Quick-start

- Clone this repo

```
git clone git@github.com:eugeneyan/papermill-mlflow.git
```
- Set up virtualenv

```
cd papermill-mlflow

# Create virtualenv based on requirements.txt
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install kernelspec for Jupyter notebooks (the name argument must be identical)
python -m ipykernel install --user --name=papermill-mlflow
```

- Start Jupyter notebook

```
cd notebooks
jupyter notebook
```

- Run the cells in `runner.ipynb`

![image of runner notebook](https://raw.githubusercontent.com/eugeneyan/papermill-mlflow/master/assets/runner.png)

- Start MLflow (in another terminal)

```
# Open another terminal

# Activate the virtualenv
cd papermill-mlflow
source venv/bin/activate

# Start the mlflow server
cd notebooks
mlflow server
```

- Access the MLflow UI opening this in a browser: [http://127.0.0.1:5000](http://127.0.0.1:5000/#/experiments/1)
	- Navigate to "indices" in the experiment tab if necessary

![image of mlflow](https://raw.githubusercontent.com/eugeneyan/papermill-mlflow/master/assets/mlflow.png)
