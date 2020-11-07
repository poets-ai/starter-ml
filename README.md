# ML Starter Repo

## 0. INTRO
We will train a model (1. TRAINING) and later served it (2. ML API Service). Start by cloning the repo:

 ``` 
    git clone https://github.com/poets-ai/starter-ml.git
```


## 1.  TRAINING

### Motivation

Machine Learning has matured in the industry and like other areas it greatly benefits from having standardized procedures and best practices whenever possible. On the other hand, most of the available educational material on ML is about the theory but rarely are these best practices taught.


### Goal

We will focus on one such best practice: building a suitable production-ready project structure for Machine Learning applications. 


### Values

*   **Reproducibility**: you should be able to reproduce any past experiment.
*   **Production Ready**: the models trained by your code should be able to easily be put into a production environment.
*   **Visibility**: you should be able to easily inspect the results, metrics, and parameters used for each experiment.
*   **Generality**: the code base should serve as a template for future machine learning projects
*   **Speed**: on your next project the template should drastically cut your time to production.


### Objectives

1. Train a model for the [Titanic Dataset](https://www.kaggle.com/c/titanic/data). The model will not be your focus but rather an excuse to create the project structure.
2. Your training code should be able to take command line arguments so it's easily usable from bash. Important parameters are:
    1. data_path: Input data should not be a constant since the repo should be general.
    2. debug: (optional) whether you are in debug mode.
    3. model_type: (optional) you can support changing the model used for training.
3. On each run / experiment your code should do the following tasks:
    4. Serialize/store the exact input parameters used for the experiment
    5. Serialize/store the resulting _metrics_ from experiment.
    6. Serialize/store the trained model plus the exact preprocessing procedure such that inference can be made **without** the original codebase. [pickle, model.save]
        1. Train -> save model to storage -> load model in sever
    7. Your code should serialize/store the exact code used in the experiment.
4. At the end of the project create a separate repo with the same code and remove any project specific parts, add comments of where the next user should probably insert important code. Create a README telling users how to easily use the template.


### Bonus
*   Try to have a way to visualize the parameters and metrics given by various experiments so you can compare them.
*   Try to separate code that sets up the experiment (which should be generic) from the code that does the preprocessing procedure and model definition (which is project specific).
*   Add nice features to the template such as data splitting, automatic exploration of the data, hyper parameter tuning, etc. 


### Tips

*   Check out tools / services like [ML Flow](https://mlflow.org/) and [Weights and Biases](https://www.wandb.com/).
*   [Scikit Learn’s custom transformers](https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156) are a great way to perform complex preprocessing.
*   Structure idea

    /src # your actual code

       ….
    /results  #&lt;- gitignore
       experiment1/ # serialized stuff goes here
           …
       experiment2/


### Requirements
* Install MLflow and scikit-learn. There are two options for installing these dependencies:

   ** Install MLflow with extra dependencies, including scikit-learn (via pip install mlflow[extras])

   ** Install MLflow (via pip install mlflow) and install scikit-learn separately (via pip install scikit-learn)

* Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
  


### Training the Model
Use values for path and parameters by passing them as arguments to train.py:

    python app/train.py data/train.csv --params-file params.yml


### Comparing the Models
On the terminal

    mlflow ui

and view it at [http://localhost:5000]([http://localhost:5000])

### Run using conda environment
To run this project, invoke 

    mlflow run . -P mode='standard' path='./'

After running this command, MLflow runs the training code in a new Conda environment with the 
dependencies specified in conda.yaml.

### Local prediction with test data 
Modify model path to your best performing model
```
    python predict.py
```

### Serving the Model with mlflow

To deploy the server, run (replace the path with your model’s actual path):
    
    mlflow models serve -m ./mlruns/0/548e6aca1fe341e4a83d151a479e4066/artifacts/model


Once you have deployed the server, you can pass it some sample data and see the predictions. 
The following example uses curl to send a JSON-serialized pandas DataFrame with the split orientation to the model server.

    curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"],"data":[[893,3,"Wilkes, Mrs. James (Ellen Needs)","female",47.0,1,0,"363272",7.0,null,"S"]]}' http://127.0.0.1:5000/invocations

the server should respond with output similar to:
```
    [[0]]
```





## 2. ML API Service

### Motivation
Machine Learning is software, as such its main purpose is to be used in a production environment which could be a server, mobile phone, web browser, IoT device, etc. Its very important for ML practitioners to have knowledge on how to deploy their own models.

### Goal
We will be deploying a trained model on a web server and exposing it via a REST API. 

### Values
* **Scalability**: you should construct the architecture such that it can be scaled in the future.
* **Portability**: if possible your model should not be tied to a specific setup.

### Objectives
1. Expose a REST endpoint for your model. You should accept a JSON request and return a JSON with the prediction.
2. You should Dockerize your application to make it portable.
3. It should be easy to train a new model and deploy it, if possible it should be an automatic process.
4. Modify your previous project such that your training code makes deployment easier.
5. Your endpoint should be secured by an Authorization token.
6. (Bonus) Deploy it to a real production environment on a cloud service.
Create a new project that works on the MNIST dataset. This should help you make your template more project independent.

### Recommendations
Use FastAPI if possible, Flask or Django are also good.
Use docker-compose to test locally.

`


### Requirements
Install Docker and Docker-compose


### API start
#### With docker compose
    sudo docker-compose up -d
    
#### With docker
    sudo docker build -t {name}api .
    sudo docker run -i --name {name}apicontainer -p 8000:8000 {name}api
    
    replace {name}
    
#### With Uvicorn
    python ./app/main.py


### Api testing

View it at [http://localhost:5000]([http://localhost:8000])

the server should respond with:

    {
    "message": "API live!"
    }

at the endpoint [http://0.0.0.0:8000/predict/](http://0.0.0.0:8000/predict/)
passing:
 
    {
    "Pclass": 3,  
    "Sex": "female",
    "Age": 25,
    "SibSp": 0,
    "Parch": 1,
    "Fare": 12,
    "Embarked": "S"
    }


the server should respond with output similar to:
```
    {
    "PassengerId": null,
    "Pclass": "1",
    "Name": "Paulo",
    "Sex": "male",
    "Age": 28.0,
    "SibSp": 0.0,
    "Parch": 0.0,
    "Ticket": 124124,
    "Fare": 80.0,
    "Cabin": null,
    "Embarked": "S",
    "Survived": 0
}
```

