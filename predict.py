"""loads a serialized model to make a prediction
"""
import typer
import pickle

from app.utils import load
app = typer.Typer()


@app.command()
def predict(input: str = 'data/test.csv',
            modelfile: str = './mlruns/0/8730c91516604d71a26fb1e2b351468e/artifacts/model/model.pkl'):

    data = load(input)

    # open a file, where you stored the pickled data
    file = open(modelfile, 'rb')

    # load the model
    model = pickle.load(file)

    # close the file
    file.close()

    # make the prediction
    y_pred = model.predict(data)
    data_pred = data.copy()
    data_pred['Survived'] = y_pred
    print(data_pred[['Age', 'Pclass', 'Sex', 'Fare', 'Survived']].head(20).T)

    total = data_pred.shape[0]
    survived = sum(data_pred['Survived'] == 1)
    percentage = round(100 * survived / total, 2)

    print(f"Survived: {survived}/{total} or {percentage}%")


if __name__ == "__main__":
    app()
