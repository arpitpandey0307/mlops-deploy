#Import Testclient so we can send simulated HTTP requests directly to the FASTAPI
from fastapi.testclient import TestClient
import pytest
#Import the app object from the main.py file
from main import create_app

#Create a reusable test client that will be used to send requests to the API
#client = TestClient(app)

#The test will no longer use the actual model, but instead will use a fake model that always returns a fixed prediction. This allows us to test the API without relying on the actual model being available or trained.
class FakeIrisModel:
    def predict(self, features):
        #Return a fixed prediction for testing purposes
        return [0]  # Always predict class 0 (setosa)

def fake_model_loader():
    #Return an instance of the FakeIrisModel for testing purposes
    return FakeIrisModel()

def fake_version_loader():
    #Return a fixed model version for testing purposes
    return "test-version"

@pytest.fixture
def client():
    app= create_app(model_loader=fake_model_loader, version_loader=fake_version_loader)
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    #Send a GET request to the /health endpoint
    #Verify that the application is running and responding correctly by testing the /health endpoint
    response = client.get("/health")
    #Assert that the response status code is 200 (OK)
    #A successful Http request should return a status code of 200, indicating that the request was successful and the server responded as expected.
    assert response.status_code == 200
    #Assert that the response JSON matches the expected output
    assert response.json() == {"status": "healthy"}

def test_valid_iris_prediction(client):
    #verify that valid flower measurement produces the predict
    #Send a POST request to the /predict endpoint with valid input data
    #Test the /predict endpoint with valid input data to ensure that the model is making predictions correctly.
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    #Assert that the response status code is 200 (OK)
    assert response.status_code == 200
    #Convert the json response to a Python dictionary and assert that the prediction matches the expected output
    result = response.json()

    #check the response contract: both fields must exist
    assert "prediction" in result
    assert "class_name" in result

    #the predicted class must be one of the three possible classes in the iris dataset
    assert result["class_name"] in ["setosa", "versicolor", "virginica"]
    #the predicted integrer must be between 0 and 2, corresponding to the three classes in the iris dataset
    assert result["prediction"] in [0, 1, 2]

    def test_invalid_iris_input():
        #verify that invalid flower measurement produces an error
        #Send a POST request to the /predict endpoint with invalid input data
        #Test the /predict endpoint with invalid input data to ensure that the model is handling errors correctly.
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": -0.2  # Invalid value (negative petal width)
        }
        response = client.post("/predict", json=payload)
        #Assert that the response status code is 422 (Unprocessable Entity)
        assert response.status_code == 422