from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status":"healthy"
    }

def test_predict_valid():
    payload = {
        "person_age": 33,
        "person_gender": "female",
        "person_income": 42920,
        "loan_amnt": 14125,
        "loan_int_rate": 10.37,
        "loan_percent_income": 0.33,
        "cb_person_cred_hist_length": 8,
        "credit_score": 675,
        "previous_loan_defaults_on_file": "No",
        "person_education": "Master",
        "person_home_ownership": "RENT",
        "loan_intent": "HOMEIMPROVEMENT"
        
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "probability" in body
    assert "result" in body

def test_predict_invalid():

    payload = {
        "person_age":15,
        "person_gender":"abc"
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422