from pydantic import BaseModel, Field
from typing import Literal


class LoanRequest(BaseModel):

    person_age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Applicant age"
    )
    person_gender: Literal[
        "male",
        "female"
    ]
    person_income: float = Field(
        ...,
        gt=0,
        description="Annual income"
    )
    loan_amnt: float = Field(
        ...,
        gt=0
    )

    loan_int_rate: float = Field(
        ...,
        ge=0,
        le=100
    )

    loan_percent_income: float = Field(
        ...,
        ge=0,
        le=1
    )

    cb_person_cred_hist_length: float = Field(
        ...,
        ge=0
    )

    credit_score: float = Field(
        ...,
        ge=300,
        le=900
    )

    previous_loan_defaults_on_file: Literal[
        "Yes",
        "No"
    ]

    person_education: Literal[
        "High School",
        "Bachelor",
        "Master",
        "Associate",
        "Doctorate"
    ]

    person_home_ownership: Literal[
        "RENT",
        "OWN",
        "MORTGAGE",
        "OTHER"
    ]

    loan_intent: Literal[
        "EDUCATION",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ]

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 30,
                "person_gender": "male",
                "person_income": 60000,
                "loan_amnt": 10000,
                "loan_int_rate": 8.5,
                "loan_percent_income": 0.16,
                "cb_person_cred_hist_length": 8,
                "credit_score": 720,
                "previous_loan_defaults_on_file": "No",
                "person_education": "Bachelor",
                "person_home_ownership": "RENT",
                "loan_intent": "PERSONAL"
            }
        }
    }

class PredictionResponse(BaseModel):

    prediction: int
    probability: float
    result: str
    