
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.pipelines.inference_pipeline import inference_pipeline


app = FastAPI()


class HouseFeatures(BaseModel):
    Id: int | None = None

    MSSubClass: int | None = None
    MSZoning: str | None = None

    LotFrontage: float | None = None
    LotArea: float | None = None

    Street: str | None = None
    Alley: str | None = None
    LotShape: str | None = None
    LandContour: str | None = None
    Utilities: str | None = None
    LotConfig: str | None = None
    LandSlope: str | None = None

    Neighborhood: str | None = None
    Condition1: str | None = None
    Condition2: str | None = None

    BldgType: str | None = None
    HouseStyle: str | None = None

    OverallQual: int | None = None
    OverallCond: int | None = None

    YearBuilt: int | None = None
    YearRemodAdd: int | None = None

    RoofStyle: str | None = None
    RoofMatl: str | None = None

    Exterior1st: str | None = None
    Exterior2nd: str | None = None

    MasVnrType: str | None = None
    MasVnrArea: float | None = None

    ExterQual: str | None = None
    ExterCond: str | None = None

    Foundation: str | None = None

    BsmtQual: str | None = None
    BsmtCond: str | None = None
    BsmtExposure: str | None = None
    BsmtFinType1: str | None = None
    BsmtFinType2: str | None = None

    BsmtFinSF1: float | None = None
    BsmtFinSF2: float | None = None
    BsmtUnfSF: float | None = None
    TotalBsmtSF: float | None = None

    Heating: str | None = None
    HeatingQC: str | None = None

    CentralAir: str | None = None
    Electrical: str | None = None

    FirstFlrSF: float = Field(alias="1stFlrSF")
    SecondFlrSF: float = Field(alias="2ndFlrSF")
    LowQualFinSF: float  | None = None
    GrLivArea: float  | None = None

    BsmtFullBath: float  | None = None
    BsmtHalfBath: float | None = None
    FullBath: int | None = None
    HalfBath: int | None = None

    BedroomAbvGr: int | None = None
    KitchenAbvGr: int | None = None
    KitchenQual: str | None = None

    TotRmsAbvGrd: int | None = None
    Functional: str | None = None

    Fireplaces: int | None = None
    FireplaceQu: str | None = None

    GarageType: str | None = None
    GarageYrBlt: float | None = None
    GarageFinish: str | None = None
    GarageCars: float | None = None
    GarageArea: float | None = None
    GarageQual: str | None = None
    GarageCond: str | None = None

    PavedDrive: str | None = None

    WoodDeckSF: float | None = None
    OpenPorchSF: float | None = None
    EnclosedPorch: float | None = None
    ThreeSsnPorch: float = Field(alias="3SsnPorch")
    ScreenPorch: float | None = None

    PoolArea: float | None = None
    PoolQC: str | None = None

    Fence: str | None = None
    MiscFeature: str | None = None
    MiscVal: float | None = None

    MoSold: int | None = None
    YrSold: int | None = None

    SaleType: str | None = None
    SaleCondition: str | None = None

@app.get("/")
def read_root():
    return {"message":"API working"}


@app.post("/predict")
def predict(data: HouseFeatures):

    input_dict = data.model_dump(by_alias=True)

    print(input_dict.keys())

    df = pd.DataFrame([input_dict])

    print(df.columns.tolist())
    print(df.shape)
    print(df.head())

    prediction = inference_pipeline(df)

    return {
        "prediction": prediction
    }


@app.post("/multipredict")
def predict(data: list[HouseFeatures]):

    df = pd.DataFrame([item.model_dump(by_alias=True) for item in data])

    prediction = inference_pipeline(df)

    return {
        "prediction": prediction
    }