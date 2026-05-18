import gradio as gr
import pandas as pd
from src.pipelines.inference_pipeline import inference_pipeline
cols_path = 'src/utils/columns.csv'
columns = pd.read_csv(cols_path, header=None)[0].tolist()

def predict(*values):
    data = dict(zip(columns, values))
    df = pd.DataFrame([data])
    return inference_pipeline(df)


demo = gr.Interface(
    fn=predict,

    inputs=[

        # ===== IDENTIFIER =====
        gr.Number(label="Id"),

        # ===== CATEGORICAL (MS SUBCLASS) =====
        gr.Dropdown([20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190], label="MSSubClass"),

        gr.Dropdown(['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'], label="MSZoning"),

        # ===== NUMERIC =====
        gr.Number(label="LotFrontage"),
        gr.Number(label="LotArea"),

        # ===== CATEGORICAL =====
        gr.Dropdown(['Grvl', 'Pave', 'NA'], label="Street"),
        gr.Dropdown(['Grvl', 'Pave','NA'], label="Alley"),

        gr.Dropdown(['Reg', 'IR1', 'IR2', 'IR3'], label="LotShape"),
        gr.Dropdown(['Lv1', 'Bnk', 'HLs', 'Low'], label="LandContour"),
        gr.Dropdown(['AllPub', 'NoSewr', 'NoSeWa', 'ELO'], label="Utilities"),

        gr.Dropdown(['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], label="LotConfig"),
        gr.Dropdown(['Gtl', 'Mod', 'Sev'], label="LandSlope"),

        gr.Dropdown([
            'Blmngtn','Blueste','BrDale','BrkSide','ClearCr','CollgCr',
            'Crawfor','Edwards','Gilbert','IDOTRR','MeadowV','Mitchel',
            'Names','NoRidge','NPkVill','NridgHt','NWAmes','OldTown',
            'SWISU','Sawyer','SawyerW','Somerst','StoneBr','Timber','Veenker'
        ], label="Neighborhood"),

        gr.Dropdown(['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRNe','RRAe'], label="Condition1"),
        gr.Dropdown(['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRNe','RRAe'], label="Condition2"),

        gr.Dropdown(['1Fam','2FmCon','Duplx','TwnhsE','TwnhsI'], label="BldgType"),
        gr.Dropdown(['1Story','1.5Fin','1.5Unf','2Story','2.5Fin','2.5Unf','SFoyer','SLvl'], label="HouseStyle"),

        gr.Dropdown(list(range(1,11)), label="OverallQual"),
        gr.Dropdown(list(range(1,11)), label="OverallCond"),

        gr.Number(label="YearBuilt"),
        gr.Number(label="YearRemodAdd"),

        gr.Dropdown(['Flat','Gable','Gambrel','Hip','Mansard','Shed'], label="RoofStyle"),
        gr.Dropdown(['ClyTile','CompShg','Membran','Metal','Roll','Tar&Grv','WdShake','WdShngl'], label="RoofMatl"),

        gr.Dropdown([
            'AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd',
            'HdBoard','ImStucc','MetalSd','Other','Plywood','PreCast',
            'Stone','Stucco','VinylSd','Wd Sdng','WdShing'
        ], label="Exterior1st"),

        gr.Dropdown([
            'AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd',
            'HdBoard','ImStucc','MetalSd','Other','Plywood','PreCast',
            'Stone','Stucco','VinylSd','Wd Sdng','WdShing'
        ], label="Exterior2nd"),

        gr.Dropdown(['BrkCmn','BrkFace','CBlock','None','Stone'], label="MasVnrType"),
        gr.Number(label="MasVnrArea"),

        gr.Dropdown(['Ex','Gd','TA','Fa','Po'], label="ExterQual"),
        gr.Dropdown(['Ex','Gd','TA','Fa','Po'], label="ExterCond"),

        gr.Dropdown(['BrkTil','CBlock','PConc','Slab','Stone','Wood'], label="Foundation"),

        gr.Dropdown(['Ex','Gd','TA','Fa','Po','NA'], label="BsmtQual"),
        gr.Dropdown(['Ex','Gd','TA','Fa','Po','NA'], label="BsmtCond"),
        gr.Dropdown(['Gd','Av','Mn','No','NA'], label="BsmtExposure"),

        gr.Dropdown(['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA'], label="BsmtFinType1"),
        gr.Number(label="BsmtFinSF1"),

        gr.Dropdown(['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA'], label="BsmtFinType2"),
        gr.Number(label="BsmtFinSF2"),
        gr.Number(label="BsmtUnfSF"),
        gr.Number(label="TotalBsmtSF"),

        gr.Dropdown(['Floor','GasA','GasW','Grav','OthW','Wall'], label="Heating"),
        gr.Dropdown(['Ex','Gd','TA','Fa','Po'], label="HeatingQC"),
        gr.Dropdown(['N','Y'], label="CentralAir"),
        gr.Dropdown(['SBrkr','FuseA','FuseF','FuseP','Mix'], label="Electrical"),

        gr.Number(label="1stFlrSF"),
        gr.Number(label="2ndFlrSF"),
        gr.Number(label="LowQualFinSF"),
        gr.Number(label="GrLivArea"),

        gr.Number(label="BsmtFullBath"),
        gr.Number(label="BsmtHalfBath"),
        gr.Number(label="FullBath"),
        gr.Number(label="HalfBath"),
        gr.Number(label="BedroomAbvGr"),
        gr.Number(label="KitchenAbvGr"),

        gr.Dropdown(['Ex','Gd','TA','Fa','Po'], label="KitchenQual"),
        gr.Number(label="TotRmsAbvGrd"),

        gr.Dropdown(['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev','Sal'], label="Functional"),

        gr.Number(label="Fireplaces"),
        gr.Dropdown(['Ex','Gd','TA','Fa','Po','NA'], label="FireplaceQu"),

        gr.Dropdown(['2Types','Attchd','Basment','BuiltIn','CarPort','Detchd','NA'], label="GarageType"),
        gr.Number(label="GarageYrBlt"),
        gr.Dropdown(['Fin','RFn','Unf','NA'], label="GarageFinish"),
        gr.Number(label="GarageCars"),
        gr.Number(label="GarageArea"),

        gr.Dropdown(['Ex','Gd','TA','Fa','Po','NA'], label="GarageQual"),
        gr.Dropdown(['Ex','Gd','TA','Fa','Po','NA'], label="GarageCond"),

        gr.Dropdown(['Y','P','N'], label="PavedDrive"),

        gr.Number(label="WoodDeckSF"),
        gr.Number(label="OpenPorchSF"),
        gr.Number(label="EnclosedPorch"),
        gr.Number(label="3SsnPorch"),
        gr.Number(label="ScreenPorch"),

        gr.Number(label="PoolArea"),
        gr.Dropdown(['Ex','Gd','TA','Fa','NA'], label="PoolQC"),

        gr.Dropdown(['GdPrv','MnPrv','GdWo','MnWw','NA'], label="Fence"),
        gr.Dropdown(['Elev','Gar2','Othr','Shed','TenC','NA'], label="MiscFeature"),
        gr.Number(label="MiscVal"),

        gr.Number(label="MoSold"),
        gr.Number(label="YrSold"),

        gr.Dropdown(['WD','CWD','VWD','New','COD','Con','ConLw','ConLI','ConLD','Oth'], label="SaleType"),
        gr.Dropdown(['Normal','Abnorml','AdjLand','Alloca','Family','Partial'], label="SaleCondition"),
    ],

    outputs="text",

    title="House Price Prediction"
)

demo.launch(server_name="0.0.0.0", server_port=7860)