
import great_expectations as ge


def validate_housing_data(df, columns):
    print('Starting data validation with Great Expectations')
    print(ge.__version__)
    context = ge.get_context()

    data_source_name = 'my_data_source'
    data_source = context.data_sources.add_pandas(name=data_source_name)
    data_asset_name = 'my_data_asset_name'
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)
    batch_definition_name = 'my_batch_definition'
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name=batch_definition_name)
    batch_parameters = {'dataframe': df}

    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    suite = context.suites.add(ge.ExpectationSuite(name='housing_expectation_suite'))
    #schema validation - essential columns
    for col in columns:
        exceptation1 = ge.expectations.ExpectColumnToExist(column=col)
        suite.add_expectation(exceptation1)
    #bussines logic validation
    my_expectation_list = [

    ge.expectations.ExpectColumnValuesToBeInSet(column='MSSubClass', value_set=[20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190]),
        ge.expectations.ExpectColumnValuesToBeInSet(
            column='MSZoning',
            value_set=['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Street',
            value_set=['Grvl', 'Paved', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Alley',
            value_set=['Grvl', 'Paved', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='LotShape',
            value_set=['Reg', 'IR1', 'IR2', 'IR3']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='LandContour',
            value_set=['Lv1', 'Bnk', 'HLs', 'Low']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Utilities',
            value_set=['AllPub', 'NoSewr', 'NoSeWa', 'ELO']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='LotConfig',
            value_set=['Inside', 'Corner', 'IR2', 'IR3']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='LotConfig',
            value_set=['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='LandSlope',
            value_set=['Gtl', 'Mod', 'Sev']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Neighborhood',
            value_set=[
                'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr',
                'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR',
                'MeadowV', 'Mitchel', 'Names', 'NoRidge', 'NPkVill',
                'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer',
                'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'
            ]
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Condition1',
            value_set=['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Condition2',
            value_set=['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='BldgType',
            value_set=['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='HouseStyle',
            value_set=['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='OverallQual',
            value_set=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='OverallCond',
            value_set=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='RoofStyle',
            value_set=['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='RoofMatl',
            value_set=['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Exterior1st',
            value_set=[
                'AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock',
                'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other',
                'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd',
                'Wd Sdng', 'WdShing'
            ]
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Exterior2nd',
            value_set=[
                'AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock',
                'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other',
                'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd',
                'Wd Sdng', 'WdShing'
            ]
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='MasVnrType',
            value_set=['BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='ExterQual',
            value_set=['Ex', 'Gd', 'TA', 'Fa', 'Po']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='ExterCond',
            value_set=['Ex', 'Gd', 'TA', 'Fa', 'Po']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Foundation',
            value_set=['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='BsmtQual',
            value_set=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='BsmtCond',
            value_set=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='BsmtExposure',
            value_set=['Gd', 'Av', 'Mn', 'No', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='BsmtFinType1',
            value_set=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='BsmtFinType2',
            value_set=['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Heating',
            value_set=['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='HeatingQC',
            value_set=['Ex', 'Gd', 'TA', 'Fa', 'Po']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='CentralAir',
            value_set=['N', 'Y']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Electrical',
            value_set=['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='KitchenQual',
            value_set=['Ex', 'Gd', 'TA', 'Fa', 'Po']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Functional',
            value_set=['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='FireplaceQu',
            value_set=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='GarageType',
            value_set=['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='GarageFinish',
            value_set=['Fin', 'RFn', 'Unf', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='GarageQual',
            value_set=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='GarageCond',
            value_set=['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='PavedDrive',
            value_set=['Y', 'P', 'N']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='PoolQC',
            value_set=['Ex', 'Gd', 'TA', 'Fa', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='Fence',
            value_set=['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='MiscFeature',
            value_set=['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='SaleType',
            value_set=['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth']
        ),

        ge.expectations.ExpectColumnValuesToBeInSet(
            column='SaleCondition',
            value_set=['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
        ),]
    for expectation in my_expectation_list:
        suite.add_expectation(expectation)


    # run validation
    print('running validation')
    validation_definition = ge.ValidationDefinition(data=batch_definition, suite=suite, name='validation_batch_definition')
    validation_definition = context.validation_definitions.add(validation_definition)
    results = validation_definition.run(batch_parameters=batch_parameters)

    #process results
    failed_expectations = []
    failed_columns = []
    for r in results['results']:
        if not r['success']:
            expectation_type = r['expectation_config']['type']
            expectation_column = r['expectation_config']['kwargs']['column']
            failed_expectations.append(expectation_type)
            failed_columns.append(expectation_column)
    #print validation summary
    total_checks = len(results['results'])
    passed_checks = sum(1 for r in results['results'] if r['success'])
    failed_checks = total_checks - passed_checks

    if results['success']:
        print(f'Data validation PASSED: {passed_checks}/{total_checks} checks successful')
    else:
        print(f'Data validation FAILED: {failed_checks}/{total_checks} checks failed')
        print(f'Failed expectations: {failed_expectations}')
        print(f'Failed columns: {failed_columns}')
    return results['success'], failed_expectations
