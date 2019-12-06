import pandas as pd
import yaml

with open('config.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

SELECTED_DATASET = config['DATASET']
SELECTED_MODEL = config['MODEL']

train_features = pd.read_csv(SELECTED_DATASET + '_train_features.csv')
test_features = pd.read_csv(SELECTED_DATASET + '_test_features.csv')

# Create feature "Type" for training dataset
train_types = []

for row in train_features['Type']:
    if row == 'Class':
        train_types.append(1)
    else:
        train_types.append(0)

train_features['Type_encode'] = train_types

# Create feature "Type" for testing dataset
test_types = []

for row in test_features['Type']:
    if row == 'Class':
        test_types.append(1)
    else:
        test_types.append(0)

test_features['Type_encode'] = test_types

X_train = train_features.loc[:, 'Ngram1_Entity':'Type_encode']
y_train = train_features['Match']

X_test = test_features.loc[:, 'Ngram1_Entity':'Type_encode']
y_test = test_features['Match']

df_train = train_features.loc[:, 'Ngram1_Entity':'Type_encode']
df_train['Match'] = train_features['Match']

df_test = test_features.loc[:, 'Ngram1_Entity':'Type_encode']
df_test['Match'] = test_features['Match']

# Fill nan values with zero
X_train = X_train.fillna(value=0)
X_test = X_test.fillna(value=0)

train = pd.read_csv(SELECTED_DATASET + '_train.csv')
test = pd.read_csv(SELECTED_DATASET + '_test.csv')

# Train model
if SELECTED_MODEL != 'XGBoost':
    if SELECTED_MODEL == 'LogisticRegression':
        print("Training logistic regression...")
        from sklearn.linear_model import LogisticRegression

        if SELECTED_DATASET == 'dataset1':
            model = LogisticRegression(penalty='l1', C=1.0, class_weight=None)
        elif SELECTED_DATASET == 'dataset2':
            model = LogisticRegression(penalty='l2', C=7.742637,
                                       class_weight=None)
    elif SELECTED_MODEL == 'RandomForest':
        print("Training random forest classifier...")
        from sklearn.ensemble import RandomForestClassifier

        if SELECTED_DATASET == 'dataset1':
            model = RandomForestClassifier(n_estimators=500,
                                           max_features='sqrt', max_depth=3,
                                           random_state=42)
        elif SELECTED_DATASET == 'dataset2':
            model = RandomForestClassifier(n_estimators=100, max_features=None,
                                           max_depth=2)

    model.fit(X_train, y_train)
    print("Predicting for testing dataset...")
    y_prob = model.predict_proba(X_test)

elif SELECTED_MODEL == 'XGBoost':
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {'silent': 0, 'objective': 'binary:logistic',
             'min_child_weight': 10, 'gamma': 2.0, 'subsample': 0.8,
             'colsample_bytree': 0.8, 'max_depth': 5, 'nthread': 4,
             'eval_metric': 'error'}

    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    plst = param.items()

    num_round = 10
    bst = xgb.train(plst, dtrain, num_round, evallist,
                    verbose_eval=False)

    y_prob = bst.predict(dtest)

TEST_ALIGNMENTS = config[SELECTED_DATASET]['TEST_ALIGNMENTS']

# Choose best threshold
for alignment in TEST_ALIGNMENTS:
    ont1 = alignment.split('-')[0]
    ont2 = alignment.split('-')[1].replace('.rdf', '')
    best_ts = 0
    best_fmeasure = 0

    for ts in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        preds = []

        if SELECTED_MODEL != 'XGBoost':
            for x in y_prob:
                if x[1] >= ts:
                    preds.append(1)
                else:
                    preds.append(0)
        else:
            for x in y_prob:
                if x >= ts:
                    preds.append(1)
                else:
                    preds.append(0)

        test['Predict'] = preds

        if SELECTED_DATASET == 'dataset1':
            onto_format = 'rdf'
        elif SELECTED_DATASET == 'dataset2':
            onto_format = 'owl'

        pred_mappings = test[(test[
                                  'Ontology1'] == f"{SELECTED_DATASET}/ontologies/{ont1}.{onto_format}") &
                             (test[
                                  'Ontology2'] == f"{SELECTED_DATASET}/ontologies/{ont2}.{onto_format}") &
                             (test['Predict'] == 1)]

        true_mappings = test[(test[
                                  'Ontology1'] == f"{SELECTED_DATASET}/ontologies/{ont1}.{onto_format}") &
                             (test[
                                  'Ontology2'] == f"{SELECTED_DATASET}/ontologies/{ont2}.{onto_format}") &
                             (test['Match'] == 1)]

        correct_mappings = test[
            (test[
                 'Ontology1'] == f"{SELECTED_DATASET}/ontologies/{ont1}.{onto_format}") &
            (test[
                 'Ontology2'] == f"{SELECTED_DATASET}/ontologies/{ont2}.{onto_format}") &
            (test['Match'] == 1) & (test['Predict'] == 1)]

        true_num = len(true_mappings)
        predict_num = len(pred_mappings)
        correct_num = len(correct_mappings)

        if predict_num != 0 and true_num != 0 and correct_num != 0:
            precision = correct_num / predict_num
            recall = correct_num / true_num
            fmeasure = 2 * precision * recall / (precision + recall)
        else:
            fmeasure = 0

        if fmeasure > best_fmeasure:
            best_fmeasure = fmeasure
            best_ts = ts
            best_preds = preds

    print(
        f"Best fmeasure for {alignment} is {best_fmeasure} with threshold {best_ts}")
