import xgboost as xgb

from data import get_num_classes
from sklearn.metrics import accuracy_score, roc_auc_score

def train_xgboost(X_train, y_train):
    
    num_classes = get_num_classes.num_classes()
    
    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes, # Dynamically set based on mappings
        'eval_metric': 'logloss',
        'learning_rate': 0.25,
        'max_depth': 23,
        'subsample': 0.8,
        'tree_method': 'hist',
        'device': 'cuda',
        'seed': 42,
    }
    
    # Prepare data for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
    
    # Set the number of boosting rounds
    num_boost_round = 200
    
    # Train the model using xgb.train
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    
    return model