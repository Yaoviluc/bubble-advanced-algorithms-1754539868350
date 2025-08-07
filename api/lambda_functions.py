
import json
import boto3
import numpy as np
from decimal import Decimal
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def bubble_ml_processor(event, context):
    """Main ML processing function for Bubble integration"""
    try:
        body = json.loads(event.get('body', '{}')) if isinstance(event.get('body'), str) else event.get('body', {})
        
        operation = body.get('operation')
        data = body.get('data')
        parameters = body.get('parameters', {})
        
        logger.info(f"Processing operation: {operation}")
        
        result = {}
        
        if operation == 'predict':
            result = perform_prediction(data, parameters)
        elif operation == 'cluster':
            result = perform_clustering(data, parameters)
        elif operation == 'optimize':
            result = perform_optimization(data, parameters)
        else:
            result = {'error': f'Unsupported operation: {operation}'}
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(result, default=decimal_default)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }

def perform_prediction(data, parameters):
    """Advanced prediction using ensemble methods"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        
        df = pd.DataFrame(data)
        target_column = parameters.get('target_column', df.columns[-1])
        features = df.drop(columns=[target_column])
        target = df[target_column]
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features_scaled, target)
        predictions = model.predict(features_scaled)
        
        return {
            'model_type': "Random Forest",
            'predictions': predictions.tolist(),
            'feature_importance': model.feature_importances_.tolist(),
            'feature_names': features.columns.tolist()
        }
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

def perform_clustering(data, parameters):
    """Perform clustering analysis"""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        
        df = pd.DataFrame(data)
        n_clusters = parameters.get('n_clusters', 3)
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)
        
        return {
            'clusters': clusters.tolist(),
            'centroids': kmeans.cluster_centers_.tolist(),
            'inertia': float(kmeans.inertia_)
        }
    except Exception as e:
        return {'error': f'Clustering failed: {str(e)}'}

def perform_optimization(data, parameters):
    """Perform optimization tasks"""
    try:
        from scipy.optimize import minimize
        import numpy as np
        
        # Simple optimization example
        def objective_function(x):
            return x[0]**2 + x[1]**2
        
        initial_guess = parameters.get('initial_guess', [0, 0])
        result = minimize(objective_function, initial_guess)
        
        return {
            'optimized_values': result.x.tolist(),
            'minimum_value': float(result.fun),
            'success': bool(result.success)
        }
    except Exception as e:
        return {'error': f'Optimization failed: {str(e)}'}
        