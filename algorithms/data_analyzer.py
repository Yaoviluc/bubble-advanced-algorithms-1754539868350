
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import networkx as nx
from collections import defaultdict
import json

class ComplexDataAnalyzer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def advanced_statistical_analysis(self, data):
        """Comprehensive statistical analysis"""
        df = pd.DataFrame(data)
        
        results = {}
        
        for column in df.select_dtypes(include=[np.number]):
            col_data = df[column].dropna()
            
            # Advanced statistical measures
            results[column] = {
                'descriptive_stats': {
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'mode': float(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                    'std': float(col_data.std()),
                    'variance': float(col_data.var()),
                    'skewness': float(stats.skew(col_data)),
                    'kurtosis': float(stats.kurtosis(col_data)),
                    'range': float(col_data.max() - col_data.min())
                },
                'distribution_tests': {
                    'shapiro_wilk': {
                        'statistic': float(stats.shapiro(col_data)[0]),
                        'p_value': float(stats.shapiro(col_data)[1])
                    },
                    'jarque_bera': {
                        'statistic': float(stats.jarque_bera(col_data)[0]),
                        'p_value': float(stats.jarque_bera(col_data)[1])
                    }
                },
                'outliers': self.detect_outliers(col_data),
                'percentiles': {
                    'p25': float(np.percentile(col_data, 25)),
                    'p50': float(np.percentile(col_data, 50)),
                    'p75': float(np.percentile(col_data, 75)),
                    'p95': float(np.percentile(col_data, 95)),
                    'p99': float(np.percentile(col_data, 99))
                }
            }
        
        return results
    
    def detect_outliers(self, data):
        """Detect outliers using IQR method"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return {
            'count': len(outliers),
            'values': outliers.tolist()
        }

def lambda_handler(event, context):
    analyzer = ComplexDataAnalyzer()
    
    operation = event.get('operation')
    data = event.get('data')
    
    try:
        if operation == 'statistical_analysis':
            result = analyzer.advanced_statistical_analysis(data)
        else:
            result = {'error': f'Unknown operation: {operation}'}
        
        return {
            'statusCode': 200,
            'body': json.dumps(result, default=str),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
        