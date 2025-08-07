
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
import redis
from celery import Celery
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'complex-algorithms-secret')

redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(redis_url)

celery = Celery(app.name, broker=redis_url)
celery.conf.update(app.config)

class AdvancedMicroservice:
    def __init__(self):
        self.cache_timeout = 3600
        
    def complex_matrix_operations(self, matrices, operation='multiply'):
        """Perform complex matrix operations"""
        try:
            matrix_arrays = [np.array(matrix) for matrix in matrices]
            
            if operation == 'multiply':
                result = matrix_arrays[0]
                for matrix in matrix_arrays[1:]:
                    result = np.dot(result, matrix)
                    
            elif operation == 'eigenvalues':
                result = []
                for matrix in matrix_arrays:
                    eigenvals, eigenvects = np.linalg.eig(matrix)
                    result.append({
                        'eigenvalues': eigenvals.real.tolist(),
                        'eigenvectors': eigenvects.real.tolist()
                    })
                    
            else:
                return {'error': f'Unsupported operation: {operation}'}
            
            return {
                'operation': operation,
                'result': result.tolist() if hasattr(result, 'tolist') else result,
                'computation_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Matrix operation failed: {str(e)}'}

microservice = AdvancedMicroservice()

@app.route('/')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Bubble Advanced Algorithms Microservice',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/matrix', methods=['POST'])
def matrix_operations():
    """Handle complex matrix operations"""
    try:
        data = request.get_json()
        matrices = data.get('matrices', [])
        operation = data.get('operation', 'multiply')
        
        if not matrices:
            return jsonify({'error': 'No matrices provided'}), 400
        
        result = microservice.complex_matrix_operations(matrices, operation)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Matrix operations error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['POST'])
def data_analytics():
    """Handle data analytics operations"""
    try:
        data = request.get_json()
        dataset = data.get('dataset', [])
        operation = data.get('operation', 'statistics')
        
        if operation == 'statistics':
            df = pd.DataFrame(dataset)
            stats = df.describe().to_dict()
            return jsonify({'statistics': stats})
        
        return jsonify({'error': 'Unknown analytics operation'})
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') == 'development')
        