// src/components/ModelTraining.js
import React, { useState } from 'react';
import { trainModel } from '../services/api';

const ModelTraining = () => {
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const handleTrainModel = async () => {
    try {
      setTraining(true);
      setError(null);
      setResult(null);
      
      const response = await trainModel();
      setResult(response);
    } catch (err) {
      setError(`Error training model: ${err.message}`);
    } finally {
      setTraining(false);
    }
  };
  
  return (
    <div className="model-training">
      <h1>Model Training</h1>
      
      <div className="training-info">
        <h2>Training Information</h2>
        <p>
          This will trigger the training process for the AMPP model. The model will be trained
          using the default dataset configuration specified in the backend.
        </p>
        <p>
          Training may take several minutes to complete. Please be patient and do not refresh
          the page while training is in progress.
        </p>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="training-controls">
        <button 
          className="train-button"
          onClick={handleTrainModel}
          disabled={training}
        >
          {training ? 'Training in Progress...' : 'Train AMPP Model'}
        </button>
      </div>
      
      {training && (
        <div className="training-progress">
          <div className="loading-spinner"></div>
          <p>Training in progress. This may take several minutes...</p>
        </div>
      )}
      
      {result && (
        <div className="training-result">
          <h2>Training Result</h2>
          <pre className="result-text">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default ModelTraining;