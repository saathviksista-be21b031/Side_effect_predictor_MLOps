// Refined ModelTraining.js
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
      setError(`Training failed: ${err.message}`);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="model-training refined-section">
      <h1>Retrain Side Effect Model</h1>

      <div className="instructions">
        <h2>What Happens?</h2>
        <p>
          This process retrains the ML model using the latest available data. It may take a few minutes. Please be patient.
        </p>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="training-controls">
        <button className="train-button" onClick={handleTrainModel} disabled={training}>
          {training ? 'Training Model...' : 'Start Training'}
        </button>
      </div>

      {training && (
        <div className="training-progress">
          <div className="loading-spinner"></div>
          <p>Model is currently being trained...</p>
        </div>
      )}

      {result && (
        <div className="training-result">
          <h2>Training Outcome</h2>
          <pre className="result-text">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default ModelTraining;
