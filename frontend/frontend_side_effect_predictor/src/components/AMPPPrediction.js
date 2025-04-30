// Refined AMPPPrediction.js
import React, { useState } from 'react';
import { predictAMPP } from '../services/api';

const AMPPPrediction = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please upload a valid CSV file');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);
      const response = await predictAMPP(file);
      setResult(response);
    } catch (err) {
      setError(`Prediction failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ampp-prediction refined-section">
      <h1>Top 5 Side Effect Prediction</h1>

      <div className="instructions">
        <h2>How to Use</h2>
        <p>Upload your drug data in CSV format with two feature columns. The model returns the five most likely side effects.</p>
      </div>

      {error && <div className="error-message">{error}</div>}

      <form onSubmit={handleSubmit} className="refined-form">
        <div className="form-group">
          <label htmlFor="drug-file">Upload CSV File:</label>
          <input id="drug-file" type="file" accept=".csv" onChange={handleFileChange} disabled={loading} />
        </div>

        <button type="submit" disabled={loading || !file}>
          {loading ? 'Predicting...' : 'Predict Side Effects'}
        </button>
      </form>

      {result && (
        <div className="prediction-result">
          <h2>Prediction Results</h2>
          <pre className="result-text">{result}</pre>
        </div>
      )}
    </div>
  );
};

export default AMPPPrediction;

