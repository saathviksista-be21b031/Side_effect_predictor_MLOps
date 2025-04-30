// Refined SideEffectPrediction.js
import React, { useState } from 'react';
import { predictSideEffect } from '../services/api';

const SideEffectPrediction = () => {
  const [file, setFile] = useState(null);
  const [sideEffect, setSideEffect] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleSideEffectChange = (e) => setSideEffect(e.target.value);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !sideEffect.trim()) {
      setError('Please upload a CSV and specify a side effect');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);
      const response = await predictSideEffect(sideEffect, file);
      setResult(response);
    } catch (err) {
      setError(`Prediction failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="se-prediction refined-section">
      <h1>Specific Side Effect Prediction</h1>

      <div className="instructions">
        <h2>Instructions</h2>
        <p>Enter a side effect name and upload your drug data as CSV to check the likelihood.</p>
      </div>

      {error && <div className="error-message">{error}</div>}

      <form onSubmit={handleSubmit} className="refined-form">
        <div className="form-group">
          <label htmlFor="side-effect">Side Effect:</label>
          <input
            id="side-effect"
            type="text"
            value={sideEffect}
            onChange={handleSideEffectChange}
            placeholder="e.g., nausea"
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="drug-file">Upload CSV File:</label>
          <input
            id="drug-file"
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            disabled={loading}
          />
        </div>

        <button type="submit" disabled={loading || !file || !sideEffect.trim()}>
          {loading ? 'Predicting...' : 'Check Side Effect'}
        </button>
      </form>

      {result && (
        <div className="prediction-result">
          <h2>Result</h2>
          <pre className="result-text">{result}</pre>
        </div>
      )}
    </div>
  );
};

export default SideEffectPrediction;

