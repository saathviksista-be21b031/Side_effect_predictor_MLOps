// src/components/SideEffectPrediction.js
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
  
  const handleSideEffectChange = (e) => {
    setSideEffect(e.target.value);
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError("Please select a CSV file");
      return;
    }
    
    if (!sideEffect.trim()) {
      setError("Please enter a side effect");
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      setResult(null);
      
      const response = await predictSideEffect(sideEffect, file);
      setResult(response);
    } catch (err) {
      setError(`Error making prediction: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="se-prediction">
      <h1>Side Effect Prediction</h1>
      
      <div className="instructions">
        <h2>Instructions</h2>
        <p>
          Enter a specific side effect and upload a CSV file containing your drug data.
          The file should have two columns with no header row.
        </p>
        <p>
          The model will predict the probability of the specified side effect for the uploaded drug.
        </p>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="side-effect">Side Effect:</label>
          <input 
            id="side-effect"
            type="text" 
            value={sideEffect} 
            onChange={handleSideEffectChange}
            placeholder="Enter side effect name" 
            disabled={loading}
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="drug-file">Upload Drug Data (CSV):</label>
          <input 
            id="drug-file"
            type="file" 
            accept=".csv" 
            onChange={handleFileChange}
            disabled={loading}
          />
        </div>
        
        <button type="submit" disabled={loading || !file || !sideEffect.trim()}>
          {loading ? 'Processing...' : 'Predict Probability'}
        </button>
      </form>
      
      {result && (
        <div className="prediction-result">
          <h2>Prediction Result</h2>
          <pre className="result-text">{result}</pre>
        </div>
      )}
    </div>
  );
};

export default SideEffectPrediction;
