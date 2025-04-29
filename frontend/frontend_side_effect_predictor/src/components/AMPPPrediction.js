// src/components/AMPPPrediction.js
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
      setError("Please select a CSV file");
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      setResult(null);
      
      const response = await predictAMPP(file);
      setResult(response);
    } catch (err) {
      setError(`Error making prediction: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="ampp-prediction">
      <h1>AMPP Prediction</h1>
      
      <div className="instructions">
        <h2>Instructions</h2>
        <p>
          Upload a CSV file containing your drug data. The file should have two columns:
          the first column for one set of features and the second column for another set.
          No header row is required.
        </p>
        <p>
          The model will predict the top 5 most probable side effects for the uploaded drug data.
        </p>
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      <form onSubmit={handleSubmit}>
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
        
        <button type="submit" disabled={loading || !file}>
          {loading ? 'Processing...' : 'Predict Side Effects'}
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
