// src/components/Dashboard.js
import React from 'react';
import { Link } from 'react-router-dom';

const Dashboard = () => {
  return (
    <div className="dashboard">
      <h1>Drug ML Platform</h1>
      
      <div className="dashboard-intro">
        <p>
          Welcome to the Drug ML Platform. This application provides machine learning capabilities
          for predicting drug side effects and interactions using advanced ML models.
        </p>
      </div>
      
      <div className="feature-cards">
        <div className="feature-card">
          <h2>AMPP Prediction</h2>
          <p>
            Predict the most probable side effects for a drug using our AMPP model.
            Upload your drug data and get the top 5 most probable side effects.
          </p>
          <Link to="/ampp-prediction" className="feature-button">
            Make AMPP Prediction
          </Link>
        </div>
        
        <div className="feature-card">
          <h2>Side Effect Prediction</h2>
          <p>
            Predict the probability of a specific side effect for a drug.
            Specify the side effect and upload your drug data for analysis.
          </p>
          <Link to="/se-prediction" className="feature-button">
            Predict Side Effect
          </Link>
        </div>
        
        <div className="feature-card">
          <h2>Model Training</h2>
          <p>
            Train or retrain the AMPP model with new data to improve prediction accuracy.
            Monitor training progress and performance metrics.
          </p>
          <Link to="/training" className="feature-button">
            Train Model
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;