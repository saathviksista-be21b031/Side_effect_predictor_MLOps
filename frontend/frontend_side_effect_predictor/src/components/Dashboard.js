// Updated Dashboard.js with polished layout and UI wording
import React from 'react';
import { Link } from 'react-router-dom';

const Dashboard = () => {
  return (
    <div className="dashboard">
      <h1>Welcome to DrugSideFX</h1>

      <div className="dashboard-intro">
        <p>
          Predict adverse effects of drug compounds using advanced machine learning models.
          Upload your molecular data and get actionable insights instantly.
        </p>
      </div>

      <div className="feature-cards">
        <div className="feature-card">
          <h2>Top 5 Side Effects</h2>
          <p>
            Upload your drug's data and receive the most likely 5 side effects predicted by our ensemble model.
          </p>
          <Link to="/ampp-prediction" className="feature-button">
            Predict Top 5
          </Link>
        </div>

        <div className="feature-card">
          <h2>Specific Side Effect</h2>
          <p>
            Wondering if a drug causes a particular side effect? Type it in and check its likelihood.
          </p>
          <Link to="/se-prediction" className="feature-button">
            Check Specific Effect
          </Link>
        </div>

        <div className="feature-card">
          <h2>Model Training</h2>
          <p>
            Retrain our prediction model using newly added drug data to enhance accuracy.
          </p>
          <Link to="/training" className="feature-button">
            Retrain Now
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
