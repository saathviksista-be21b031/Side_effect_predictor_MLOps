// src/components/PipelineVisualization.js
import React, { useEffect, useState } from 'react';

const PipelineVisualization = () => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Optional: Replace this with a real API call
    const fetchMockStatus = async () => {
      try {
        setLoading(true);
        await new Promise((r) => setTimeout(r, 1000)); // simulate delay
        setStatus({
          status: 'Success',
          duration: '14s',
          timestamp: new Date().toLocaleString(),
        });
      } catch (err) {
        setError('Failed to fetch pipeline status');
      } finally {
        setLoading(false);
      }
    };

    fetchMockStatus();
  }, []);

  return (
    <div className="refined-section">
      <h1>ML Pipeline Visualization</h1>

      <div className="instructions">
        <h2>Status Summary</h2>
        {loading ? (
          <p>Loading pipeline status...</p>
        ) : error ? (
          <div className="error-message">{error}</div>
        ) : (
          <div>
            <p><strong>Status:</strong> ✅ {status.status}</p>
            <p><strong>Duration:</strong> {status.duration}</p>
            <p><strong>Timestamp:</strong> {status.timestamp}</p>
          </div>
        )}
      </div>

      <div className="instructions">
        <h2>DVC DAG (Pipeline Structure)</h2>
        <img
  src="http://localhost:8000/pipeline/dvc_dag.svg"
  alt="DVC DAG"
  style={{
    width: '100%',
    maxWidth: '1000px',       // increased from 600px
    height: '500px',
    borderRadius: '8px',
    border: '1px solid #ccc',
    display: 'block',
    margin: '1rem auto',
  }}
/>
      </div>

      <div className="instructions">
        <h2>Grafana Dashboard</h2>
        <iframe
  title="Grafana"
  src="http://localhost:3000/d/dekfdzy7gzu9sb/sample-dash?orgId=1&refresh=10s"
  width="100%"
  height="500"
  style={{
    border: '1px solid #ccc',
    borderRadius: '8px',
  }}
  allowFullScreen
></iframe>

        <p style={{ fontSize: '0.9rem', color: '#555', marginTop: '0.5rem' }}>
          Make sure Grafana is running and accessible at the above address.
        </p>
      </div>
    </div>
  );
};

export default PipelineVisualization;
