// src/App.js - Main application component
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Dashboard from './components/Dashboard';
import AMPPPrediction from './components/AMPPPrediction';
import SideEffectPrediction from './components/SideEffectPrediction';
import ModelTraining from './components/ModelTraining';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app-container">
        <Navigation />
        <main className="content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/ampp-prediction" element={<AMPPPrediction />} />
            <Route path="/se-prediction" element={<SideEffectPrediction />} />
            <Route path="/training" element={<ModelTraining />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;