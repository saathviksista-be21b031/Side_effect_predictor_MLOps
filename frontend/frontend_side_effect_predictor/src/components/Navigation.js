// src/components/Navigation.js
import React from 'react';
import { NavLink } from 'react-router-dom';

const Navigation = () => {
  return (
    <nav className="sidebar">
      <div className="logo">
        <h2>Drug ML Platform</h2>
      </div>
      <ul className="nav-links">
        <li>
          <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
            Dashboard
          </NavLink>
        </li>
        <li>
          <NavLink to="/ampp-prediction" className={({ isActive }) => isActive ? 'active' : ''}>
            AMPP Prediction
          </NavLink>
        </li>
        <li>
          <NavLink to="/se-prediction" className={({ isActive }) => isActive ? 'active' : ''}>
            Side Effect Prediction
          </NavLink>
        </li>
        <li>
          <NavLink to="/training" className={({ isActive }) => isActive ? 'active' : ''}>
            Model Training
          </NavLink>
        </li>
      </ul>
    </nav>
  );
};

export default Navigation;





