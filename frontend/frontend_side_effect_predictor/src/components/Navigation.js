// Updated Navigation.js with refined sidebar design
import React from 'react';
import { NavLink } from 'react-router-dom';

const Navigation = () => {
  return (
    <nav className="sidebar refined-sidebar">
      <div className="logo">
        <h2>DrugSideFX</h2>
      </div>
      <ul className="nav-links">
        <li>
          <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
            Dashboard
          </NavLink>
        </li>
        <li>
          <NavLink to="/ampp-prediction" className={({ isActive }) => isActive ? 'active' : ''}>
            Top 5 Side Effects
          </NavLink>
        </li>
        <li>
          <NavLink to="/se-prediction" className={({ isActive }) => isActive ? 'active' : ''}>
            Specific Side Effect
          </NavLink>
        </li>
        <li>
          <NavLink to="/training" className={({ isActive }) => isActive ? 'active' : ''}>
            Retrain Model
          </NavLink>
        </li>
	<li>
  	  <NavLink to="/pipeline" className={({ isActive }) => isActive ? 'active' : ''}>
            ML Pipeline
  	  </NavLink>
        </li>

      </ul>
    </nav>
  );
};

export default Navigation;






