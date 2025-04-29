// src/services/api.js
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Make a prediction using the AMPP model
 * @param {File} file - CSV file with drug data
 * @returns {Promise<string>} - Prediction result as text
 */
export const predictAMPP = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to make prediction');
    }
    
    return await response.text();
  } catch (error) {
    console.error('AMPP Prediction error:', error);
    throw error;
  }
};

/**
 * Predict probability of a specific side effect
 * @param {string} sideEffect - The side effect to predict
 * @param {File} file - CSV file with drug data
 * @returns {Promise<string>} - Prediction result as text
 */
export const predictSideEffect = async (sideEffect, file) => {
  const formData = new FormData();
  formData.append('side_effect', sideEffect);
  formData.append('file', file);
  
  try {
    const response = await fetch(`${API_BASE_URL}/predict_se`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to predict side effect');
    }
    
    return await response.text();
  } catch (error) {
    console.error('Side Effect Prediction error:', error);
    throw error;
  }
};

/**
 * Train the AMPP model
 * @returns {Promise<Object>} - Training result
 */
export const trainModel = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/train`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to train model');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Model Training error:', error);
    throw error;
  }
};