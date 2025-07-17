"""
PLS (Partial Least Squares) model adapter for PAD Analytics.

This module handles PLS model loading, prediction, and management,
including concentration prediction from region-based features.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import tempfile
import os
import csv

from .base_adapter import BaseAdapter


class PLSAdapter(BaseAdapter):
    """
    Adapter for PLS (Partial Least Squares) models.
    
    Handles loading PLS coefficient models, processing preprocessed feature data,
    and returning concentration prediction results.
    """
    
    def _get_model_info(self) -> Dict[str, Any]:
        """
        Get PLS model information from the PAD API or local registry.
        
        Returns:
            Dictionary containing model metadata
        """
        try:
            # Try to get model info from API using get_model function
            from .. import padanalytics as pad
            model_df = pad.get_model(self.model_id)
            
            if not model_df.empty:
                model_data = model_df.iloc[0]
                return {
                    'name': model_data.get('name', f'Model_{self.model_id}'),
                    'description': model_data.get('description', 'PLS Model'),
                    'model_url': model_data.get('weights_url', None),
                    'model_type': 'pls',
                    'file_extension': '.pkl'
                }
        except Exception as e:
            print(f"Warning: Could not fetch model info from API: {e}")
        
        # Fallback to static model mapping (in case API fails)
        pls_models = {
            18: {
                'name': '24fhiPLS1conc',
                'description': 'PLS Concentration Prediction Model',
                'model_url': None,  # Will be populated by API
                'model_type': 'pls',
                'file_extension': '.pkl'
            }
        }
        
        return pls_models.get(self.model_id, {
            'name': f'Unknown_PLS_Model_{self.model_id}',
            'description': 'Unknown PLS Model',
            'model_url': None,
            'model_type': 'pls',
            'file_extension': '.pkl'
        })
    
    def load_model(self) -> bool:
        """
        Load the PLS model coefficients.
        
        Returns:
            True if model was loaded successfully
        """
        try:
            # Download model if needed
            model_url = self.model_info.get('model_url')
            if not model_url:
                print(f"No model URL available for model {self.model_id}")
                return False
            
            model_path = self.download_model(model_url)
            if not model_path:
                return False
            
            # Load PLS coefficients from CSV file
            self.model = self._load_pls_coefficients(model_path)
            
            if self.model is None:
                return False
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Failed to load PLS model {self.model_id}: {e}")
            return False
    
    def predict_single(self, preprocessed_data: Dict[str, Any]) -> float:
        """
        Make a prediction for a single preprocessed data point.
        
        Args:
            preprocessed_data: Output from PLS preprocessor
            
        Returns:
            Concentration prediction as float
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self.validate_preprocessed_data(preprocessed_data):
            raise ValueError("Invalid preprocessed data format")
        
        try:
            # Get features
            features = preprocessed_data['features']
            
            # Get drug name for coefficient lookup
            drug_name = preprocessed_data.get('sample_name', '').lower()
            
            if drug_name not in self.model:
                print(f"Warning: Drug '{drug_name}' not found in model coefficients")
                return 0.0
            
            # Get coefficients for this drug
            coefficients = self.model[drug_name]
            
            # Calculate PLS concentration
            concentration = self._calculate_pls_concentration(features, coefficients)
            
            return float(concentration)
            
        except Exception as e:
            print(f"PLS prediction failed: {e}")
            raise
    
    def predict_batch(self, preprocessed_batch: List[Dict[str, Any]]) -> List[float]:
        """
        Make predictions for a batch of preprocessed data.
        
        Args:
            preprocessed_batch: List of preprocessed data dictionaries
            
        Returns:
            List of concentration predictions
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        for preprocessed_data in preprocessed_batch:
            try:
                result = self.predict_single(preprocessed_data)
                results.append(result)
            except Exception as e:
                print(f"Batch prediction failed for item: {e}")
                # Add placeholder result for failed prediction
                results.append(0.0)
        
        return results
    
    def get_expected_input_format(self) -> str:
        """
        Get the expected input format for this adapter.
        
        Returns:
            String describing expected input format
        """
        return "Features array with 360 elements (12 lanes × 10 regions × 3 colors)"
    
    def validate_preprocessed_data(self, preprocessed_data: Dict[str, Any]) -> bool:
        """
        Validate that preprocessed data meets PLS adapter requirements.
        
        Args:
            preprocessed_data: Preprocessed data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if not super().validate_preprocessed_data(preprocessed_data):
            return False
        
        # Check for PLS specific fields
        if 'features' not in preprocessed_data:
            return False
        
        # Check preprocessor type
        if preprocessed_data.get('preprocessor_type') != 'pls':
            return False
        
        # Validate features array
        features = preprocessed_data['features']
        if not isinstance(features, (list, np.ndarray)):
            return False
        
        if len(features) != 360:  # 12 lanes × 10 regions × 3 colors
            return False
        
        return True
    
    def _load_pls_coefficients(self, model_path: str) -> Optional[Dict[str, List[float]]]:
        """
        Load PLS coefficients from CSV file.
        
        Args:
            model_path: Path to the PLS coefficients file
            
        Returns:
            Dictionary mapping drug names to coefficient arrays
        """
        try:
            coefficients = {}
            
            with open(model_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                
                for row in reader:
                    if len(row) < 2:
                        continue
                    
                    drug_name = row[0].lower()
                    coeffs = []
                    
                    for j in range(1, len(row)):
                        try:
                            coeffs.append(float(row[j]))
                        except ValueError:
                            print(f"Warning: Invalid coefficient '{row[j]}' for drug '{drug_name}'")
                            coeffs.append(0.0)
                    
                    coefficients[drug_name] = coeffs
            
            return coefficients
            
        except Exception as e:
            print(f"Failed to load PLS coefficients: {e}")
            return None
    
    def _calculate_pls_concentration(self, features: List[float], coefficients: List[float]) -> float:
        """
        Calculate PLS concentration using features and coefficients.
        
        Args:
            features: List of extracted features (360 elements)
            coefficients: List of PLS coefficients
            
        Returns:
            Calculated concentration
        """
        if len(coefficients) == 0:
            return 0.0
        
        # Start with offset (first coefficient)
        concentration = coefficients[0]
        
        # Add weighted features
        coeff_index = 1
        for i, feature_value in enumerate(features):
            if coeff_index < len(coefficients):
                concentration += float(feature_value) * coefficients[coeff_index]
                coeff_index += 1
        
        return concentration
    
    def get_supported_drugs(self) -> List[str]:
        """
        Get list of drugs supported by this PLS model.
        
        Returns:
            List of drug names
        """
        if not self.is_loaded:
            return []
        
        return list(self.model.keys())
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded model.
        
        Returns:
            Dictionary containing model summary information
        """
        if not self.is_loaded:
            return {"status": "Model not loaded"}
        
        return {
            "model_id": self.model_id,
            "model_name": self.model_info.get('name', 'Unknown'),
            "model_type": "pls",
            "supported_drugs": self.get_supported_drugs(),
            "num_features": 360,
            "feature_pattern": "12 lanes × 10 regions × 3 colors",
            "model_file": self.model_info.get('model_url', 'Unknown')
        }
    
    def predict_for_drug(self, features: List[float], drug_name: str) -> float:
        """
        Make a prediction for a specific drug.
        
        Args:
            features: List of extracted features
            drug_name: Name of the drug to predict for
            
        Returns:
            Concentration prediction
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        drug_name = drug_name.lower()
        
        if drug_name not in self.model:
            raise ValueError(f"Drug '{drug_name}' not supported by this model")
        
        coefficients = self.model[drug_name]
        return self._calculate_pls_concentration(features, coefficients)