"""
Neural Network model adapter for PAD Analytics.

This module handles neural network model loading, prediction, and management,
including TensorFlow Lite model inference and drug classification.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import tempfile
import os

from .base_adapter import BaseAdapter


class NeuralNetworkAdapter(BaseAdapter):
    """
    Adapter for Neural Network models using TensorFlow Lite.
    
    Handles loading TensorFlow Lite models, processing preprocessed image data,
    and returning drug classification results with confidence scores.
    """
    
    def _get_model_info(self) -> Dict[str, Any]:
        """
        Get neural network model information from the PAD API or local registry.
        
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
                    'description': model_data.get('description', 'Neural Network Model'),
                    'model_url': model_data.get('weights_url', None),
                    'model_type': 'neural_network',
                    'file_extension': '.tflite'
                }
        except Exception as e:
            print(f"Warning: Could not fetch model info from API: {e}")
        
        # Fallback to static model mapping (in case API fails)
        nn_models = {
            16: {
                'name': '24fhiNN1classifyAPI',
                'description': 'Neural Network Drug Classification',
                'model_url': None,  # Will be populated by API
                'model_type': 'neural_network',
                'file_extension': '.tflite'
            },
            17: {
                'name': '24fhiNN1concAPI',
                'description': 'Neural Network Concentration Prediction',
                'model_url': None,  # Will be populated by API
                'model_type': 'neural_network',
                'file_extension': '.tflite'
            },
            19: {
                'name': '24fhiNN1concAPIv2',
                'description': 'Neural Network Concentration Prediction v2',
                'model_url': None,  # Will be populated by API
                'model_type': 'neural_network',
                'file_extension': '.tflite'
            },
            20: {
                'name': 'ChemoPAD NN training 2024',
                'description': 'ChemoPAD Neural Network Training 2024',
                'model_url': None,  # Will be populated by API
                'model_type': 'neural_network', 
                'file_extension': '.tflite'
            }
        }
        
        return nn_models.get(self.model_id, {
            'name': f'Unknown_NN_Model_{self.model_id}',
            'description': 'Unknown Neural Network Model',
            'model_url': None,
            'model_type': 'neural_network',
            'file_extension': '.tflite'
        })
    
    def load_model(self) -> bool:
        """
        Load the TensorFlow Lite model.
        
        Returns:
            True if model was loaded successfully
        """
        try:
            import tensorflow as tf
            
            # Download model if needed
            model_url = self.model_info.get('model_url')
            if not model_url:
                print(f"No model URL available for model {self.model_id}")
                return False
            
            model_path = self.download_model(model_url)
            if not model_path:
                return False
            
            # Load TensorFlow Lite model
            self.model = tf.lite.Interpreter(model_path=model_path)
            self.model.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()
            
            # Load drug labels
            self.labels = self._load_drug_labels()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Failed to load neural network model {self.model_id}: {e}")
            return False
    
    def predict_single(self, preprocessed_data: Dict[str, Any]) -> Tuple[str, float, float]:
        """
        Make a prediction for a single preprocessed data point.
        
        Args:
            preprocessed_data: Output from neural network preprocessor
            
        Returns:
            Tuple of (drug_name, confidence, energy)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self.validate_preprocessed_data(preprocessed_data):
            raise ValueError("Invalid preprocessed data format")
        
        try:
            import tensorflow as tf
            
            # Get preprocessed image array
            image_array = preprocessed_data['preprocessed_image']
            
            # Ensure correct shape and type
            if image_array.shape != (1, 454, 454, 3):
                raise ValueError(f"Expected image shape (1, 454, 454, 3), got {image_array.shape}")
            
            # Ensure correct dtype (TensorFlow Lite expects FLOAT32)
            image_array = image_array.astype(np.float32)
            
            # Set input tensor
            self.model.set_tensor(self.input_details[0]["index"], image_array)
            
            # Run inference
            self.model.invoke()
            
            # Get output
            result = self.model.get_tensor(self.output_details[0]["index"])
            
            # Process result
            num_label = np.argmax(result[0])
            
            # Check if the predicted index is within bounds
            if num_label >= len(self.labels):
                print(f"Warning: Predicted label index {num_label} is out of bounds for {len(self.labels)} labels")
                return ("unknown", 0.0, 0.0)
            
            drug_name = self.labels[num_label]
            
            confidence = tf.nn.softmax(result[0])[num_label].numpy()
            energy = tf.reduce_logsumexp(result[0], -1).numpy()
            
            return (drug_name, float(confidence), float(energy))
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, preprocessed_batch: List[Dict[str, Any]]) -> List[Tuple[str, float, float]]:
        """
        Make predictions for a batch of preprocessed data.
        
        Args:
            preprocessed_batch: List of preprocessed data dictionaries
            
        Returns:
            List of (drug_name, confidence, energy) tuples
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
                results.append(("unknown", 0.0, 0.0))
        
        return results
    
    def get_expected_input_format(self) -> str:
        """
        Get the expected input format for this adapter.
        
        Returns:
            String describing expected input format
        """
        return "Preprocessed image array with shape (1, 454, 454, 3) and dtype float32"
    
    def validate_preprocessed_data(self, preprocessed_data: Dict[str, Any]) -> bool:
        """
        Validate that preprocessed data meets neural network adapter requirements.
        
        Args:
            preprocessed_data: Preprocessed data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if not super().validate_preprocessed_data(preprocessed_data):
            return False
        
        # Check for neural network specific fields
        if 'preprocessed_image' not in preprocessed_data:
            return False
        
        # Check preprocessor type
        if preprocessed_data.get('preprocessor_type') != 'neural_network':
            return False
        
        # Validate image array
        image_array = preprocessed_data['preprocessed_image']
        if not isinstance(image_array, np.ndarray):
            return False
        
        if image_array.shape != (1, 454, 454, 3):
            return False
        
        return True
    
    def _load_drug_labels(self) -> List[str]:
        """
        Load drug labels for classification.
        
        Returns:
            List of drug names in the order expected by the model
        """
        try:
            # Try to get labels from model API (same as original predict function)
            from .. import padanalytics as pad
            
            # Get model data to access labels
            model_df = pad.get_model(self.model_id)
            if not model_df.empty:
                labels = model_df.labels.values[0]
                
                # Check if labels are for concentration (numbers) or classification (strings)
                try:
                    # If labels are numbers, this is a concentration model
                    labels = list(map(int, labels))
                    # For concentration models, we need drug names from dataset
                    dataset_name = pad.get_dataset_name_from_model_id(self.model_id)
                    dataset = pad.CachedDataset(dataset_name)
                    metadata = dataset.load_dataset_metadata()
                    return sorted(metadata['sample_name'].unique().tolist())
                except:
                    # If labels are strings, this is a classification model
                    labels = list(map(pad.standardize_names, labels))
                    return labels
            
        except Exception as e:
            print(f"Warning: Could not load drug labels from model API: {e}")
        
        # Fallback: try to get from dataset
        try:
            from .. import padanalytics as pad
            dataset_name = pad.get_dataset_name_from_model_id(self.model_id)
            dataset = pad.CachedDataset(dataset_name)
            metadata = dataset.load_dataset_metadata()
            return sorted(metadata['sample_name'].unique().tolist())
        except Exception as e:
            print(f"Warning: Could not load drug labels from dataset: {e}")
            
            # Final fallback to common drug names
            return [
                "acetaminophen", "amoxicillin", "aspirin", "atorvastatin",
                "ciprofloxacin", "dexamethasone", "hydroxychloroquine",
                "ibuprofen", "metformin", "prednisolone"
            ]
    
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
            "model_type": "neural_network",
            "input_shape": self.input_details[0]["shape"].tolist(),
            "output_shape": self.output_details[0]["shape"].tolist(),
            "num_labels": len(self.labels),
            "labels": self.labels,
            "model_file": self.model_info.get('model_url', 'Unknown')
        }