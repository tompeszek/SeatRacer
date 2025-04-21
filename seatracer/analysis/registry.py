import streamlit as st
from typing import Dict, List, Type, Any, Optional

class ModelRegistry:
    """Registry for analysis model types with improved persistence for hot-reloading."""
    
    @classmethod
    def _get_models(cls):
        """Get models from session state or initialize if needed"""
        if "registered_models" not in st.session_state:
            st.session_state.registered_models = {}
        return st.session_state.registered_models
    
    @classmethod
    def register(
        cls,
        key: str,
        name: str,
        description: str = "",
        uses_custom_weighting: bool = False,
        can_have_stern_bias: bool = False,
        show_athletes: bool = False,
        order: int = 99,
        recommended: bool = False,
        ) -> Any:
        """
        Decorator to register a model class in the registry.
        This version works better with Streamlit's hot-reloading.
        """
        def decorator(model_class):
            # Store directly in session state
            models = cls._get_models()
            models[key] = {
                "key": key,
                "name": name,
                "description": description,
                "class": model_class,
                "uses_custom_weighting": uses_custom_weighting,
                "can_have_stern_bias": can_have_stern_bias,
                "show_athletes": show_athletes,
                "order": order,
            }
            
            # Add metadata to the class itself
            model_class.model_key = key
            model_class.model_name = name 
            model_class.model_description = description
            model_class.uses_custom_weighting = uses_custom_weighting
            model_class.can_have_stern_bias = can_have_stern_bias
            model_class.show_athletes = show_athletes
            model_class.order = order
            model_class.recommended = recommended
                
            return model_class
        return decorator
    
    @classmethod
    def get_model_class(cls, key: str) -> Type:
        """Get the class for the specified model type."""
        models = cls._get_models()
        
        model_info = models.get(key)
        if model_info is None:
            # Default to the first registered model or raise an error
            if models:
                return next(iter(models.values()))["class"]
            raise KeyError(f"No models registered and model key '{key}' not found")
        return model_info["class"]
    
    @classmethod
    def get_model_class_by_name(cls, name: str) -> Type:
        """Get the class for the specified model name."""
        models = cls._get_models()
        
        # Find the model with the matching name
        for key, model_info in models.items():
            if model_info["name"] == name:
                return model_info["class"]
        
        # Default to the first registered model or raise an error
        if models:
            return next(iter(models.values()))["class"]
        raise KeyError(f"No models registered and model name '{name}' not found")
    
    @classmethod
    def get_model_uses_custom_weighting(cls, key: str) -> bool:
        """Check if the model uses custom weighting."""
        models = cls._get_models()
        
        model_info = models.get(key)
        if model_info is None:
            return False
        return model_info["uses_custom_weighting"]
    
    @classmethod
    def get_model_choices(cls) -> List[Dict[str, str]]:
        """Get model choices for UI display, sorted by order then alphabetically."""
        models = cls._get_models()

        # Create choices with order included, filtering out non-recommended models
        choices = [
            {
                "value": info["key"], 
                "label": info["name"],
                "order": getattr(info["class"], "order", 99)
            } 
            for info in models.values() 
            if getattr(info["class"], "recommended", False) == True  # Explicit comparison to ensure filtering works
        ]
        
        # Sort by order, then alphabetically
        sorted_choices = sorted(choices, key=lambda x: (x["order"], x["label"]))
        
        # Return only the needed fields
        return [{"value": choice["value"], "label": choice["label"]} for choice in sorted_choices]
    
    @classmethod
    def get_all_models(cls) -> Dict:
        """Get all registered models."""
        return cls._get_models()