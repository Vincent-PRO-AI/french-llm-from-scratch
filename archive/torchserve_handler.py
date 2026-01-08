"""
🔥 TorchServe Handler for French GPT-2
Custom handler for serving the French GPT-2 model
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ts.torch_handler.base_handler import BaseHandler
import json
import logging

logger = logging.getLogger(__name__)

class FrenchGPT2Handler(BaseHandler):
    """
    Handler for serving French GPT-2 model with TorchServe
    """
    
    def initialize(self, ctx):
        """
        Initialize the model and tokenizer
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        
        logger.info(f"Loading model from {model_dir}")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        
        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.eval()
        
        # Check if GPU available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model parameters: {self.model.num_parameters() / 1e6:.0f}M")
        
        self.initialized = True
    
    def preprocess(self, data):
        """
        Preprocess the input data
        """
        if isinstance(data, dict):
            payload = data
        else:
            payload = json.loads(data[0]["body"])
        
        prompt = payload.get("prompt", "")
        max_length = payload.get("max_length", 100)
        temperature = payload.get("temperature", 0.7)
        top_p = payload.get("top_p", 0.9)
        
        logger.info(f"Prompt: {prompt[:50]}...")
        
        return {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p
        }
    
    def inference(self, data):
        """
        Run inference
        """
        prompt = data["prompt"]
        max_length = data["max_length"]
        temperature = data["temperature"]
        top_p = data["top_p"]
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        tokens_generated = output_ids.shape[1] - input_ids.shape[1]
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "tokens_generated": tokens_generated
        }
    
    def postprocess(self, inference_output):
        """
        Postprocess the output
        """
        return [inference_output]
