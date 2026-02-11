import yaml
import os

class Config:
    def __init__(self, config_path=None):
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value
    
    def save(self, save_path):
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f)
