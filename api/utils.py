import os
import json
from collections import defaultdict
import importlib

class CaseInsensitiveDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(str.lower, *args, **kwargs)

def get_keywords_for_pdf(filename, keyword_dict):
    filename_lower = filename.lower()
    ci_keyword_dict = CaseInsensitiveDict()
    for key, value in keyword_dict.items():
        ci_keyword_dict[key] = [v.lower() for v in value]
    for key, keywords in ci_keyword_dict.items():
        if key in filename_lower:
            return keywords
    return []

def load_keywords(keywords_path="constant/keywords.json"):
    with open(keywords_path, "r") as f:
        keywords_data = json.load(f)
    return keywords_data

def load_questions(questions_path="constant/constant.json"):
    with open(questions_path, "r") as f:
        questions_data = json.load(f)
    # Flatten questions from each section
    questions = [q for section in questions_data["data"] for q in section["questions"]]
    questions = [q if isinstance(q, str) else list(q.values())[0] for q in questions]
    return questions

def get_model(section_name):
    """Dynamically imports and returns the model class based on section_name."""
    module_name = f"models.{section_name.lower()}"  # Convert to lowercase (matches filename)
    class_name = "".join(word.capitalize() for word in section_name.split("_"))  # Convert to PascalCase
    
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)  # Get the class from the module
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading model {class_name}: {e}")
