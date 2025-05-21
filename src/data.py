import os
import pandas as pd


from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Prompt:
    template: str
    clinical_case: str
    discharge_summary: str
    guidelines: str = ""
    examples: str = ""
    extra_fields: Dict[str, Any] = field(default_factory=dict)  # Supports extra dynamic fields

    def __post_init__(self):
        """Generate the formatted prompt dynamically, including examples and extra fields."""

        # Combine predefined fields and extra fields for template formatting
        format_dict = {
            "guidelines": self.guidelines,
            "examples": self.examples,
            "clinical_case": self.clinical_case,
            "discharge_summary": self.discharge_summary,
            **self.extra_fields  # Inject extra fields dynamically
        }

        # Format the template safely
        self.text = self.template.format(**format_dict)

    def __str__(self):
        return self.text

def create_examples(examples: List[Dict[str, str]], example_template: str) -> str:
    if examples == []:
        return ""
    
    examples = "\n".join([Prompt(
                                template=example_template, 
                                clinical_case=example["clinical_case"], 
                                discharge_summary=example["discharge_summary"],
                                extra_fields={
                                    "human_score": example["human_score"],
                                    "auto_score": example["auto_score"]
                                    }
                             ).text for example in examples])
    return examples

def extract_txt(path, filename):
    total_path = os.path.join(path, filename)
    return filename.split(".")[0], open(total_path, "r").read()

def files_to_df(path, extensions=["txt"]):
    files = [x for x in os.listdir(path) if x.split(".")[-1] in extensions]
    data = [extract_txt(path, f) for f in files]
    return pd.DataFrame(data, columns=["filenameid", "text"])

def read_document(doc:str) -> str:
    """
    Read document from a file if exists, otherwise return the text
    """
    return open(doc, "r").read() if os.path.exists(doc) else doc
