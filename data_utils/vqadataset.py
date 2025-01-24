from typing import List, Dict
from torch.utils.data import Dataset
import json
import os
from utils import preprocess_sentence
from utils.instance import Instance

class VQADataset(Dataset):
    def __init__(self, annotation_path):
        with open(annotation_path, 'r') as file:
            json_data = json.load(file)
        
        self.annotations = self.load_annotations(json_data)
    
    def load_annotations(self, json_data) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
                question = ann["question"]
                answer = ann["answers"][0]
                annotation = {
                    "id": ann['id'],
                    "image_id": str(ann['image_id']),
                    "question": preprocess_sentence(question),
                    "answer": preprocess_sentence(answer),
                }
                annotations.append(annotation)

        return annotations
    
    def __getitem__(self, index):
        item = self.annotations[index]
        
        return Instance(**item)
    
    def __len__(self) -> int:
        return len(self.annotations)
