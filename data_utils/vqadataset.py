from typing import List, Dict
from torch.utils.data import Dataset
import json
import os
from utils import preprocess_sentence
from utils.instance import Instance

class VQADataset(Dataset):
    def __init__(self, annotation_path, image_path, remove_accents_rate=0, use_word_seg=False):
        with open(annotation_path, 'r') as file:
            json_data = json.load(file)
        
        self.annotations = self.load_annotations(json_data, image_path, remove_accents_rate, use_word_seg)
    
    def load_annotations(self, json_data, image_path) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
                question = ann["question"].replace('?','')
                if "answers" in answer:
                    answer = preprocess_sentence(ann['answers'][0])
                else:
                    answer = None
                question = preprocess_sentence(question)
                annotation = {
                    "id": ann['id'],
                    "image_id": os.path.join(image_path,str(ann['image_id'])),
                    "question": question,
                    "answer": answer,
                }
                annotations.append(annotation)

        return annotations
    
    def __getitem__(self, index):
        item = self.annotations[index]
        
        return Instance(item)
    
    def __len__(self) -> int:
        return len(self.annotations)
