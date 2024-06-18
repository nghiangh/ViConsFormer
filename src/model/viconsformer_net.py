from typing import List, Dict
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from src.model.encoding_text import T5_Encode_Feature, MultimodalBackbone
from model.vision_obj_encoding import Vision_Encode_Obj_Feature
from model.vision_ocr_encoding import Vision_Encode_Ocr_Feature

class ViConsFormer_Model(nn.Module):
    def __init__(self,config: Dict):

        super().__init__()
        self.text_encoder = T5_Encode_Feature(config)
        self.vision_encoder_ocr = Vision_Encode_Ocr_Feature(config)
        self.vision_encoder_obj = Vision_Encode_Obj_Feature(config)
        self.cuda_device=config['train']['cuda_device']
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_embedding']['text_encoder'])
        self.backbone = MultimodalBackbone(config)
        self.with_image = config['train']['with_image']
        self.max_scene_text = config['ocr_embedding']['max_scene_text']
        self.max_bbox = config['obj_embedding']['max_bbox']
        self.generator_args ={
            'max_length': config['generator_args']['max_length'],
            'min_length': config['generator_args']['min_length'],
            'num_beams': config['generator_args']['num_beams'],
            'length_penalty': config['generator_args']['length_penalty'],
            'no_repeat_ngram_size': config['generator_args']['no_repeat_ngram_size'],
            'early_stopping': config['generator_args']['early_stopping'],
        }

    def forward(self, questions: List[str], images: List[str], labels: List[str] = None):
        ocr_info = self.vision_encoder_ocr(images)
        obj_info = self.vision_encoder_obj(images)
        ocr_list=[ocr['texts']  for ocr in ocr_info]
        obj_list = [obj['object_list'] for obj in obj_info]

        ocr_texts = []
        for ocr_tokens in ocr_list:
            if ocr_tokens == []:
                ocr_tokens = ["<unk>"]
            ocr_tokens = " ".join(ocr_tokens)
            ocr_texts.append(ocr_tokens)

        ocr_ids = self.tokenizer(ocr_texts,
                                    padding="max_length",
                                    max_length=self.max_scene_text,
                                    truncation=True,
                                    return_tensors="pt",
                                    add_special_tokens=True).input_ids
        ocr_ids = ocr_ids.to(self.device)
        
        obj_tags = []
        for obj_tokens in obj_list:
            if obj_tokens == []:
                obj_tokens = ["<unk>"]
            obj_tokens = " ".join(obj_tokens)
            obj_tags.append(obj_tokens)
        obj_ids = self.tokenizer(obj_tags,
                                    padding="max_length",
                                    max_length=self.max_bbox,
                                    truncation=True,
                                    return_tensors="pt",
                                    add_special_tokens=True).input_ids
        obj_ids = obj_ids.to(self.device)
        inputs = self.text_encoder(questions,ocr_texts,labels)
        inputs.update({'ocr_info': ocr_info,
                        'obj_info': obj_info,
                        'ocr_ids': ocr_ids,
                        'obj_ids': obj_ids})

        if labels is not None:
            outputs = self.backbone(**inputs)
            return outputs.logits, outputs.loss
        else:
            pred_ids=self.backbone.generate(**inputs,**self.generator_args)
            pred_tokens=self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            return pred_tokens
