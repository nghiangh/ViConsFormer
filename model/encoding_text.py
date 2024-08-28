import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig
from typing import List, Dict
from model.viconsformer import ViConsFormer
from peft import LoraConfig, get_peft_model, TaskType

def MultimodalBackbone(config):
    tokenizer = AutoTokenizer.from_pretrained(config['text_embedding']['text_encoder'])
    model_config = AutoConfig.from_pretrained(config['model']['backbone'])
    model_config.update({'max_2d_position_embeddings' : config['ocr_embedding']['max_2d_position_embeddings'],
                            'num_distances' : config['ocr_embedding']['num_distances'],
                            'd_det': config['ocr_embedding']['d_det'],
                            'd_rec': config['ocr_embedding']['d_rec'],
                            'd_obj': config['obj_embedding']['d_obj'],
                            'd_grid': config['obj_embedding']['d_obj'],
                            "vocab_size": tokenizer.vocab_size,
                            "num_deps": config["text_embedding"]["num_deps"],
                            "max_scene_text": config["ocr_embedding"]["max_scene_text"],
                            "max_bbox": config["obj_embedding"]["max_bbox"],
                            "cuda_device":config['train']['cuda_device']})

    embedding = ViConsFormer.from_pretrained(config["model"]["backbone"], 
                                                config=model_config,
                                                ignore_mismatched_sizes=True)

    if config['text_embedding']['use_lora']==True:
        lora_config = LoraConfig(
            r=config['text_embedding']['lora_r'],
            lora_alpha=config['text_embedding']['lora_alpha'],
            # target_modules=config['text_embedding']['lora_target_modules'],
            lora_dropout=config['text_embedding']['lora_dropout'],
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        # self.embedding = prepare_model_for_int8_training(self.embedding)
        embedding = get_peft_model(embedding, lora_config)
    return embedding

class T5_Encode_Feature(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_embedding']['text_encoder'])
        self.padding = config['tokenizer']['padding']
        self.max_input_length = config["tokenizer"]["max_input_length"]
        self.max_target_length = config["tokenizer"]["max_target_length"]
        self.truncation = config["tokenizer"]["truncation"]
        self.cuda_device=config['train']['cuda_device']
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_text: List[str], text_pair: List[str]=None, answers: List[str]=None):
        encodings = self.tokenizer(
                                input_text,text_pair,
                                padding= self.padding,
                                max_length=self.max_input_length,
                                truncation=self.truncation,
                                return_tensors='pt',
                            ).to(self.device)
        if answers is not None:
            encoded_targets = self.tokenizer(
                                    answers,
                                    padding= self.padding,
                                    max_length=self.max_target_length,
                                    truncation=self.truncation,
                                    return_tensors='pt',
                                ).to(self.device)
            labels_input_ids = encoded_targets["input_ids"].clone()
            decoder_attention_mask=encoded_targets["attention_mask"].clone()
            labels_input_ids[decoder_attention_mask== self.tokenizer.pad_token_id]=-100
            encodings.update({
                'labels': labels_input_ids,
                'decoder_attention_mask': decoder_attention_mask,
            })
        return encodings
