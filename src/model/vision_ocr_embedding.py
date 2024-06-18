import torch
from torch import nn
import numpy as np

class VisionOcrEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()      
        self.linear_det_features = nn.Linear(config.d_det, config.d_model)
        self.linear_rec_features = nn.Linear(config.d_rec, config.d_model)
        self.linear_boxes = nn.Linear(4,config.d_model)

        self.layer_norm_det = nn.LayerNorm(config.d_model)
        self.layer_norm_rec = nn.LayerNorm(config.d_model)
        self.layer_norm_boxes = nn.LayerNorm(config.d_model)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.cuda_device=config.cuda_device
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')

    def forward(self,ocr_info):
        det_features = torch.stack([det["det_features"] for det in ocr_info]).to(self.device)
        rec_features = torch.stack([rec["rec_features"] for rec in ocr_info]).to(self.device)
        boxes = torch.stack([box["boxes"] for box in ocr_info]).to(self.device)
        
        det_features=self.linear_det_features(det_features)
        rec_features=self.linear_rec_features(rec_features)
        boxes = self.linear_boxes(boxes)
        
        ocr_features = self.layer_norm_det(det_features) + self.layer_norm_rec(rec_features) + self.layer_norm_boxes(boxes)
        ocr_features = self.dropout(self.gelu(ocr_features))
        return ocr_features
    
class SpatialModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_left_x = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.bottom_right_x = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.top_left_y = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.bottom_right_y = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.width_emb = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.height_emb = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.max_2d_position_embeddings=config.max_2d_position_embeddings
        self.cuda_device=config.cuda_device
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')

    def forward(self, ocr_info):
        coordinates = torch.stack([bbox["boxes"] for bbox in ocr_info]).to(torch.int).to(self.device)
        w=torch.stack([w["width"] for w in ocr_info]).to(torch.int).to(self.device)
        h=torch.stack([h["height"] for h in ocr_info]).to(torch.int).to(self.device)

        coordinates=torch.where(coordinates >= self.max_2d_position_embeddings, self.max_2d_position_embeddings - 1, coordinates)
        h=torch.where(h>=self.max_2d_position_embeddings,self.max_2d_position_embeddings-1,h)
        w=torch.where(w>=self.max_2d_position_embeddings,self.max_2d_position_embeddings-1,w)
        
        top_left_x_feat = self.top_left_x(coordinates[:, :, 0])
        top_left_y_feat = self.top_left_y(coordinates[:, :, 1])
        bottom_right_x_feat = self.bottom_right_x(coordinates[:, :, 2])
        bottom_right_y_feat = self.bottom_right_y(coordinates[:, :, 3])
        width_feat = self.width_emb(w.unsqueeze(1))
        height_feat = self.height_emb(h.unsqueeze(1))
        layout_feature = top_left_x_feat + top_left_y_feat + \
            bottom_right_x_feat + bottom_right_y_feat + width_feat + height_feat
        return layout_feature

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()

        d_model = config.d_model
        h = config.num_attention_heads
        d_k = d_model // h
        d_v = d_model // h

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            att += attention_mask
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out, att

class SpatialCirclePosition(ScaledDotProductAttention):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.cuda_device=config.cuda_device
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.dist_embedding = nn.Embedding(
            num_embeddings=config.num_distances,
            embedding_dim=config.num_attention_heads
        )
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def calculate_distances(self, patch_x, patch_y):
        # Tính toán khoảng cách
        dx = patch_x.unsqueeze(1) - patch_x.unsqueeze(2)
        dy = patch_y.unsqueeze(1) - patch_y.unsqueeze(2)

        distances = torch.sqrt(dx**2 + dy**2)

        return distances

    def patch(self, ocr_boxes: torch.Tensor, image_sizes: torch.Tensor) -> tuple:
        """
            ocr_boxes: (bs, n_ocr, 4)
            image_sizes: (bs, 4)
            return: (bs, x_centroid, y_centroid)
        """
        
        size_per_area = (image_sizes[:, :2] // 11).unsqueeze(1) # (bs, 1, 2)
        lower_bounds = torch.arange(
            start=0, 
            end=11, 
            step=1
        ).unsqueeze(0).repeat(ocr_boxes.shape[0], 1).to(ocr_boxes.device) # (bs, 11)
        higher_bounds = lower_bounds + 1
        # width boundaries
        width_lower_bounds = lower_bounds * size_per_area[:, :, 0]
        width_higher_bounds = higher_bounds * size_per_area[:, :, 0]
        # height boundaries
        height_lower_bounds = lower_bounds * size_per_area[:, :, 1]
        height_higher_bounds = higher_bounds * size_per_area[:, :, 1]

        # reshape the bounds so that we can broadcast the dimension
        width_lower_bounds = width_lower_bounds.unsqueeze(1) # (bs, 1, 11, 2)
        width_higher_bounds = width_higher_bounds.unsqueeze(1) # (bs, 1, 11, 2)
        height_lower_bounds = height_lower_bounds.unsqueeze(1) # (bs, 1, 11, 2)
        height_higher_bounds = height_higher_bounds.unsqueeze(1) # (bs, 1, 11, 2)
        ocr_boxes = ocr_boxes.unsqueeze(-2) # (bs, n_ocr, 1, 4)
        ocr_x_centroid = (ocr_boxes[:, :, :, 0] + ocr_boxes[:, :, :, 2]) // 2
        ocr_y_centroid = (ocr_boxes[:, :, :, 1] + ocr_boxes[:, :, :, 3]) // 2
        selected_x_centroid = torch.logical_and(torch.le(width_lower_bounds, ocr_x_centroid), torch.le(ocr_x_centroid, width_higher_bounds)) # (bs, n_ocr, 11)
        selected_y_centroid = torch.logical_and(torch.le(height_lower_bounds, ocr_y_centroid), torch.le(ocr_y_centroid, height_higher_bounds)) # (bs, n_ocr, 11)
        # determine the appropriate patch
        selected_x_centroid = selected_x_centroid.to(torch.float).argmax(dim=-1) # (bs, n_ocr)
        selected_y_centroid = selected_y_centroid.to(torch.float).argmax(dim=-1) # (bs, n_orc)

        return selected_x_centroid, selected_y_centroid

    def forward(self, features, info) -> torch.Tensor:
        features=self.layer_norm(features)
        image_sizes = []
        boxes =[]
        for item in info:
            size=torch.tensor([item['width'],item['height'],item['width'],item['height']])
            image_sizes.append(size)
            boxes.append(item["boxes"]*size)
        image_sizes = torch.stack(image_sizes).to(self.device)
        boxes = torch.stack(boxes).to(self.device)
        bs, nq, _ = boxes.shape
        patch_x, patch_y = self.patch(boxes, image_sizes)
        dist=self.calculate_distances(patch_x,patch_y)*2
        dist = self.dist_embedding(dist.long()).view(bs, nq, nq, -1).permute((0, -1, 1, 2)) # (bs, h, nq, nq)

        q = self.fc_q(features).view(bs, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (bs, h, nq, d_k)
        k = self.fc_k(features).view(bs, nq, self.h, self.d_k).permute(0, 2, 3, 1)  # (bs, h, d_k, nk)
        v = self.fc_v(features).view(bs, nq, self.h, self.d_v).permute(0, 2, 1, 3)  # (bs, h, nk, d_v)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (bs, h, nq, nq)
        att = torch.softmax(att + dist, dim=-1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(bs, nq, self.h * self.d_v)  # (bs, nq, h*d_v)
        out = self.fc_o(out)  # (bs, nq, d_model)

        return out, att

class SemanticOCREmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.max_scene_text = config.max_scene_text

        self.linear_det_features = nn.Linear(config.d_det, config.d_model)
        self.linear_rec_features = nn.Linear(config.d_rec, config.d_model)
        self.linear_boxes = nn.Linear(4,config.d_model)

        self.layer_norm_det = nn.LayerNorm(config.d_model)
        self.layer_norm_rec = nn.LayerNorm(config.d_model)
        self.layer_norm_bboxes = nn.LayerNorm(config.d_model)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self.cuda_device=config.cuda_device
        self.device = torch.device(f'{self.cuda_device}' if torch.cuda.is_available() else 'cpu')

    def pad_tensor(self, tensor: torch.Tensor, max_len: int, value):
        if max_len == 0:
            tensor = torch.zeros((0, tensor.shape[-1]))
        else:
            pad_value_tensor = torch.zeros((max_len-tensor.shape[0], tensor.shape[-1])).fill_(value).to(self.device)
            tensor = torch.cat([tensor, pad_value_tensor], dim=0)
        return tensor

    def forward(self,
                ocr_info,
                ocr_embs):

        det_features = torch.stack([det["det_features"] for det in ocr_info]).to(self.device)
        rec_features = torch.stack([rec["rec_features"] for rec in ocr_info]).to(self.device)
        ocr_boxes = torch.stack([box["boxes"] for box in ocr_info]).to(self.device)
        
        ocr_feature_emb = (self.layer_norm_det(self.linear_det_features(det_features))+
                        self.layer_norm_rec(self.linear_rec_features(rec_features)))
        ocr_box_emb = self.layer_norm_bboxes(self.linear_boxes(ocr_boxes))
        
        ocr_features = ocr_feature_emb + ocr_box_emb + ocr_embs
        ocr_features = self.dropout(self.gelu(ocr_features))

        return ocr_features