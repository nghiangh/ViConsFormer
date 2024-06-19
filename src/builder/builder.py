from model.viconsformer_net import ViConsFormer_Model

def build_model(config):
    if config['model']['type_model']=='viconsformer':
        return ViConsFormer_Model(config)

    