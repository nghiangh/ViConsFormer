import torch
from torch.amp.grad_scaler import GradScaler
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from model.viconsformer import ViConsFormer
from eval_metric.evaluate import ScoreCalculator
from data_utils.vqadataset import VQADataset
from utils import countTrainableParameters, countParameters
from utils.logging_utils import setup_logger
import json

class Task:
    def __init__(self, config):

        self.logger = setup_logger()
        self.initialize_hyperparameters(config)
        self.create_dataloaders(config)
        
        # set the device for the training process
        cuda_device=config.train.cuda_device
        device = torch.device(f'{cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.model = ViConsFormer(config).to(device)
        total_params = countParameters(self.model)
        trainable_param = countTrainableParameters(self.model)
        self.logger.info(f'Trainable parameters: {100*(trainable_param / total_params):.2f}%')
        
        self.compute_score = ScoreCalculator()
        learning_rate = config.train.learning_rate
        weight_decay=config.train.weight_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scaler = GradScaler(self.device)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)

    def initialize_hyperparameters(self, config):
        # training hyperparameters
        self.num_epochs = config.train.num_train_epochs
        self.patience = config.train.patience
        self.save_path = os.path.join(config.train.output_dir, config.model.type_model)
        self.best_metric= config.train.metric_for_best_model
        # precision floating points
        if config.train.precision == 'float32':
            self.cast_dtype=torch.float32
        else:
            self.cast_dtype=torch.float16

    def create_dataloaders(self, config):
        num_worker = config.data.num_worker

        train_images = config.data.images_train_folder
        train_annotations = config.data.train_dataset
        train_set = VQADataset(train_annotations, train_images)
        train_batch = config.train.per_device_train_batch_size
        self.train_loader = DataLoader(
            dataset=train_set, 
            batch_size=train_batch, 
            num_workers=num_worker,
            shuffle=True
        )

        val_images = config.data.images_val_folder
        val_annotations = config.data.val_dataset
        val_set = VQADataset(val_annotations, val_images)
        valid_batch = config.train.per_device_valid_batch_size
        self.val_loader = DataLoader(
            dataset=val_set,
            batch_size=valid_batch,
            num_workers=num_worker,
            shuffle=True
        )

        test_images = config.infer.images_test_folder
        test_annotations = config.infer.test_dataset
        test_set = VQADataset(test_annotations, test_images)
        test_batch = config.infer.per_device_eval_batch_size
        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=test_batch,
            num_workers=num_worker,
            shuffle=False
        )

    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
        
        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Loaded saved model')
            initial_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']
        else:
            best_score = 0.
            
        threshold=0
        self.model.train()
        epoch = 0
        while True:
            valid_em = 0.
            valid_f1 =0.
            train_loss = 0.
            with tqdm(desc='Epoch %d - Training' % (epoch+1), unit='it', total=len(self.train_loader)) as pbar:
                for it, item in enumerate(self.train_loader):
                    with torch.autocast(device_type='cuda', dtype=self.cast_dtype, enabled=True):
                        _, loss = self.model(item.question, item.image_id, item.answer)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    train_loss += loss.item()
                    pbar.set_postfix(loss=train_loss / (it + 1))
                    pbar.update()
                self.scheduler.step()
                train_loss /=len(self.train_loader)

                pbar.set_postfix({"loss": train_loss / (it+1)})
                pbar.update()
            
            with torch.no_grad():
                with tqdm(desc='Epoch %d - Validating' % (epoch+1), unit='it', total=len(self.val_loader)) as pbar:
                    for it, item in enumerate(tqdm(self.val_loader)):
                        with torch.autocast(device_type='cuda', dtype=self.cast_dtype, enabled=True):
                            pred_answers = self.base_model(item.question, item.image_id)    
                            valid_em+=self.compute_score.em(item.answer, pred_answers)
                            valid_f1+=self.compute_score.f1_token(item.answer, pred_answers)
                    valid_em /= len(self.val_loader)
                    valid_f1 /= len(self.val_loader)

            self.logger.info(f"Train loss: {train_loss:.4f}")
            self.logger.info(f"Validation - EM: {valid_em:.4f}, F1: {valid_f1:.4f}")

            if self.best_metric =='em':
                score=valid_em
            if self.best_metric=='f1':
                score=valid_f1

            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'score': score}, os.path.join(self.save_path, 'last_model.pth'))
            
            # save the best model
            if epoch > 0 and score <= best_score:
              threshold += 1
            else:
              threshold = 0

            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    # 'optimizer_state_dict': self.optimizer.state_dict(),
                    'score':score}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with {self.best_metric} of {score:.4f}")
            
            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break

    def get_predictions(self):
        # Load the model
        self.logger.info("Loadding best model...")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Obtain the prediction from the model
        self.logger.info("Obtaining predictions...")
        test_set =self.dataloader.load_test(self.with_answer)
        y_preds={}
        self.model.eval()
        with torch.no_grad():
            for item in tqdm(test_set):
                with torch.autocast(device_type='cuda', dtype=self.cast_dtype, enabled=True):
                    answers = self.model(item['question'],item['image_id'])
                    for i in range(len(answers)):
                        if isinstance(item['id'][i],torch.Tensor):
                            ids=item['id'].tolist()
                        else:
                            ids=item['id']
                        y_preds[str(ids[i])] = answers[i]
        with open(os.path.join(self.save_path,'results.json'), 'w', encoding='utf-8') as r:
            json.dump(y_preds, r, ensure_ascii=False, indent=4)
