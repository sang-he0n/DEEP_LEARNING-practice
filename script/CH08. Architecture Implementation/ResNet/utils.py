# 00. Import libraies
import tqdm
import sklearn
import torch
import models

# 01. Define `SupervisedLearning` class
class SupervisedLearning() :
    def __init__(self, model_nm:str, use_checkpoint_tf:bool, checkpoint_path:str=None) :
        self.device = self._check_device()
        self.model = models.check_model(model_nm=model_nm)
        self.criterion = torch.nn.CrossEntropyLoss()
        (self.checkpoint_path, self.model, self.init_epoch, 
         self.train_cost_hist, self.best_train_cost, 
         self.val_cost_hist, self.best_val_cost) = self._use_checkpoint(
            use_checkpoint_tf=use_checkpoint_tf,
            checkpoint_path=checkpoint_path, 
            model_nm=model_nm,
            model=self.model,
            device=self.device
        )
        self.model.to(device=self.device)
    def _check_device(self) -> torch.device :
        if torch.cuda.is_available() :
            device = torch.device(device='cuda')
        elif torch.backends.mps.is_available() :
            device = torch.device(device='mps')
        else :
            device = torch.device(device='cpu') 
        return device
    def _use_checkpoint(self, use_checkpoint_tf:bool, checkpoint_path:str, model_nm:str, model:torch.nn.Module, device:torch.device) -> tuple : 
        init_epoch = 0
        train_cost_hist = []
        best_train_cost = float('inf')
        val_cost_hist = []
        best_val_cost = float('inf')
        if use_checkpoint_tf == True : 
            if checkpoint_path is None : 
                checkpoint_path = f'checkpoint/{model_nm}Best.pt'
            print(f'>> Checkpoint Path is "{checkpoint_path}".')
            try : 
                checkpoint = torch.load(f=checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
                init_epoch = checkpoint['best_epoch']
                train_cost_hist = checkpoint['train_cost_hist']
                best_train_cost = train_cost_hist[-1]
                val_cost_hist = checkpoint['val_cost_hist']
                best_val_cost = val_cost_hist[-1]
                print(f'>> Loaded pretrained model successfully.')
                print(f'  - Best(Last) Epoch={init_epoch}, Best Train Loss={best_train_cost}, Best Validation Loss={best_val_cost}.')
            except : 
                print(f'>> Failed to Load pretrained model. Loaded default model.')
                pass
        else : 
            pass
        output = (checkpoint_path, model, init_epoch, train_cost_hist, best_train_cost, val_cost_hist, best_val_cost)
        return output
    def _split_train_val(self, loader:torch.utils.data.DataLoader, val_ratio:float) -> tuple :
        dataset = loader.dataset
        total_size = len(dataset)
        val_size = int(total_size*val_ratio)
        train_size = total_size - val_size
        train_subset, val_subset = torch.utils.data.random_split(dataset=dataset, lengths=[train_size, val_size])
        train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=loader.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_subset, batch_size=loader.batch_size, shuffle=False)
        output = (train_loader, val_loader)
        return output
    def train(self, train_loader:torch.utils.data.DataLoader, epoch_num:int, learning_rate:float, l2_rate:float) :
        print('>> Training start.')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=l2_rate)
        train_loader, val_loader = self._split_train_val(loader=train_loader, val_ratio=0.3)
        best_epoch = self.init_epoch
        train_batch_len = len(train_loader)
        val_batch_len = len(val_loader)
        progress_bar = tqdm.trange(self.init_epoch+1, epoch_num+1)
        for epoch in progress_bar :
            # ---- Train ---- #
            train_cost = 0.0
            for inputs, targets in train_loader :
                inputs = inputs.to(device=self.device)
                targets = targets.to(device=self.device)
                optimizer.zero_grad()
                preds = self.model(x=inputs)
                train_loss = self.criterion(input=preds, target=targets)
                train_loss.backward()
                optimizer.step()
                train_cost += train_loss.item()
            train_cost = train_cost / train_batch_len
            self.train_cost_hist.append(train_cost)
            # ---- _____ ---- # 
            
            # ---- Validation ---- # 
            val_cost = 0.0
            self.model.eval()
            with torch.no_grad() :
                for inputs, targets in val_loader :
                    inputs = inputs.to(device=self.device)
                    targets = targets.to(device=self.device)
                    preds = self.model(x=inputs)
                    val_loss = self.criterion(input=preds, target=targets)
                    val_cost += val_loss.item()
            val_cost = val_cost / val_batch_len
            self.val_cost_hist.append(val_cost)
            self.model.train()
            if val_cost <= self.best_val_cost :
                best_epoch = epoch
                self.best_val_cost = val_cost
                torch.save(
                    obj={
                        'model'           : self.model.state_dict(),
                        'optimizer'       : optimizer.state_dict(),
                        'best_epoch'      : epoch,
                        'train_cost_hist' : self.train_cost_hist,
                        'val_cost_hist'   : self.val_cost_hist 
                    }, 
                    f=self.checkpoint_path
                )
            # ---- __________ ---- # 

            progress_bar.set_postfix(ordered_dict={
                'last_train_cost' : train_cost, 
                'last_val_cost'   : val_cost,
                'best_val_epoch'  : best_epoch,
                'best_val_cost'   : self.best_val_cost
            })
        print('>> Training End.')
    def eval(self, loader:torch.utils.data.DataLoader) -> dict :
        targets = []
        preds = []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in loader :
                inputs = inputs.to(device=self.device)
                labels = labels.to(device=self.device)
                preds.extend(
                    torch.argmax(input=self.model(x=inputs), dim=1).to(device='cpu').numpy()
                )
                targets.extend(
                    labels.to(device='cpu').numpy()
                )
        accuracy = sklearn.metrics.accuracy_score(y_true=targets, y_pred=preds)
        precision = sklearn.metrics.precision_score(y_true=targets, y_pred=preds, average='weighted', zero_division=0)
        recall = sklearn.metrics.recall_score(y_true=targets, y_pred=preds, average='weighted')
        f1 = sklearn.metrics.f1_score(y_true=targets, y_pred=preds, average='weighted')
        metrics = {
            'accuracy'  : accuracy,
            'precision' : precision,
            'recall'    : recall,
            'f1'        : f1 
        }
        return metrics