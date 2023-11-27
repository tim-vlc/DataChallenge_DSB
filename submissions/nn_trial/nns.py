import torch.nn as nn
import torch

class NN(nn.Module):
    def __init__(self, input_size, output_size, dense1_output, dense2_output, dense3_output, dense4_output, dropratio):
        super(NN, self).__init__()
        
        self.dense1 = nn.Linear(input_size, dense1_output)
        self.sigmoid1 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(dropratio)
        
        self.dense2 = nn.Linear(dense1_output, dense2_output)
        self.sigmoid2 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(dropratio)
        
        self.dense3 = nn.Linear(dense2_output, dense3_output)
        self.sigmoid3 = nn.Sigmoid()
        
        self.dense4 = nn.Linear(dense3_output, dense4_output)
        self.sigmoid4 = nn.Sigmoid()
        
        self.dense5 = nn.Linear(dense4_output, output_size)
        self.sigmoid5 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dense1(x)
        x = self.sigmoid1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.sigmoid2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.sigmoid3(x)
        x = self.dense4(x)
        x = self.sigmoid4(x)
        x = self.dense5(x)
        x = self.sigmoid5(x)
        return x
    
    def train_nn(model, X_train, y_train, ep, batch, optimizer, criterion):
        for epoch in range(ep):
            model.train()
            running_loss = 0.0
            for i in range(0, len(X_train), batch):
                batch_X = X_train[i:i+batch].clone().detach().float()
                batch_y = y_train[i:i+batch].clone().detach().float()
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = torch.sqrt(criterion(outputs, batch_y))
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Print the average loss for the epoch
            avg_loss = running_loss / (len(X_train) / batch)
            print(f'Epoch [{epoch+1}/{ep}], Loss: {avg_loss:.4f}')
    
    def test_nn(model, X_test, y_test, batch, criterion):
        with torch.no_grad():
            loss = 0
            for i in range(0, len(X_test), batch):
                batch_X = X_test[i:i+batch].clone().detach().float()
                batch_y = y_test[i:i+batch].clone()
                outputs = (model(batch_X)).detach().cpu().numpy()

                loss += torch.sqrt(criterion(outputs, batch_y))

                torch.cuda.empty_cache()
            return loss / len(X_test)
