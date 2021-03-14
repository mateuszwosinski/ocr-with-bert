import torch
from torch import nn
from transformers import BertModel


class TypoDetectorBERT(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        self.dropout = nn.Dropout(0.2)
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5
        self.embedding_dim = 768
        self.out_size = 32
        self.num_labels = 2
        
        self.conv_1 = nn.Conv1d(self.embedding_dim, self.out_size, self.kernel_1, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_1.weight, mode='fan_out')
        self.pool_1 = nn.MaxPool1d(kernel_size=self.kernel_1, stride=1)
        
        self.conv_2 = nn.Conv1d(self.embedding_dim, self.out_size, self.kernel_2, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_2.weight, mode='fan_out')
        self.pool_2 = nn.MaxPool1d(kernel_size=self.kernel_2, stride=1, padding=1)
        
        self.conv_3 = nn.Conv1d(self.embedding_dim, self.out_size, self.kernel_3, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv_3.weight, mode='fan_out')
        self.pool_3 = nn.MaxPool1d(kernel_size=self.kernel_3, stride=1, padding=1)
        
        self.conv_4 = nn.Conv1d(self.embedding_dim, self.out_size, self.kernel_4, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv_4.weight, mode='fan_out')
        self.pool_4 = nn.MaxPool1d(kernel_size=self.kernel_4, stride=1, padding=2)
        
        self.classifier = nn.Linear(self.out_size * 4, self.num_labels)  
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out')
        
    def forward(self, input, attention_mask):
        out = self.bert(input, attention_mask)
        seq_out = out[0]
        
        conv_inp = seq_out.permute(0, 2, 1)
        
        conv_out_1 = self.pool_1(torch.relu(self.conv_1(conv_inp)))
        conv_out_2 = self.pool_2(torch.relu(self.conv_2(conv_inp)))
        conv_out_3 = self.pool_3(torch.relu(self.conv_3(conv_inp)))
        conv_out_4 = self.pool_4(torch.relu(self.conv_4(conv_inp)))
        
        conv_out = torch.cat([conv_out_1, conv_out_2, conv_out_3, conv_out_4],
                             axis = 1)
        conv_out = self.dropout(conv_out.permute(0, 2, 1))
        
        logits = self.classifier(conv_out)
        return logits

        