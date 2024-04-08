from transformers import AutoConfig, AutoModel
import torch

class FeatureBERT(torch.nn.Module):
    """ Add default C-Test features on top without the BERT based features (59 in total).
    """

    def __init__(self, model_name):
        # num_extra_dims corresponds to the number of extra dimensions of numerical/categorical data

        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        num_hidden_size = self.transformer.config.hidden_size 
        # We set num labels to 1 since we do regression
        # We have 59 extra features
        self.classifier = torch.nn.Linear(num_hidden_size + 59, 1) 

    def forward(self, input_ids, extra_data, attention_mask=None):
        """
        extra_data should be of shape [batch_size, dim] 
        where dim is the number of additional numerical/categorical dimensions
        """

        hidden_states = self.transformer(input_ids=input_ids, attention_mask=attention_mask) # [batch size, sequence length, hidden size]
        cls_embeds = hidden_states.last_hidden_state[:, 0, :] # [batch size, hidden size]
        concat = torch.cat((cls_embeds, extra_data), dim=-1) # [batch size, hidden size+num extra dims]
        output = self.classifier(concat) # [batch size, num labels]

        return output

