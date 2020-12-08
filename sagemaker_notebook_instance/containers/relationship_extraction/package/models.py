import torch
from transformers import BertModel
import pytorch_lightning as pl


class RelationshipEncoderModule(torch.nn.Module):
    def __init__(
        self,
        vocab_length,
        num_classes,
        entity_one_start_token_id,
        entity_two_start_token_id
    ):
        super(RelationshipEncoderModule, self).__init__()
        self.entity_one_start_token_id = entity_one_start_token_id
        self.entity_two_start_token_id = entity_two_start_token_id
        self.text_encoder = BertModel.from_pretrained(
            pretrained_model_name_or_path='bert-base-uncased'
        )
        self.text_encoder.resize_token_embeddings(vocab_length)
        self.layer_norm = torch.nn.LayerNorm(self.text_encoder.config.hidden_size * 2)
        self.linear = torch.nn.Linear(self.text_encoder.config.hidden_size * 2, num_classes)

    def forward(
        self,
        token_ids,
        attention_mask
    ):
        output = self.text_encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden_state = output['last_hidden_state']
        entity_one_mask = (token_ids == self.entity_one_start_token_id).int()
        entity_one_embedding = (entity_one_mask.unsqueeze(dim=-1) * last_hidden_state).sum(dim=1)
        entity_two_mask = (token_ids == self.entity_two_start_token_id).int()
        entity_two_embedding = (entity_two_mask.unsqueeze(dim=-1) * last_hidden_state).sum(dim=1)
        relationship_embedding = torch.cat([entity_one_embedding, entity_two_embedding], dim=1)
        relationship_embedding_norm = self.layer_norm(relationship_embedding)
        logits = self.linear(relationship_embedding_norm)
        return logits
    
    
class RelationshipEncoderLightningModule(pl.LightningModule):
    def __init__(self, tokenizer, label_encoder, learning_rate=0.0007):
        super().__init__()
        self.model = RelationshipEncoderModule(
            vocab_length=len(tokenizer),
            num_classes=len(label_encoder),
            entity_one_start_token_id=tokenizer.entity_one_start_token_id,
            entity_two_start_token_id=tokenizer.entity_two_start_token_id
        )
        self.learning_rate = learning_rate
        self.train_acc = pl.metrics.Accuracy()
        self.train_f1 = pl.metrics.Fbeta(num_classes=len(label_encoder))
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_f1 = pl.metrics.Fbeta(num_classes=len(label_encoder))

    def forward(self, token_ids, attention_mask):
        output = self.model(token_ids, attention_mask)
        return output

    def training_step(self, batch, batch_idx):
        token_ids = batch['token_ids']
        attention_mask = batch['attention_mask']
        label_id = batch['label_id']
        output = self.model(token_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(output, label_id)
        self.log('train_loss', loss)
        self.train_acc(output, label_id)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.train_f1(output, label_id)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        token_ids = batch['token_ids']
        attention_mask = batch['attention_mask']
        label_id = batch['label_id']
        output = self.model(token_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(output, label_id)
        self.log('valid_loss', loss)
        self.valid_acc(output, label_id)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
        self.valid_f1(output, label_id)
        self.log('valid_f1', self.valid_f1, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)