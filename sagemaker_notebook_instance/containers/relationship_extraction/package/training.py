import argparse
import os
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader

from package.data.tokenizers import RelationshipTokenizer
from package.data.label_encoders import LabelEncoder
from package.data.semeval import label_set
from package.data.dataset import RelationStatementDataset
from package.models import RelationshipEncoderLightningModule


def parse_args(sys_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0007
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=os.environ.get("SM_NUM_GPUS", 0)
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
    )
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST")
    )

    args, _ = parser.parse_known_args(sys_args)
    return args

    
def train_fn(args):
    print(args)
    
    # load tokenizer
    tokenizer = RelationshipTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-uncased',
        contains_entity_tokens=False
    )
    tokenizer.save(file_path=Path(args.model_dir, 'tokenizer.json'), pretty=True)
    
    # load data
    train_file_path = Path(args.train_data_dir, 'train.txt')
    test_file_path = Path(args.test_data_dir, 'test.txt')
    
    # construct label encoder
    labels = list(label_set(train_file_path))
    label_encoder = LabelEncoder.from_str_list(sorted(labels))
    print('Using the following label encoder mappings:\n\n', label_encoder)
    label_encoder.save(file_path=str(Path(args.model_dir, 'label_encoder.json')))
    
    # prepare datasets
    model_size = 512
    tokenizer.set_truncation(model_size)
    tokenizer.set_padding(model_size)
    train_dataset = RelationStatementDataset(
        file_path=train_file_path,
        tokenizer=tokenizer,
        label_encoder=label_encoder
    )
    test_dataset = RelationStatementDataset(
        file_path=test_file_path,
        tokenizer=tokenizer,
        label_encoder=label_encoder
    )

    batch_size = 16
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=4
    )
    
    # create model
    relationship_encoder = RelationshipEncoderLightningModule(
        tokenizer,
        label_encoder,
        learning_rate=float(args.learning_rate)
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        filepath=str(Path(args.model_dir, 'model'))
    )
    
    # train model
    trainer = Trainer(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        max_epochs=1,
        weights_summary='full',
        gpus=args.gpus,
        checkpoint_callback=checkpoint_callback,
        fast_dev_run=True
    )
    
    trainer.fit(relationship_encoder, train_dataloader, test_dataloader)
