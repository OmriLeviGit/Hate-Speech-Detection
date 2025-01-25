from transformers import Trainer, TrainingArguments

def train_model(model, tokenizer, train_dataset, val_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=14,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=35,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    trainer.train()

    # Save the final model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer