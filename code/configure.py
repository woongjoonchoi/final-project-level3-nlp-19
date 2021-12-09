from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer , Seq2SeqTrainingArguments , EncoderDecoderModel ,AutoModelForSeq2SeqLM

import torch

class CustomSeq2Seq(EncoderDecoderModel) :

    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        if decoder_start_token_id is None:
            raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
        shifted_input_ids[:, 0] = decoder_start_token_id
 
        if pad_token_id is None:
            raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

def configure_model(model_args , training_args ,data_args):
    if model_args.run_extraction:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name is not None
            else model_args.model_name_or_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name is not None
            else model_args.model_name_or_path,
            use_fast=True,
        )

        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    elif model_args.run_generation:
        model_name='wjc123/double_klue2'
        tokenizer_name = 'klue/bert-base'
        training_args = Seq2SeqTrainingArguments(
            do_train=True, 
            do_eval=True, 
            predict_with_generate=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            logging_dir='./logs',
            logging_steps=100,
            output_dir = training_args.output_dir , 
            evaluation_strategy = 'steps' , 
            eval_steps = 300,
            report_to="wandb",
            fp16 = True,
            save_total_limit=2
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        model = CustomSeq2Seq.from_pretrained(model_name)

        
        
        model.config.decoder_start_token_id = tokenizer.mask_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size  if not data_args.run_seq2seq   else model.config.vocab_size
        

    return model , tokenizer 