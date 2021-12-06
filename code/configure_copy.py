from transformers import (
    AutoConfig, 
    AutoModelForQuestionAnswering, 
    AutoTokenizer,
    AutoModel,
    Seq2SeqTrainingArguments, 
    EncoderDecoderModel
)

from CustomKoBigbird import KoBigBird_with_Decoder

def cofngiure_model(model_args , training_args ,data_args):
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
        model_name = "monologg/kobigbird-bert-base"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        KoBigBird = AutoModel.from_pretrained(model_name)

        config = KoBigBird.config
        config.encoder_layers = 6
        config.decoder_layers = 6
        
        # huggingface의 BigBird 모델은 Encoder, Decoder class가 따로 있지 않고, 모델 생성시의 config값에 따라 인코더 모델, 디코더 모델로 생성이 됨.
        # 기존 KoBigBird 모델(Encoder)에 추가로 BigBird모델(Decoder)을 생성 후 연결 시켜서 Encoder-Decoder모델을 만듬.
        config.is_decoder = True # Bigbird layer를 Decoder로 생성
        config.add_cross_attention = True # Decodef에서 필요한 cross_attentio 생성
        config.attention_type='original_full' # 아웃풋은 길이가 길지 않으므로 original_full 어텐션 사용

        model = KoBigBird_with_Decoder(KoBigBird, config)

    return model, tokenizer 