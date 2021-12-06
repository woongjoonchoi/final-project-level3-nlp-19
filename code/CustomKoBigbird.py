import copy

import torch
import torch.nn as nn
from transformers.models.big_bird.modeling_big_bird import BigBirdLayer, BigBirdModel


class KoBigBird_with_Decoder(nn.Module):
    def __init__(self, KoBigBird, config):
        super(KoBigBird_with_Decoder, self).__init__()

        self.config = config

        # 기존 KoBigBird 모델에서 Embedding층을 가져와서 Shared Embedding으로 사용.
        self.embdedding_block = copy.deepcopy(KoBigBird.embeddings)

        self.encoder = KoBigBird # 기존 KoBigBird 모델을 Encoder로 사용.

        self.encoder.embeddings = self.embdedding_block # Encoder의 Embedding 설정
        self.encoder.encoder.layer = self.encoder.encoder.layer[:self.config.encoder_layers] # Config값에 따라 layer 숫자 조정

        self.decoder = BigBirdModel(config) # Decoder로 사용할 BigBird모델 생성
        self.decoder.embeddings = self.embdedding_block # Decoder의 Embedding 설정
        self.decoder.encoder.layer = self.decoder.encoder.layer[:self.config.decoder_layers] # Config값에 따라 layer 숫자 조정

        # Decoder의 출력값 batch_size,d_model -> batch_size, n_vocab으로 바꿔줄 Linear
        self.tokens_logit_layer = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.config.vocab_size)))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # Encoder와 Decoder 모두 Bigbird 모델이지만 input값과 내부 구조가 다름.

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # bs, seq_len, d_model -> bs, seq_len, n_vocab으로 token별 logit 생성
        output = self.tokens_logit_layer(decoder_outputs[0]) + self.final_logits_bias

        return output