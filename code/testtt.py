from transformers import AutoConfig,AutoTokenizer  , AutoModel,AutoModelForQuestionAnswering, AutoTokenizer , Seq2SeqTrainingArguments , EncoderDecoderModel , AutoModelForSeq2SeqLM
import datasets
model_name='klue/bert-base'
model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

# model.save_pretrained('my-model')

# model = AutoModel.from_pretrained('my-model')
print(datasets.__version__)

# model.save_pretrained('wjcmodelc')
# print(model)

# model = AutoModelForSeq2SeqLM.from_pretrained('wjcmodel')
# model = AutoModel.from_pretrained(model_name)
# print(model)
# model.push_to_hub('dobule_klue')

model = AutoModelForSeq2SeqLM.from_pretrained("wjc123/dobule_klue")

print(hasattr(model, "prepare_decoder_input_ids_from_labels"))
# print(model._preapre_decoder_input_ids_for_generation)
print(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)


with tokenizer.as_target_tokenizer() :
    print(tokenizer(["[MASK]안녕"] , add_special_tokens = False))
# Push the model to your namespace with the name "my-finetuned-bert" and have a local clone in the
# _my-finetuned-bert_ folder.
# model.push_to_hub("my-finetuned-bert")

# print(list(model.named_parameters()))
# for name, moudle  in model.named_modules():
#     print(name)
print(model.__class__.__name__)