from datasets import load_metric
from transformers import EvalPrediction

from utils_qa import postprocess_qa_predictions


# Post-processing:
def post_processing_function(examples, features, predictions, training_args):
    # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
    answer_column_name = "answers" if "answers" in examples.column_names else examples.column_names[2]

    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=30, # data_args.max_answer_length,
        output_dir=training_args.output_dir,
    )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    if training_args.do_predict:
        return formatted_predictions

    elif training_args.do_eval:
        references = [
            {"id": ex["id"], "answers": ex[answer_column_name]}
            for ex in examples
        ]
        return EvalPrediction(
            predictions=formatted_predictions, label_ids=references
        )

    
def compute_metrics(p: EvalPrediction):
    metric = load_metric("squad")

    return metric.compute(predictions=p.predictions, references=p.label_ids)

"""
def postprocess_text(preds, labels, tokenizer):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
        
    preds = ["\n".join(tokenizer.tokenize(pred)) for pred in preds]
    labels = ["\n".join(tokenizer.tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    metric = load_metric("squad")
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    max_val_samples = 16

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in         
                                 enumerate(datasets["validation"].select(range(max_val_samples)))]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"].select(range(max_val_samples))]

    result = metric.compute(predictions=formatted_predictions, references=references)

    return result
"""