import os
import logging
from typing import Dict, NoReturn, Tuple

from configure import *
from preprocess import *
from mrc_metrics import *
from datasets import (
    load_metric,
    Value,
    Features,
    Dataset,
    DatasetDict,
    load_dataset,
    
)

from transformers import (
    Seq2SeqTrainer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    TrainerCallback,
    default_data_collator
    # PrinterCallback,
)


from utils_qa import postprocess_qa_predictions, check_no_error
from trainer_qa import QuestionAnsweringTrainer
from postprocessing import post_processing_function, compute_metrics

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

logger = logging.getLogger(__name__)
# class MyCallback(TrainerCallback):
#     "A callback that prints a message at the beginning of training"

#     def on_evaulate(self, args, state, control, **kwargs):
#         breakpoint()
#         print("Starting training")
class PrinterCallback(TrainerCallback):
    """
    A bare :class:`~transformers.TrainerCallback` that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        # breakpoint()
        print(logs)
        print(state.log_history)
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
# run_extraction_mrc, run_mrc 합침
def run_combine_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # print(training_args.do_train)
    # print(training_args.do_eval)
    # print(training_args.do_predict)

    # if training_args.do_predict == True:
    #     training_args.do_train = False
    # elif training_args.do_train == True:
    #     training_args.do_predict = False


    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:

        print("anjdieffffffffffffffffffeff")
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        
        # prepare_train_features = preprocess_gen(tokenizer)
        # dataset에서 train feature를 생성합니다.
        prepare_train_features = preprocess_extract_train(tokenizer, data_args, column_names, max_seq_length)
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Validation preprocessing / 전처리를 진행합니다.
    if training_args.do_eval or training_args.do_predict:
        eval_dataset = datasets["validation"]

        prepare_valid_features = preprocess_extract_valid(tokenizer, data_args, column_names, max_seq_length)
        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_valid_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )
    # data_collator = default_data_collator

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer( 
        model=model,
        args=training_args,
        # train_dataset=train_dataset if training_args.do_train else None,
        train_dataset=eval_dataset,
        # output_dir= o
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Evaluate ***")
    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )
        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )
    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


def run_generation_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # Training Arguments를 Seq2Seq로 바꿔준다. 이 때, predict_with_generate만 True로 변경해주고 나머지는 Default 사용
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        output_dir = training_args.output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        do_train = training_args.do_train,
        do_eval = training_args.do_eval,
        eval_steps = 100,
        evaluation_strategy = 'steps',
        num_train_epochs = 12
    )
    # print(training_args)
    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    
    print(data_args.pad_to_max_length)
    print(tokenizer)

    max_seq_length = data_args.max_seq_length
    max_length = data_args.max_answer_length

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")

        train_dataset = datasets["train"]

        preprocess_function = preprocess_gen(tokenizer,model.__class__.__name__)
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model = model,
        # pad_to_multiple_of=8 if training_args.fp16 else None
    )

    compute_metrics = gen_metrics(tokenizer,datasets["validation"])
    print(train_dataset)


    trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=eval_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,   
        )
    trainer.add_callback(PrinterCallback)
    if training_args.do_train:
        
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)