import argparse
from pprint import pprint


def get_args(parser):
    ### Data ###
    parser.add_argument("--train_data_path", dest="train_data_path", action="store", type=str)
    parser.add_argument("--eval_data_path", dest="eval_data_path", action="store", type=str)
    parser.add_argument("--passage_data_path", dest="passage_data_path", action="store", type=str)

    ### Models ###
    parser.add_argument("--pretrained_model_name_or_path", dest="pretrained_model_name_or_path", action="store", default="google-bert/bert-base-uncased", type=str)
    parser.add_argument("--label_smoothing", dest="label_smoothing", action="store", default=0.0, type=float)

    ### Training Arguments ###
    parser.add_argument("--output_dir", dest="output_dir", action="store", default="outputs", type=str)
    parser.add_argument("--logging_dir", dest="logging_dir", action="store", default="logs", type=str)
    parser.add_argument("--save_total_limit", dest="save_total_limit", action="store", default=5, type=int)
    parser.add_argument("--report_to", dest="report_to", action="store", default="all", choices=["no", "all", "azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", "wandb"], type=str)
    parser.add_argument("--run_name", dest="run_name", action="store", default=None, type=str)

    parser.add_argument("--eval_strategy", dest="eval_strategy", action="store", default="steps", choices=["no", "steps", "epoch"], type=str)
    parser.add_argument("--eval_steps", dest="eval_steps", action="store", default=500, type=float)
    parser.add_argument("--logging_strategy", dest="logging_strategy", action="store", default="steps", choices=["no", "steps", "epoch"], type=str)
    parser.add_argument("--logging_steps", dest="logging_steps", action="store", default=500, type=float)
    parser.add_argument("--save_strategy", dest="save_strategy", action="store", default="steps", choices=["no", "steps", "epoch"], type=str)
    parser.add_argument("--save_steps", dest="save_steps", action="store", default=500, type=float)

    parser.add_argument("--per_device_train_batch_size", dest="per_device_train_batch_size", action="store", default=8, type=int)
    parser.add_argument("--per_device_eval_batch_size", dest="per_device_eval_batch_size", action="store", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", action="store", default=1, type=int)

    parser.add_argument("--num_train_epochs", dest="num_train_epochs", action="store", default=3, type=int)
    parser.add_argument("--max_steps", dest="max_steps", action="store", default=-1, type=int)

    parser.add_argument("--learning_rate", dest="learning_rate", action="store", default=5e-05, type=float)
    parser.add_argument("--weight_decay", dest="weight_decay", action="store", default=0.0, type=float)
    parser.add_argument("--lr_scheduler_type", dest="lr_scheduler_type", action="store", default="linear", type=str)
    parser.add_argument("--warmup_ratio", dest="warmup_ratio", action="store", default=0.0, type=float)
    parser.add_argument("--warmup_steps", dest="warmup_steps", action="store", default=0, type=int)

    parser.add_argument("--dataloader_pin_memory", dest="dataloader_pin_memory", action="store", default=True, type=bool)
    parser.add_argument("--dataloader_persistent_workers", dest="dataloader_persistent_workers", action="store", default=False, type=bool)
    parser.add_argument("--dataloader_num_workers", dest="dataloader_num_workers", action="store", default=0, type=int)
    parser.add_argument("--dataloader_drop_last", dest="dataloader_drop_last", action="store", default=False, type=bool)

    ### Weights & Biases ###
    parser.add_argument("--wandb_project", dest="wandb_project", action="store", type=str)
    parser.add_argument("--wandb_entity", dest="wandb_entity", action="store", type=str)

    args = parser.parse_args()
    return args
