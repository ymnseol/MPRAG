import argparse
from pprint import pprint


def get_args(parser):
    parser.add_argument("--output_dir", dest="output_dir", action="store", type=str)

    parser.add_argument("--pair_data_path", dest="pair_data_path", action="store", type=str)
    parser.add_argument("--question_data_path", dest="question_data_path", action="store", type=str)
    parser.add_argument("--passage_data_path", dest="passage_data_path", action="store", type=str)

    parser.add_argument("--pair_id_column_name", dest="pair_id_column_name", action="store", default=None, type=str)
    parser.add_argument("--question_id_column_name", dest="question_id_column_name", action="store", default=None, type=str)
    parser.add_argument("--passage_id_column_name", dest="passage_id_column_name", action="store", default=None, type=str)

    args = parser.parse_args()
    return args
