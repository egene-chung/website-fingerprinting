import os
from data.preprocess import check_and_preprocess_data


if __name__ == "__main__":
    # 파일 경로
    ow_path = 'data/csv/openworld_data.csv'
    cw_path = 'data/csv/closedworld_data.csv'

    check_and_preprocess_data(ow_path, cw_path)