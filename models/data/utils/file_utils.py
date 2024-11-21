import os
import zipfile

def unzip_files(zip_dir="../02.라벨링데이터"):
    """

    :param zip_dir: root dir
    :return: json
    """

    for file_name in os.listdir(zip_dir):
        file_path = os.path.join(zip_dir, file_name)
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                extract_dir = os.path.join(zip_dir, os.path.splitext(file_name)[0])
                os.makedirs(extract_dir, exist_ok=True)
                zip_ref.extractall(extract_dir)
                print(f"{file_name}의 압축이 {extract_dir}에 풀렸습니다.")