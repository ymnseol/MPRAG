import csv
import pandas as pd
def load_passages(path):
    passages = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for k, row in enumerate(reader):
            if not row[0] == 'id':
                try:
                    passages.append((row[0], row[1], row[2]))
                except:
                    print('Error!')
    return passages


def parse_trec_line(line):
    # 각 라인을 적절히 파싱하여 필요한 정보를 추출하는 함수
    # 예시: 라인을 공백으로 분할하여 필요한 정보를 추출
    parts = line.strip().split()
    doc_id = parts[2]  # 예시: 세 번째 열에 문서 ID가 있는 경우
    return doc_id



def remove_comments(strings):
    cleaned_strings = []
    for string in strings:
        # '#' 문자가 있는 부분을 찾습니다.
        hash_index = string.find('#')
        if hash_index != -1:
            # '#' 이전의 문자를 잘라냅니다.
            cleaned_string = string[:hash_index]
            # 문자열 끝의 공백을 제거합니다.
            cleaned_string = cleaned_string.rstrip()
            cleaned_strings.append(cleaned_string)
        else:
            # '#' 문자가 없는 경우 그대로 추가합니다.
            cleaned_strings.append(string)
    return cleaned_strings

def load_list(path):
    list1=[]
    cur=[]
    with open(path, 'r') as f:
        for line in f:
            doc_id= parse_trec_line(line)
            cur.append(doc_id)
            if (len(cur)==1000):
                list1.append(remove_comments(cur))
                cur=[]
    return list1


def load_data(path):
    df=pd.read_csv(path,sep='\t',header=None)
    df.columns=['answers','question']
    return df




        