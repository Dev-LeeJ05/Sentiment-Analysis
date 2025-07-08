from tokenizers import BertWordPieceTokenizer
import os

def SaveDataFrameTextsTo_txt(dataframe, text_column_name, output_file_path):
    print(f"'{text_column_name}' 컬럼의 텍스트를 '{output_file_path}' 에 저장 중...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for index, row in dataframe.iterrows():
            text = row[text_column_name]
            if isinstance(text, str):
                f.write(text.strip() + '\n')
            else:
                print(f"경고: {index}번째 행의 '{text_column_name}' 컬럼이 문자열이 아닙니다: {text}. 이 행은 건너뜁니다.")
    print(f"텍스트 저장 완료: 총 {len(dataframe)}개 항목 중 유효한 텍스트 저장됨.")

def GenerateVocab(files,vocab_size,min_frequency,special_tokens,output_dir):
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )

    tokenizer.train(
        files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer.save_model(output_dir)

    print(f"새로운 vocab.txt 파일이 '{output_dir}/sentiment_wordpiece-vocab.txt' 에 저장되었습니다.")
    print(f"새로운 tokenizer.json 파일이 '{output_dir}/sentiment_wordpiece-tokenizer.json' 에 저장되었습니다.")