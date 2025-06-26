import os
import logging
from gensim.corpora import WikiCorpus
import multiprocessing # 멀티프로세싱 모듈 임포트

# 로그 메시지를 보기 위한 설정 (이것은 __main__ 밖에서도 괜찮습니다)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 이 부분이 중요합니다: Windows에서 멀티프로세싱 시작 방식 설정
# 이 코드는 스크립트가 임포트될 때가 아닌, 직접 실행될 때만 작동합니다.
# 안전을 위해 set_start_method는 if __name__ == '__main__': 블록 내에서 가장 먼저 호출하는 것이 좋습니다.
# 그러나 gensim의 경우, 이 오류를 피하기 위해 때로는 freeze_support()와 함께 사용되기도 합니다.
# 여기서는 가장 일반적인 권장사항인 __main__ 내부에 전체 로직을 넣는 방식을 따릅니다.

if __name__ == '__main__':
    # freeze_support()는 실행 파일을 만들 때 (예: PyInstaller) 필요하지만,
    # 일반 스크립트 실행에서도 안전하게 추가하여 이러한 bootstrapping 오류를 방지할 수 있습니다.
    multiprocessing.freeze_support() 
    
    # 위키백과 덤프 파일 경로
    input_dump_path = 'datasets/kowiki-latest-pages-articles.xml.bz2'
    # 텍스트를 저장할 디렉토리 (자동 생성됩니다)
    output_dir = 'datasets/extracted_wiki_text_gensim'

    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 텍스트 파일을 분할해서 저장하기 위한 변수
    file_idx = 0
    doc_count = 0
    max_docs_per_file = 100000 # 한 파일에 저장할 최대 문서 수 (적절히 조절 가능)
    current_output_file = None
    output_file_path = ""

    print(f"Gensim을 사용하여 '{input_dump_path}' 에서 텍스트 추출 시작...")

    try:
        # WikiCorpus 인스턴스 생성
        wiki = WikiCorpus(input_dump_path, dictionary={},
                          metadata=False,
                          article_min_tokens=10
                         )

        # get_texts()는 문서를 하나씩 반환하는 이터레이터
        for i, text_list in enumerate(wiki.get_texts()):
            # gensim은 단어 리스트를 반환하므로, 다시 문자열로 합쳐줍니다.
            clean_text = ' '.join(text_list)

            if clean_text: # 내용이 있는 경우에만 저장
                if doc_count % max_docs_per_file == 0:
                    if current_output_file:
                        current_output_file.close() # 이전 파일 닫기
                    
                    # 새로운 출력 파일 경로 설정
                    output_file_path = os.path.join(output_dir, f'wiki_text_{file_idx:05d}.txt')
                    current_output_file = open(output_file_path, 'w', encoding='utf-8')
                    file_idx += 1
                    print(f"새로운 출력 파일 시작: {output_file_path}")

                current_output_file.write(clean_text + '\n')
                doc_count += 1

            if i % 100000 == 0 and i > 0:
                logging.info(f"{i}개 문서 처리 완료...")

    except Exception as e:
        # 오류 메시지를 정확히 출력하도록 수정
        logging.error(f"오류 발생: {e}", exc_info=True)
    finally:
        if current_output_file:
            current_output_file.close() # 마지막 파일 닫기

    print(f"\n--- Gensim 텍스트 추출 완료 ---")
    print(f"총 {doc_count}개의 문서가 {output_dir} 디렉토리에 저장되었습니다.")