import json
import os
import pandas as pd
from datetime import datetime

def process_json_files(root_dir, file_paths, output_path):
    # 빈 데이터프레임 딕셔너리 초기화
    data_frames = {
        "METEOR score": pd.DataFrame(),
        "BertScore": pd.DataFrame(),
        "ROUGE-1": pd.DataFrame(),
        "ROUGE-2": pd.DataFrame(),
        "ROUGE-L": pd.DataFrame(),
        "LLM_Judge Score": pd.DataFrame()
    }

    for file_path in file_paths:
        with open(os.path.join(root_dir, file_path), 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 알고리즘 이름을 파일 이름에서 추출 (확장자 제거)
        algorithm_name = file_path.split('/')[-1].replace('.txt', '')

        # m_scores -> METEOR score
        m_scores = data['avg_scores']['m_scores']
        m_df = pd.DataFrame(m_scores, index=[algorithm_name])
        m_df.columns = [col.split('/')[-2].upper() for col in m_df.columns]
        data_frames["METEOR score"] = pd.concat([data_frames["METEOR score"], m_df])

        # b_scores -> BertScore (f1 값만 사용)
        b_scores = {k: v['f1'] for k, v in data['avg_scores']['b_scores'].items()}
        b_df = pd.DataFrame(b_scores, index=[algorithm_name])
        b_df.columns = [col.split('/')[-2].upper() for col in b_df.columns]
        data_frames["BertScore"] = pd.concat([data_frames["BertScore"], b_df])

        # r_scores -> ROUGE-1, ROUGE-2, ROUGE-L
        r_scores = data['avg_scores']['r_scores']
        for rouge_type in ["rouge1", "rouge2", "rougeL"]:
            r_df = pd.DataFrame({k: v[rouge_type] for k, v in r_scores.items()}, index=[algorithm_name])
            r_df.columns = [col.split('/')[-2].upper() for col in r_df.columns]
            score_type = rouge_type.upper().replace("ROUGE", "ROUGE-")
            data_frames[score_type] = pd.concat([data_frames[score_type], r_df])

        # llm_scores -> LLM_Judge Score
        llm_scores = data['avg_scores']['llm_scores']
        llm_df = pd.DataFrame(llm_scores, index=[algorithm_name])
        llm_df.columns = [col.split('/')[-2].upper() for col in llm_df.columns]
        data_frames["LLM_Judge Score"] = pd.concat([data_frames["LLM_Judge Score"], llm_df])

    # 엑셀 파일로 저장
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in data_frames.items():
            df.to_excel(writer, sheet_name=sheet_name)


def main():
    # 파일 경로 리스트 예시
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"comp_results_{current_time}.txt"
    output_filename = f'/mnt/data_disk/work/Space-awareVLM/eval_results/{output_filename}.xlsx'

    root_dir = '/mnt/data_disk/work/Space-awareVLM/eval_results/'
    file_paths = [
        'llava-v1.6-mistral-7b-hf/llava-v1.6-mistral-7b-hf_20240904_001856.txt',
        'llava-v1.6-vicuna-7b-hf/llava-v1.6-vicuna-7b-hf_20240903_214859.txt',
        'llama3-llava-next-8b-hf/llama3-llava-next-8b-hf_20240904_001730.txt',
        'llava-v1.6-vicuna-13b-hf/llava-v1.6-vicuna-13b-hf_20240904_005155.txt',
        'gpt-4o-mini-2024-07-18/gpt-4o-mini-2024-07-18_20240904_052623.txt',
        'llava-onevision-qwen2-0.5b-si/llava-onevision-qwen2-0.5b-si_20240904_075029.txt',
        'llava-onevision-qwen2-7b-si/llava-onevision-qwen2-7b-si_20240904_083049.txt',
        'ft_llava-onevision-qwen2-0.5b-si_bs2_gas4_20240902_11_07_20/ft_llava-onevision-qwen2-0.5b-si_bs2_gas4_20240902_11_07_20_20240902_221143.txt',
        'ft_llava-onevision-qwen2-7b-si_bs1_gas8_20240902_12_25_58/ft_llava-onevision-qwen2-7b-si_bs1_gas8_20240902_12_25_58_20240903_202201.txt',
    ]

    # JSON 파일 처리 및 엑셀 파일 생성
    process_json_files(root_dir, file_paths, output_filename)


if __name__ == "__main__":
    main()
