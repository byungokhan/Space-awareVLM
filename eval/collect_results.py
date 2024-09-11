import json
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


# Helper function to convert column names
def convert_column_name(col_name):
    if col_name.endswith('recommend'):
        return 'RECOMMEND'
    else:
        return col_name.split('/')[-2].upper()

def process_json_files(root_dir, file_paths, output_path):
    # 빈 데이터프레임 딕셔너리 초기화
    data_frames = {
        "METEOR score": pd.DataFrame(),
        "BertScore": pd.DataFrame(),
        "ROUGE-1": pd.DataFrame(),
        "ROUGE-2": pd.DataFrame(),
        "ROUGE-L": pd.DataFrame(),
        "LLM_Judge": pd.DataFrame(),
        "LLM_Judge_w_C": pd.DataFrame(),
        "NUM_WORDS": pd.DataFrame(),
    }

    for file_path in file_paths:
        with open(os.path.join(root_dir, file_path), 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 알고리즘 이름을 파일 이름에서 추출 (확장자 제거)
        algorithm_name = file_path.split('/')[-1].replace('.txt', '')

        # m_scores -> METEOR score
        m_scores = data['avg_scores']['m_scores']
        m_df = pd.DataFrame(m_scores, index=[algorithm_name])
        m_df.columns = [convert_column_name(col) for col in m_df.columns]
        data_frames["METEOR score"] = pd.concat([data_frames["METEOR score"], m_df])

        # b_scores -> BertScore (f1 값만 사용)
        b_scores = {k: v['f1'] for k, v in data['avg_scores']['b_scores'].items()}
        b_df = pd.DataFrame(b_scores, index=[algorithm_name])
        b_df.columns = [convert_column_name(col) for col in b_df.columns]
        data_frames["BertScore"] = pd.concat([data_frames["BertScore"], b_df])

        # r_scores -> ROUGE-1, ROUGE-2, ROUGE-L
        r_scores = data['avg_scores']['r_scores']
        for rouge_type in ["rouge1", "rouge2", "rougeL"]:
            r_df = pd.DataFrame({k: v[rouge_type] for k, v in r_scores.items()}, index=[algorithm_name])
            r_df.columns = [convert_column_name(col) for col in r_df.columns]
            score_type = rouge_type.upper().replace("ROUGE", "ROUGE-")
            data_frames[score_type] = pd.concat([data_frames[score_type], r_df])

        # llm_scores -> LLM_Judge Score
        llm_scores = data['avg_scores']['llm_scores']
        llm_df = pd.DataFrame(llm_scores, index=[algorithm_name])
        llm_df.columns = [convert_column_name(col) for col in llm_df.columns]
        data_frames["LLM_Judge"] = pd.concat([data_frames["LLM_Judge"], llm_df])

        # llm_scores_c -> LLM_Judge Score with conciseness
        llm_scores_c = data['avg_scores']['llm_scores_c']
        llm_df = pd.DataFrame(llm_scores_c, index=[algorithm_name])
        llm_df.columns = [convert_column_name(col) for col in llm_df.columns]
        data_frames["LLM_Judge_w_C"] = pd.concat([data_frames["LLM_Judge_w_C"], llm_df])

        # avg # words
        avg_num_words = data['avg_scores']['num_words']
        llm_df = pd.DataFrame(avg_num_words, index=[algorithm_name])
        llm_df.columns = [convert_column_name(col) for col in llm_df.columns]
        data_frames["NUM_WORDS"] = pd.concat([data_frames["NUM_WORDS"], llm_df])

    # 엑셀 파일로 저장
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in data_frames.items():
            df.to_excel(writer, sheet_name=sheet_name)

    return data_frames


# Updated function to handle a single selected description
def plot_metrics_bar_chart(data_frames, algorithms, selected_metrics, selected_description):
    # Ensure only valid selected metrics are used
    metrics = [metric for metric in selected_metrics if metric in data_frames.keys()]
    algorithms = [algo.split('/')[-1].replace('.txt', '') for algo in algorithms]

    # Prepare data for plotting
    algorithm_data = {alg: [] for alg in algorithms}

    # Loop through selected metrics and gather data for the selected description
    for metric in metrics:
        df = data_frames[metric].loc[algorithms].copy()  # Select only chosen algorithms

        # Filter for the selected description (only one description at a time)
        if selected_description in df.columns:
            df = df[[selected_description]]
        else:
            raise ValueError(f"Description '{selected_description}' not found in metric '{metric}'.")

        # Apply specific scaling to LLM_Judge and LLM_Judge_w_C (divide by 10)
        if metric in ["LLM_Judge", "LLM_Judge_w_C"]:
            df = df / 10  # Scale by 10 for these metrics

        # Collect the data for each algorithm
        for alg in algorithms:
            algorithm_data[alg].append(df.loc[alg].values[0])  # Get the score for the selected description

    # Set up the plot
    bar_width = 0.15  # Width of each bar
    index = np.arange(len(metrics))  # The x locations for the groups

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for each algorithm
    for i, alg in enumerate(algorithms):
        ax.bar(index + i * bar_width, algorithm_data[alg], bar_width, label=alg)

    # Add labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title(f'Comparison of Algorithms Across Metrics for "{selected_description}"')
    ax.set_xticks(index + bar_width / 2 * len(algorithms))
    ax.set_xticklabels(metrics)
    ax.legend()

    # Rotate x-axis labels for clarity
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def main():
    # 파일 경로 리스트 예시
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"comp_results_{current_time}.txt"
    output_filename = f'/mnt/data_disk/work/Space-awareVLM/eval_results/{output_filename}.xlsx'

    root_dir = '/mnt/data_disk/work/Space-awareVLM/eval_results/'
    file_paths = [
        'llava-v1.6-mistral-7b-hf/llava-v1.6-mistral-7b-hf_20240905_084551.txt',
        'llava-v1.6-vicuna-7b-hf/llava-v1.6-vicuna-7b-hf_20240905_194140.txt',
        'llama3-llava-next-8b-hf/llama3-llava-next-8b-hf_20240906_003633.txt',
        'llava-v1.6-vicuna-13b-hf/llava-v1.6-vicuna-13b-hf_20240906_023444.txt',
        'llava-v1.6-34b-hf/llava-v1.6-34b-hf_20240909_071951.txt',
        'llava-next-72b-hf/llava-next-72b-hf_20240905_144521.txt',
        'gpt-4o-mini-2024-07-18/gpt-4o-mini-2024-07-18_20240911_082843.txt',
        'gpt-4o-2024-08-06/gpt-4o-2024-08-06_20240906_070053.txt',
        'llava-onevision-qwen2-0.5b-si/llava-onevision-qwen2-0.5b-si_20240906_015732.txt',
        'llava-onevision-qwen2-7b-si/llava-onevision-qwen2-7b-si_20240906_024924.txt',
        'ft_llava-onevision-qwen2-0.5b-si_ng5_bs2_gas4_epoch1_20240908_22_28_05/ft_llava-onevision-qwen2-0.5b-si_ng5_bs2_gas4_epoch1_20240908_22_28_05_20240910_211042.txt',
        'ft_llava-onevision-qwen2-0.5b-si_ng5_bs2_gas4_epoch5_20240909_00_29_54/ft_llava-onevision-qwen2-0.5b-si_ng5_bs2_gas4_epoch5_20240909_00_29_54_20240910_210738.txt',
 #       'ft_llava-onevision-qwen2-0.5b-si_ng8_bs2_gas8_epoch1_20240911_02_41_43/',
        'ft_llava-onevision-qwen2-7b-si_ng8_bs1_gas8_epoch1_20240909_09_42_47/ft_llava-onevision-qwen2-7b-si_ng8_bs1_gas8_epoch1_20240909_09_42_47_20240910_213428.txt',
        'ft_llava-onevision-qwen2-7b-si_ng8_bs1_gas8_epoch5_20240909_13_54_06/ft_llava-onevision-qwen2-7b-si_ng8_bs1_gas8_epoch5_20240909_13_54_06_20240910_213219.txt',
 #       'ft_llava-onevision-qwen2-7b-si_ng8_bs1_gas16_epoch1_20240910_22_31_28/',
    ]

    # JSON 파일 처리 및 엑셀 파일 생성
    data_frames = process_json_files(root_dir, file_paths, output_filename)

    selected_algorithms = [
        'gpt-4o-2024-08-06/gpt-4o-2024-08-06_20240906_070053.txt',
        'llava-onevision-qwen2-0.5b-si/llava-onevision-qwen2-0.5b-si_20240906_015732.txt',
        'llava-onevision-qwen2-7b-si/llava-onevision-qwen2-7b-si_20240906_024924.txt',
    ]

    # Select specific metrics to plot
    selected_metrics = [
        "METEOR score",
        "BertScore",
        "ROUGE-1",
        "LLM_Judge"
    ]


        # 'DEST',
        # 'LEFT',
        # 'RIGHT',
        # 'PATH',
        # 'RECOMMEND',
    selected_description = 'RECOMMEND'

    # Plot the bar chart with selected algorithms and metrics
    plot_metrics_bar_chart(data_frames, selected_algorithms, selected_metrics, selected_description)

if __name__ == "__main__":
    main()
