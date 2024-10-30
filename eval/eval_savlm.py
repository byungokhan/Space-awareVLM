from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import argparse
import glob
import xml.etree.ElementTree as ET
from nltk.translate.meteor_score import meteor_score
from bert_score import score
from rouge_score import rouge_scorer
import os
import numpy as np
from datetime import datetime
import json
import re
import logging
import time
from eval_xmls_llm2 import eval_text_llm_judge, eval_text_llm_judge_w_conciseness
from private import GPT_API_KEY

def init_logging(outdir, model_ckpt_name):
    # 로거 생성
    logger = logging.getLogger('evaluation_logger')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 파일 핸들러 설정
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'{outdir}/{current_time}_evaluation_log_{model_ckpt_name}_.txt')
    file_handler.setLevel(logging.INFO)

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 핸들러를 로거에 추가
    logger.addHandler(file_handler)
    #logger.addHandler(console_handler)

    logger.info('== logger initialized ==')

    return logger

def eval_text(gt_text, infer_text):

    gt_tokens = gt_text.split()
    infer_tokens = infer_text.split()
    num_tokens = len(infer_tokens)
    m_score = meteor_score([gt_tokens], infer_tokens)

    P, R, F1 = score([infer_text], [gt_text], lang="en", model_type="bert-base-uncased")
    b_scores = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r_scores = r_scorer.score(gt_text, infer_text)

    llm_score, llm_score_c = None, None
    while llm_score is None:
        _, llm_score = eval_text_llm_judge(gt_text, infer_text)
    while llm_score_c is None:
        _, llm_score_c = eval_text_llm_judge_w_conciseness(gt_text, infer_text)

    return m_score, b_scores, r_scores, llm_score, llm_score_c, num_tokens

def update_and_logging_results(tag_scores, avg_scores, xml_filename, gt_tag, file_count, results, logger):

    m_score, b_scores, r_scores, llm_score, llm_score_c, num_words = results

    tag_scores[xml_filename]['m_scores'][gt_tag].append(m_score)
    tag_scores[xml_filename]['b_scores'][gt_tag]['precision'].append(b_scores['precision'])
    tag_scores[xml_filename]['b_scores'][gt_tag]['recall'].append(b_scores['recall'])
    tag_scores[xml_filename]['b_scores'][gt_tag]['f1'].append(b_scores['f1'])
    tag_scores[xml_filename]['r_scores'][gt_tag]['rouge1'].append(r_scores['rouge1'].fmeasure)
    tag_scores[xml_filename]['r_scores'][gt_tag]['rouge2'].append(r_scores['rouge2'].fmeasure)
    tag_scores[xml_filename]['r_scores'][gt_tag]['rougeL'].append(r_scores['rougeL'].fmeasure)
    tag_scores[xml_filename]['llm_scores'][gt_tag].append(llm_score)
    tag_scores[xml_filename]['llm_scores_c'][gt_tag].append(llm_score_c)
    tag_scores[xml_filename]['num_words'][gt_tag].append(num_words)

    avg_scores['m_scores'][gt_tag] += (m_score - avg_scores['m_scores'][gt_tag]) / file_count
    avg_scores['b_scores'][gt_tag]['precision'] += (b_scores['precision'] - avg_scores['b_scores'][gt_tag]['precision']) / file_count
    avg_scores['b_scores'][gt_tag]['recall'] += (b_scores['recall'] - avg_scores['b_scores'][gt_tag]['recall']) / file_count
    avg_scores['b_scores'][gt_tag]['f1'] += (b_scores['f1'] - avg_scores['b_scores'][gt_tag]['f1']) / file_count
    avg_scores['r_scores'][gt_tag]['rouge1'] += (r_scores['rouge1'].fmeasure - avg_scores['r_scores'][gt_tag]['rouge1']) / file_count
    avg_scores['r_scores'][gt_tag]['rouge2'] += (r_scores['rouge2'].fmeasure - avg_scores['r_scores'][gt_tag]['rouge2']) / file_count
    avg_scores['r_scores'][gt_tag]['rougeL'] += (r_scores['rougeL'].fmeasure - avg_scores['r_scores'][gt_tag]['rougeL']) / file_count
    avg_scores['llm_scores'][gt_tag] += (llm_score - avg_scores['llm_scores'][gt_tag]) / file_count
    avg_scores['llm_scores_c'][gt_tag] += (llm_score_c - avg_scores['llm_scores_c'][gt_tag]) / file_count
    avg_scores['num_words'][gt_tag] += (num_words - avg_scores['num_words'][gt_tag]) / file_count

    logger.info(f"  # of words: {num_words}")
    logger.info(f"  METEOR Score: {m_score}")
    logger.info(f"  BERTScore Precision: {b_scores['precision']}")
    logger.info(f"  BERTScore Recall: {b_scores['recall']}")
    logger.info(f"  BERTScore F1: {b_scores['f1']}")
    logger.info(f"  ROUGE-1: {r_scores['rouge1'].fmeasure}")
    logger.info(f"  ROUGE-2: {r_scores['rouge2'].fmeasure}")
    logger.info(f"  ROUGE-L: {r_scores['rougeL'].fmeasure}")
    logger.info(f"  LLM Score: {llm_score}")
    logger.info(f"  LLM Score with Conciseness: {llm_score_c}")

    logger.info(f"  [Avg] # of words: {avg_scores['num_words'][gt_tag]}")
    logger.info(f"  [Avg] METEOR Score: {avg_scores['m_scores'][gt_tag]}")
    logger.info(f"  [Avg] BERTScore Precision: {avg_scores['b_scores'][gt_tag]['precision']}")
    logger.info(f"  [Avg] BERTScore Recall: {avg_scores['b_scores'][gt_tag]['recall']}")
    logger.info(f"  [Avg] BERTScore F1: {avg_scores['b_scores'][gt_tag]['f1']}")
    logger.info(f"  [Avg] ROUGE-1: {avg_scores['r_scores'][gt_tag]['rouge1']}")
    logger.info(f"  [Avg] ROUGE-2: {avg_scores['r_scores'][gt_tag]['rouge2']}")
    logger.info(f"  [Avg] ROUGE-L: {avg_scores['r_scores'][gt_tag]['rougeL']}")
    logger.info(f"  [Avg] LLM Score: {avg_scores['llm_scores'][gt_tag]}")
    logger.info(f"  [Avg] LLM Score with Conciseness: {avg_scores['llm_scores_c'][gt_tag]}")

    return tag_scores, avg_scores

def evaluate_vlm(anno_list, model, image_processor, tokenizer, device, logger):

    eval_gt_degree = 'simple'  # or 'complex'
    eval_gt_tags = ['dest', 'left', 'right', 'path']
    for i, tag in enumerate(eval_gt_tags):
        eval_gt_tags[i] = './/' + tag + '/' + eval_gt_degree
    eval_gt_tags.append('.//recommend')
    eval_gt_tags.append('.//recommend/decision')
    eval_infer_tags = ['dest_desc', 'left_desc', 'right_desc', 'path_desc', 'recommend', 'decision']

    tag_scores = {}
    avg_scores = {
        'm_scores': {tag: 0.0 for tag in eval_gt_tags},
        'b_scores': {tag: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for tag in eval_gt_tags},
        'r_scores': {tag: {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0} for tag in eval_gt_tags},
        'llm_scores': {tag: 0.0 for tag in eval_gt_tags},
        'llm_scores_c': {tag: 0.0 for tag in eval_gt_tags},
        'num_words': {tag: 0.0 for tag in eval_gt_tags},
    }

    file_count = 1
    avg_infer_time = 0.0

    num_tp = 0
    num_fp = 0
    num_fn = 0
    num_tn = 0

    for xml_idx, xml_path in enumerate(anno_list):

        # if xml_idx <= 550:
        #     continue

        xml_filename = os.path.basename(xml_path)
        logger.info(f"[{xml_idx+1}] ********** [GT FILENAME]: {xml_path}")
        print(file_count, " : ", xml_path)
        tag_scores[xml_filename] = {
            'm_scores': {tag: [] for tag in eval_gt_tags},
            'b_scores': {tag: {'precision': [], 'recall': [], 'f1': []} for tag in eval_gt_tags},
            'r_scores': {tag: {'rouge1': [], 'rouge2': [], 'rougeL': []} for tag in eval_gt_tags},
            'llm_scores': {tag: [] for tag in eval_gt_tags},
            'llm_scores_c': {tag: [] for tag in eval_gt_tags},
            'num_words': {tag: [] for tag in eval_gt_tags},
        }

        gt_tree = ET.parse(xml_path)
        gt_root = gt_tree.getroot()
        gp_x = gt_root.find("goal_position/x").text.strip()
        gp_y = gt_root.find("goal_position/y").text.strip()
        gp_xy = f"({float(gp_x):.3f}, {float(gp_y):.3f})"

        img_path = xml_path.replace('.xml', '.JPG')
        if not os.path.exists(img_path):
            img_path = xml_path.replace('.xml', '.jpeg')
        if not os.path.exists(img_path):
            img_path = xml_path.replace('.xml', '.jpg')
        if not os.path.exists(img_path):
            img_path = xml_path.replace('.xml', '.png')

        image = Image.open(img_path)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + f"\nPlease provide a brief walking guide for a visually impaired person by analyzing the image based on the goal position {gp_xy}."

        logger.info(f"query: {question}" )
        conv = copy.deepcopy(conv_templates[conv_template])
        #conv.add_generation_prompt = True
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question,
                                          tokenizer,
                                          IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]

        start_time = time.time()
        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        avg_infer_time = ((avg_infer_time * (file_count - 1)) + elapsed_time) / file_count
        logger.info(f"{file_count}: AVG. Inference Time: {avg_infer_time} secs")

        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

        # 태그를 자동으로 닫음
        #xml_output = fix_broken_xml('<root>' + text_outputs[0] + '</root>')

        # 파싱 가능한 XML로 변경된 내용을 출력
        logger.info(f"[ANSWER]: {text_outputs[0]}" )
        fixed_output = text_outputs[0].replace('&', '&amp;')
        output_root = ET.fromstring('<root>' + fixed_output + '</root>')
        #print(ET.tostring(output_root, encoding='unicode'))

        for i, gt_tag in enumerate(eval_gt_tags):

            logger.info(f"***** gt_tag: {gt_tag}" )
            output_desc = output_root.find(eval_infer_tags[i])
            if output_desc == None:
                continue
            else:
                output_desc = output_desc.text.strip()
            logger.info(f"   [answer]: {output_desc}")

            gt_dest_desc = gt_root.find(gt_tag)
            if gt_dest_desc == None or gt_dest_desc.text == None:
                continue
            else:
                gt_dest_desc = gt_dest_desc.text.strip()
            gt_dest_desc = gt_dest_desc.replace("India", "sidewalk")
            logger.info(f"   [gt]: {gt_dest_desc}")

            if "decision" in gt_tag:
                if gt_dest_desc == 'go' and output_desc == 'go':
                    num_tp += 1
                elif gt_dest_desc == 'stop' and output_desc == 'go':
                    num_fp += 1
                elif gt_dest_desc == 'go' and output_desc == 'stop':
                    num_fn += 1
                elif gt_dest_desc == 'stop' and output_desc == 'stop':
                    num_tn += 1
                else:
                    logger.info(f"Strange Decision - gt:{gt_dest_desc}, pred:{output_desc}")
                logger.info(f"decision: #TP={num_tp}, #FP={num_fp}, #FN={num_fn}, #TN={num_tn}")
            else:
                results = eval_text(gt_dest_desc, output_desc)
                tag_scores, avg_scores = update_and_logging_results(tag_scores,
                                                                avg_scores,
                                                                xml_filename,
                                                                gt_tag,
                                                                file_count,
                                                                results,
                                                                logger)
        # Update running averages
        file_count += 1
        decision_result = {'num_tp': num_tp,
                           'num_fp': num_fp,
                           'num_fn': num_fn,
                           'num_tn': num_tn,
                           'recall': num_tp / (num_tp + num_fn) if (num_tp + num_fn) != 0 else 0,
                           'precision': num_tp / (num_tp + num_fp) if (num_tp + num_fp) != 0 else 0,
                           'accuracy': (num_tp + num_tn) / (num_tp + num_fp + num_fn + num_tn) if (num_tp + num_fp + num_fn + num_tn) != 0 else 0
                            }

    return tag_scores, avg_scores, avg_infer_time, decision_result

def main():

    parser = argparse.ArgumentParser(description='Evaluate Space-aware Vision-Language Model')
    parser.add_argument('--model_ckpt_path', type=str, default="./checkpoints/ft_llava-onevision-qwen2-7b-si_20240828_19_06_08", help='ckpt directory')
    parser.add_argument('--model_base_path', type=str, default=None, help='base model directory')
    parser.add_argument('--eval_db_dir', type=str, default='/mnt/data_disk/dbs/gd_space_aware/manual/240826en', help='dataset directory for evaluation')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='directory to save the output scores')
    args = parser.parse_args()

    # Create the output file name
    model_ckpt_name = os.path.basename(args.model_ckpt_path)
    output_dir = os.path.join(args.output_dir, model_ckpt_name)
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # init logger
    logger = init_logging(output_dir, model_ckpt_name)
    # logging parameters
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    # model setting
    model_name = get_model_name_from_path(args.model_ckpt_path)
    print("MODEL NAME: ", model_name)
    device = 'cuda'
    device_map = 'auto'

    # works for 7b
    if '7b' in model_name:
        overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}  # necessary for right results for 7b
    # works for 0.5b
    elif '0.5b' in model_name:
        overwrite_config = {'tie_word_embeddings': True, 'use_cache': True, "vocab_size": 151936} # necessary for right results for 0.5b


    # Add any other thing you want to pass in llava_model_args
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_ckpt_path,
                                                                          args.model_base_path,
                                                                          model_name,
                                                                          overwrite_config=overwrite_config,
                                                                          device_map=device_map)
    print('MAX LENGTH: ', max_length)
    model.eval()
    # model.tie_weights()
    # model.config.image_aspect_ratio = 'pad'  # or "anyres" or None

    # data setting
    anno_list = glob.glob(f'{args.eval_db_dir}/**/*xml', recursive=True)
    # 특정 폴더명을 포함하는 파일만 필터링
    target_folders = ['1-500', 'm1-500']  # exclude 'demo' folder
    filtered_anno_list = [
        xml_file for xml_file in anno_list
        if any(folder in xml_file for folder in target_folders)
    ]

    tag_scores, avg_scores, avg_infer_time, decision_result = evaluate_vlm(filtered_anno_list, model, image_processor, tokenizer, device, logger)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"{model_ckpt_name}_{current_time}.txt"
    output_file_path = os.path.join(output_dir, output_file_name)

    result = {
        "avg_scores": avg_scores,
        "avg_inference_sec": avg_infer_time,
        "decision_result": decision_result,
        "tag_scores": tag_scores,
    }

    # Write the result to the text file
    with open(output_file_path, 'w') as f:
        f.write(json.dumps(result, indent=4))

    print(f"Average scores and tag scores saved to {output_file_path}")

if __name__ == "__main__":
    main()
