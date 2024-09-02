from nltk.translate.meteor_score import meteor_score
from bert_score import score
from rouge_score import rouge_scorer
import numpy as np
from eval_xmls_llm2 import eval_text_llm_judge

def eval_text(gt_text, infer_text):

    gt_tokens = gt_text.split()
    infer_tokens = infer_text.split()
    m_score = meteor_score([gt_tokens], infer_tokens)

    P, R, F1 = score([infer_text], [gt_text], lang="en", model_type="bert-base-uncased")
    b_scores = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r_scores = r_scorer.score(gt_text, infer_text)

    return m_score, b_scores, r_scores

gt_texts = [
    'The destination is sidewalk.',
    'There is a chair on the left.',
    'There is a wooden post on the right.',
    'There is a chair along the route.',
    'There are chairs along the route to the destination, making it difficult to walk.',

    'The destination is not the floor surface and is obscured by the car.',
    'There is a car on the left.',
    'There is a wall on the right side.',
    'There is a car on the route.',
    'The route to the destination is blocked by a car, making it impossible to walk.',

    'The destination is Gongwon-gil.',
    'There is a bollard on the left.',
    'There is a bollard on the right.',
    'There is a bollard on the route.',
    'There are bollards along the route to the destination, making it difficult to walk.',
]

infer_texts = [
    'The destination is ahead, just past the bench on the sidewalk.',
    'There is a bench on the left side of the path.',
    'There is a bench and a chair on the right side of the path.',
    'There is a bench and a chair on the path.',
    'Stop and wait. There are benches and chairs on the path, which obstructs the user\'s movement towards the destination.',

    'The destination is ahead, but there is a car and some objects in the way.',
    'There are cars on the left side.',
    'There is a pole and a wall on the right side of the path.',
    'There is a car and some objects on the path.',
    'Stop and wait. There is a car and some objects obstructing the path, making it impossible to walk to the destination safely.',

    'The destination is ahead, leading through a narrow passage with a metal railing on the right side.',
    'There is a metal railing on the left side of the path.',
    'There is a tree trunk on the right side.',
    'There is a metal railing on the path.',
    'Stop and wait. The user is in front of a construction site, which poses a danger, so walking to the destination is not safe.',
]

manual_scores = [0.5, 1.0, 0.0, 1.0, 1.0,
                 1.0, 1.0, 0.5, 1.0, 1.0,
                 0.0, 0.0, 0.0, 0.0, 1.0]

# except recommend
# manual_scores = [0.5, 1.0, 0.0, 1.0,
#                  1.0, 1.0, 0.5, 1.0,
#                  0.0, 0.0, 0.0, 0.0]

# infer_text = "The destination is ahead, just past the bench on the sidewalk."
# infer_text = infer_text.replace(',', '')
# infer_text = infer_text.replace('.', '')
# gt_text = "The destination is sidewalk."
# gt_text = gt_text.replace(',', '')
# gt_text = gt_text.replace('.', '')

final_manual_scores = []
meteor_scores = []
bert_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
llm_scores = []

exclude_recommend = True
for i, infer_text in enumerate(infer_texts):
    if (i+1)%5==0 and exclude_recommend == True:
        continue
    final_manual_scores.append(manual_scores[i])

    gt_text = gt_texts[i]
    m_score, b_scores, r_scores = eval_text(gt_text, infer_text)
    meteor_scores.append(m_score)
    bert_scores.append(b_scores['f1'])
    rouge1_scores.append(r_scores['rouge1'].fmeasure)
    rouge2_scores.append(r_scores['rouge2'].fmeasure)
    rougeL_scores.append(r_scores['rougeL'].fmeasure)
    llm_reason_score, llm_score= eval_text_llm_judge(gt_text, infer_text)
    llm_scores.append(llm_score)

print("The number of sentences=", len(final_manual_scores))
print(meteor_scores)
print(bert_scores)
print(rouge1_scores)
print(rouge2_scores)
print(rougeL_scores)
print(llm_scores)

manual_scores = np.array(final_manual_scores)

meteor_scores = np.array(meteor_scores)
bert_scores = np.array(bert_scores)
rouge1_scores = np.array(rouge1_scores)
rouge2_scores = np.array(rouge2_scores)
rougeL_scores = np.array(rougeL_scores)
llm_scores = np.array(llm_scores)

print('METEOR CORRELATION = ', np.corrcoef(meteor_scores, manual_scores))
print('BERT CORRELATION = ', np.corrcoef(bert_scores, manual_scores))
print('ROUGE1 CORRELATION = ', np.corrcoef(rouge1_scores, manual_scores))
print('ROUGE2 CORRELATION = ', np.corrcoef(rouge2_scores, manual_scores))
print('ROUGEL CORRELATION = ', np.corrcoef(rougeL_scores, manual_scores))
print('LLM_JUDGE CORRELATION = ', np.corrcoef(llm_scores, manual_scores))



