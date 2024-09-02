import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

class llm_wrapper():
    def __init__(self, llm_model_name):
        # Load the LLaMA 3 model and tokenizer from Hugging Face
        self.token = "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"

        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_auth_token=self.token)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name, use_auth_token=self.token,
                                                              torch_dtype=torch.bfloat16, device_map="auto")

    def generate_llm_response(self, sys_prompt, user_prompt, max_new_tokens=256):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = self.tokenizer.apply_chat_template(messages, 
                                                add_generation_prompt=True, 
                                                return_tensors="pt").to(self.model.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        # outputs = self.model.generate(inputs_ids, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)

        response_ids = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        logging.info(f'#### From LLM ####\n')
        logging.info(f'system_prompt: {sys_prompt}')
        logging.info(f'user_prompt: {user_prompt}')
        logging.info(f'response: {response}')

        return response

if __name__ == '__main__':
    llm_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    # llm_model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
    llm_model = llm_wrapper(llm_model_name)
    
    llm_system = 'A chat between a human and an AI that understands visuals in English. '
    llm_prompt = \
    'Based on the description, select the most appropriate action: \'Follow the path\' or \'Stop and wait\'. Then, explain in one sentence whether walking to the destination is possible and the reason. \n'\
    # 'Description: The destination is on the pedestrian road and located at the center of the visible area. There are trees on the left side of the path. There is a black fence on the right side. There is a brick sidewalk in the visible area. The user is on a pedestrian pathway that runs parallel to a street. The pathway appears to be a designated area for walking, and there are trees and a fence on one side. The user is looking towards the center of the pathway, which extends into the distance with more trees and a street on the other side. The sky is visible, suggesting it is a clear day. There are no visible crosswalks or pedestrian traffic lights in the image provided. \n'\
    'Description: The destination is on the pedestrian road and located at the center of the visible area. There is a brick sidewalk in the visible area. The user is on a pedestrian pathway that runs parallel to a street. The pathway appears to be a designated area for walking, and there are trees and a fence on one side. The user is looking towards the center of the pathway, which extends into the distance with more trees and a street on the other side. The sky is visible, suggesting it is a clear day. There are no visible crosswalks or pedestrian traffic lights in the image provided. \n'\
    'Answer: The user is expected to move along the path. The path is a brick sidewalk that the user can walk on, and there are no obstacles on the path. Therefore, the user can move along the path safely. '\
    'So, the answer is \'Follow the path\'. The path is clear of obstacles, so walking to the destination is possible. \n'\
    \
    # 'Based on the description, select the most appropriate action: \'Follow the path\' or \'Stop and wait\'. Then, explain in one sentence whether walking to the destination is possible and the reason. \n'\
    # 'Description: The destination is on the crosswalk and is not obscured by any objects. There is nothing other than the floor visible in the image to the left of the path. There is a pole on the right side of the path. There is nothing on the path. The user is in front of a crosswalk. There is red pedestrian traffic light in the image. \n'\
    'Description: The destination is on the crosswalk and is not obscured by any objects. There is nothing on the path. The user is in front of a crosswalk. There is red pedestrian traffic light in the image. \n'\
    'Answer: The user is expected to move along the path. There is a red pedestrian signal. The red pedestrian signal means the user cannot walk for safety reasons. Therefore, the user cannot move along the path safely. '\
    'So, the answer is \'Stop and wait\'. It is in front of the crosswalk and the pedestrian traffic light is red, so walking to the destination is impossible. \n'\
    \
    # 'Based on the description, select the most appropriate action: \'Follow the path\' or \'Stop and wait\'. Then, explain in one sentence whether walking to the destination is possible and the reason. \n'\
    # 'Description: The destination is on the pedestrian road and is not obscured by any objects. There are cars on the left side. There is a tree trunk on the right side. There is a cart and a tree trunk on the path. The user is on a sidewalk, and there is a tree in front of them. The pedestrian traffic light is not visible in the image. \n'\
    'Description: The destination is on the pedestrian road and is not obscured by any objects. There is a cart and a tree trunk on the path. The user is on a sidewalk, and there is a tree in front of them. The pedestrian traffic light is not visible in the image. \n'\
    'Answer: The user is expected to move along the path. There is a cart and a tree trunk on the path and they are obstacles blocking the way. Therefore, the user cannot proceed along the path. '\
    'So, the answer is \'Stop and wait\'. A cart and a tree truck block the path, so walking to the destination is impossible. \n'\
    \
    # 'Based on the description, select the most appropriate action: \'Follow the path\' or \'Stop and wait\'. Then, explain in one sentence whether walking to the destination is possible and the reason. \n'\
    # 'Description: The destination is on the pedestrian road and is not obscured by any object. There is a white car on the left side. There is a grassy hill on the right side of the path. There are cars parked on the path. The user is on a sidewalk, and there is a yellow and black diamond-shaped sign on a pole to their right. The sign is at the height of 80cm above the ground, which is typical for pedestrian navigation. The background includes buildings, parked cars, and a street. There are no crosswalks or pedestrian traffic lights visible in the image. \n'\
    'Description: The destination is on the pedestrian road and is not obscured by any object. There are cars parked on the path. The user is on a sidewalk, and there is a yellow and black diamond-shaped sign on a pole to their right. The sign is at the height of 80cm above the ground, which is typical for pedestrian navigation. The background includes buildings, parked cars, and a street. There are no crosswalks or pedestrian traffic lights visible in the image. \n'\
    'Answer: Let\'s think step by step. '
    # 'Reasoning and answer: '

    response = llm_model.generate_llm_response(llm_system, llm_prompt, max_new_tokens=256)

    print(f'Assistant: {response}')