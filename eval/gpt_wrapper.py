import openai
import base64
import logging
import time

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class gpt_wrapper():
    def __init__(self, llm_model_name="gpt-4o-mini-2024-07-18", api_key=None):
        # OpenAI API 키 설정 (환경 변수에서 가져옴)
        openai.api_key = api_key
        self.openai_model = llm_model_name #"gpt-4-vision-preview"

        self.max_retry = 5
        self.retry_delay = 2

    def generate_llm_response(self, sys_prompt, user_prompt, image_path=None, max_tokens=1024, seed=17, temperature=1.0):
        if image_path is not None:
            # 이미지를 base64로 인코딩
            encoded_image = encode_image_to_base64(image_path)
        else:
            encoded_image = None
        
        if len(sys_prompt) == 0:
            messages = []
        else:
            messages = [
                {"role": "system", "content": sys_prompt}
            ]

        print(f'@gpt - automatic user input: {user_prompt}')

        user_content = [
            {
                "type": "text", 
                "text": user_prompt
            }
        ]
        if encoded_image is not None:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                        "detail": "high"    # "low", "high", "auto"
                    }
                } #, "mimetype": "image/jpeg"
            )
            encoded_image = None

        messages.append(
            {
                "role": "user",
                "content": user_content
            }
        )

        for attempt in range(self.max_retry):
            try:
                completion = openai.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    seed=seed,
                    temperature=temperature
                )
                response = completion.choices[0].message.content

                logging.info(f'#### From GPT ####\n')
                logging.info(f'system_prompt: {sys_prompt}')
                logging.info(f'user_prompt: {user_prompt}')
                logging.info(f'response: {response}')

                return response
            except openai.OpenAIError as e:
                logging.info(f"OpenAIError Error: {e}")

                if attempt < self.max_retry - 1:
                    logging.info(f"{self.retry_delay}second after {attempt+1}-th try again.")
                    time.sleep(self.retry_delay)
                else:
                    print(f"OpenAI API call all failed, skip this request.")

                    return None
                

if __name__ == '__main__':
    llm_model_name = "gpt-4o-mini-2024-07-18"
    llm_model = gpt_wrapper(llm_model_name)
    
    llm_system = 'A chat between a human and an AI that understands visuals in English. '
    llm_prompt = \
    'Based on the description, select the most appropriate action: \'Follow the path\' or \'Stop and wait\'. Then, explain in one sentence whether walking to the destination is possible and the reason. \n'\
    'Description: The destination is on the pedestrian road and located at the center of the visible area. There are trees on the left side of the path. There is a black fence on the right side. There is a brick sidewalk in the visible area. The user is on a pedestrian pathway that runs parallel to a street. The pathway appears to be a designated area for walking, and there are trees and a fence on one side. The user is looking towards the center of the pathway, which extends into the distance with more trees and a street on the other side. The sky is visible, suggesting it is a clear day. There are no visible crosswalks or pedestrian traffic lights in the image provided. \n'\
    'Answer: The user is expected to move along the path. The path is a brick sidewalk that the user can walk on, and there are no obstacles on the path. Therefore, the user can move along the path safely. '\
    'So, the answer is \'Follow the path\'. The path is clear of obstacles, so walking to the destination is possible. \n'\
    \
    'Description: The destination is on the crosswalk and is not obscured by any objects. There is nothing other than the floor visible in the image to the left of the path. There is a pole on the right side of the path. There is nothing on the path. The user is in front of a crosswalk. There is red pedestrian traffic light in the image. \n'\
    'Answer: The user is expected to move along the path. There is a red pedestrian signal. The red pedestrian signal means the user cannot walk for safety reasons. Therefore, the user cannot move along the path safely. '\
    'So, the answer is \'Stop and wait\'. It is in front of the crosswalk and the pedestrian traffic light is red, so walking to the destination is impossible. \n'\
    \
    'Description: The destination is on the pedestrian road and is not obscured by any objects. There are cars on the left side. There is a tree trunk on the right side. There is a cart and a tree trunk on the path. The user is on a sidewalk, and there is a tree in front of them. The pedestrian traffic light is not visible in the image. \n'\
    'Answer: The user is expected to move along the path. There is a cart and a tree trunk on the path and they are obstacles blocking the way. Therefore, the user cannot proceed along the path. '\
    'So, the answer is \'Stop and wait\'. A cart and a tree truck block the path, so walking to the destination is impossible. \n'\
    \
    'Description: The destination is on the pedestrian road and is not obscured by any object. There is a white car on the left side. There is a grassy hill on the right side of the path. There are cars parked on the path. The user is on a sidewalk, and there is a yellow and black diamond-shaped sign on a pole to their right. The sign is at the height of 80cm above the ground, which is typical for pedestrian navigation. The background includes buildings, parked cars, and a street. There are no crosswalks or pedestrian traffic lights visible in the image. \n'\
    'Answer: Let\'s think step by step. '

    response = llm_model.generate_llm_response(llm_system, llm_prompt, max_tokens=1024)

    print(f'Assistant: {response}')