from typing import Dict
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import re

class NLPManager:
    def __init__(self):
        # initialize the model here
        model_name = "nlp_roberta"
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)
        print("Model loaded")
        
    
    def extract_heading(self, s):
        s = s.lower()
        words_to_numbers = {
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'niner': '9',
            'zero': '0'
        }

        pattern = re.compile(r'\b(' + '|'.join(words_to_numbers.keys()) + r')\b')
        text = re.sub(pattern, lambda x: words_to_numbers[x.group()], s)
        pattern2 = re.compile(r'\d \d \d')
        text = re.findall(pattern2, text)
        try:
            return text[0].replace(' ', '')
        except Exception as e:
            print(s)
            return '000'

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        return {"heading": self.extract_heading(context), 
                "tool": self.question_answerer(context=context, question='What is the tool used?')['answer'], 
                "target": self.question_answerer(context=context, question='What is the target?')['answer']}

    
if __name__ == "__main__":
    print(transformers.__version__)