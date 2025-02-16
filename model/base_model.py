from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import uuid
import torch

class BaseModel:
    def __init__(self, uuid, model_name="Qwen/Qwen2.5-3B-Instruct", tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
                 max_new_tokens=100, top_p=0.9, top_k=50, temperature=0.9):
        self.DIALOGS_DIR = "dialogues"
        os.makedirs(self.DIALOGS_DIR, exist_ok=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.uuid = uuid
        self.history = self.load_history()

        # Сохраняем параметры генерации в объекте (меньше кода при вызове)
        self.generation_params = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id
        }

        # Предварительно токенизируем историю (ускорит обработку)
        self.tokenized_history = self.tokenize_history()

    def load_history(self):
        history_path = os.path.join(self.DIALOGS_DIR, f"{self.uuid}.json")
        if os.path.exists(history_path):
            print("Load old dialogue:", self.uuid + ".json")
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print("New dialogue:", self.uuid + ".json")
            return [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}]

    def save_history(self):
        history_path = os.path.join(self.DIALOGS_DIR, f"{self.uuid}.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)

    def tokenize_history(self):
        """Предварительно токенизирует историю, чтобы не делать это заново при каждом запросе."""
        input_text = "\n".join([msg["content"] for msg in self.history])
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        return inputs.input_ids.to("cuda"), inputs.attention_mask.to("cuda")

    def ask(self, question):
        """Добавляет вопрос пользователя, обновляет токенизацию и отправляет запрос модели."""
        self.history.append({"role": "user", "content": question})
        self.save_history()

        # Обновляем токенизированные данные, добавляя новый вопрос
        input_text = "\n".join([msg["content"] for msg in self.history])
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        # Генерация ответа
        output_ids = self.model.generate(input_ids, attention_mask=attention_mask, **self.generation_params)

        # Декодируем ответ
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        self.history.append({"role": "assistant", "content": answer})
        self.save_history()

        return answer

# Пример использования
if __name__ == "__main__":
    dialog_id = str(uuid.uuid4())
    model = BaseModel(dialog_id)
    print(model.history)
    print("Ответ модели:", model.ask("What is your name?"))
