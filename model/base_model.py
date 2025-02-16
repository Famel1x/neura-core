import os
import json
import uuid
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

class BaseModel:
    def __init__(self, uuid, model_name="ministral-8b-2410"):
        self.DIALOGS_DIR = "dialogues"
        os.makedirs(self.DIALOGS_DIR, exist_ok=True)

        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY не найден в .env файле")
        
        self.client = Mistral(api_key=self.api_key)
        self.model_name = model_name
        self.uuid = uuid
        self.history = self.load_history()
    
    def get_dialog_path(self):
        return os.path.join(self.DIALOGS_DIR, f"{self.uuid}.json")

    def load_history(self):
        """Загружает историю сообщений из JSON-файла."""
        file_path = self.get_dialog_path()
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return [{"role": "system", "content": "Ты Zeno, мультимодальный агент помогающий в домашних делах"}]

    def save_history(self):
        """Сохраняет историю сообщений в JSON-файл."""
        file_path = self.get_dialog_path()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)

    def ask(self, question):
        """Отправляет запрос в Mistral API и сохраняет ответ."""
        self.history.append({"role": "user", "content": question})
        self.save_history()

        response = self.client.chat.complete(
            model=self.model_name,
            messages=self.history,
            max_tokens=100  # Ограничиваем размер ответа
        )
        answer = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": answer})
        self.save_history()

        return answer

# Пример использования
if __name__ == "__main__":
    dialog_id = str(uuid.uuid4())  # Генерируем уникальный ID диалога
    model = BaseModel(dialog_id)
    print(f"Новый диалог: {model.uuid}")
    print(model.ask("Как тебя зовут?"))
