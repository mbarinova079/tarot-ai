from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


from pydantic import BaseModel


class TarotRequest(BaseModel):
    question: str
    cards: str


@app.post("/api/tarot")
async def get_tarot_reading(req: TarotRequest):
    try:

        system_prompt = """Ты — опытный таролог с 15-летним стажем. Ты делаешь расклады Таро и трактуешь их живым, тёплым, но честным языком. Ты не гадалка-шарлатанка и не сухой учебник — ты собеседник, который помогает человеку увидеть ситуацию объёмно.

        Структура ответа:
        1. Короткое вступление (1-2 предложения).
        2. По картам — каждая карта в контексте вопроса и позиции.
        3. Общая картина — синтез.
        4. Практический вывод — 2-4 конкретных пункта.

        Правила:
        - Учитывай перевёрнутые карты (если указано).
        - Привязывай всё к вопросу. Не выдумывай факты.
        - Никаких медицинских/юридических диагнозов.
        - Тон тёплый, уважительный, без клише вроде "Вселенная говорит тебе".
        - Длина 300-600 слов."""

        user_prompt = f"Вопрос: {req.question}\nВыпавшие карты: {req.cards}"


        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )

        return {"interpretation": response.choices[0].message.content}

    except Exception as e:
        return {"error": repr(e)}


