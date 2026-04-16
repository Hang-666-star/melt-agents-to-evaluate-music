from openai import OpenAI

client = OpenAI(
    api_key="sk-bde5f512f8b346e2b9ea0129d2538016",
    base_url="https://api.deepseek.com"
)

def generate_music_report(evaluations, final_score):

    prompt = f"""
You are a professional music critic.

Evaluation results:
{evaluations}

Final score: {final_score}

Write a detailed music evaluation report including:
1 Melody and Harmony
2 Rhythm and Groove
3 Structure
4 Emotional Expression
5 Innovation
6 Overall evaluation
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role":"system","content":"You are a professional music analyst."},
            {"role":"user","content":prompt}
        ]
    )

    return response.choices[0].message.content