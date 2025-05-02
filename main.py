import os
from pymongo import MongoClient
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from datetime import datetime

# === Config ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = "mongodb://192.168.48.200:27017/"
DB_NAME = "test"
COLLECTION_NAME = "drive"
LOG_COLLECTION = "conversations"

# === Setup ===
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ Set your OPENAI_API_KEY in environment variables.")

# MongoDB Connection
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
summary_collection = db[COLLECTION_NAME]
log_collection = db[LOG_COLLECTION]

# === LangChain LLM Setup ===
llm = ChatOpenAI(model="gpt-4", temperature=0.4)

# === Templates ===
ASK_QUESTION_TEMPLATE = """
You are a strict technical interviewer. Ask ONE specific technical question strictly based on the content of this summary:

SUMMARY:
{summary}

Do NOT introduce unrelated concepts. Keep it concise and focused.
"""

FOLLOWUP_TEMPLATE = """
You are a technical interviewer. Evaluate the candidate's answer in detail.

If the answer is off-topic or irrelevant (e.g., a name or nonsense), respond with:
"This answer is unrelated to the topic discussed. Please stay focused on the Linux concepts covered."

Otherwise:
- Provide technical feedback
- Ask exactly ONE follow-up question tied to the original summary.

SUMMARY:
{summary}

CANDIDATE ANSWER:
"{answer}"

Respond with only feedback and a relevant follow-up.
"""

# === Utilities ===
def get_latest_summary() -> tuple[str, str]:
    doc = summary_collection.find_one(sort=[("timestamp", -1)])
    if not doc:
        raise ValueError("No lecture summaries found in MongoDB.")
    return doc["file_name"], doc["summary"]

def log_qa(file_name: str, question: str, answer: str, followup: str):
    log_collection.insert_one({
        "file": file_name,
        "question": question,
        "answer": answer,
        "followup": followup,
        "timestamp": datetime.now()
    })

def print_tokens(response):
    usage = response.response_metadata.get("token_usage", {})
    print(f"🔢 Tokens used: prompt={usage.get('prompt_tokens')} | completion={usage.get('completion_tokens')}")

# === Main Interview Flow ===
def run_interviewer():
    print("🤖 Technical Interviewer Agent (Auto from MongoDB)\n")

    file_name, summary = get_latest_summary()
    print(f"📄 Loaded summary from: {file_name}\n")

    prompt = PromptTemplate.from_template(ASK_QUESTION_TEMPLATE)
    question_resp = llm.invoke(prompt.format(summary=summary))
    question = question_resp.content
    print(f"🤖 Interviewer: {question}")
    print_tokens(question_resp)

    while True:
        answer = input("👤 You: ").strip()
        if answer.lower() in ["exit", "quit"]:
            print("👋 Session ended.")
            break

        followup_prompt = PromptTemplate.from_template(FOLLOWUP_TEMPLATE)
        followup_resp = llm.invoke(followup_prompt.format(summary=summary, answer=answer))
        followup = followup_resp.content
        print(f"\n🤖 Interviewer: {followup}")
        print_tokens(followup_resp)

        log_qa(file_name, question, answer, followup)
        question = followup  # update for context

if __name__ == "__main__":
    run_interviewer()
