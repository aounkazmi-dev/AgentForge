# 📊 Data Buddy — AI Data Analyst Agent

**Upload a CSV or Excel file. Ask questions in plain English. Get real SQL, real statistics, real charts — not guesses.**

Data Buddy is an agentic AI data analyst built with LangGraph. Instead of hallucinating an answer, it decides *how* to answer — writing SQL, running statistical tests, building interactive charts, or generating its own Python — then explains the result in natural language.

🌐 **Live Demo:** [databuddyhere.streamlit.app](https://databuddyhere.streamlit.app)

<!-- Suggestion: drop a short GIF or screenshot of the chat + a rendered chart right here — READMEs with a visual up top get far more engagement than text-only ones. -->

---

## Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Example Questions](#-example-questions)
- [What It Can Actually Do](#-what-it-can-actually-do)
- [Limitations](#-current-limitations)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## 🚀 Features

- 📂 Upload a CSV or Excel dataset — no setup, no schema definitions
- 💬 Ask questions in plain English
- 🗄️ Automatic SQL generation and execution (via DuckDB)
- 📊 Statistical analysis — correlation, hypothesis testing, descriptive statistics
- 📈 Interactive Plotly visualizations, rendered inline
- 📝 AI-generated insight reports — not just numbers, an actual narrative
- 🐍 A Python code-execution fallback for questions no built-in tool covers
- ⚡ Fast inference via Groq (Llama 4 Scout)
- 🧠 Multi-tool agent orchestration built with LangGraph
- 🔒 Per-session data isolation — safe for multiple simultaneous users

---

## 🧠 How It Works

Data Buddy runs an **agentic loop**, not a single prompt-and-response. The LLM acts as an orchestrator: it looks at the question, decides which tool actually answers it, executes that tool, reads the result, and only then writes a final response — so the numbers it reports come from real computation, not from the model's memory.

```
                        User Question
                              │
                              ▼
                     ┌─────────────────┐
                     │    Agent Node    │  ← LLM decides what's needed
                     └─────────────────┘
                              │
              ┌───────────────┼────────────────┬───────────────┐
              ▼               ▼                ▼               ▼
          SQL Tool     Statistics Tools    Chart Tool     Report Tool
        (DuckDB query)  (correlation,     (Plotly bar/    (full dataset
                         t-test,           line/scatter/   overview +
                         describe)         histogram/box)  narrative)
              │               │                │               │
              └───────────────┴────────────────┴───────────────┘
                              │
                    Covers the question?
                       │             │
                      Yes            No
                       │             │
                       │             ▼
                       │     Python Code Fallback
                       │   (LLM writes its own code,
                       │    run in a sandboxed exec())
                       │             │
                       └─────┬───────┘
                              ▼
                   Natural Language Final Answer
```

If none of the five structured tools fit the question, the agent writes a short Python snippet directly in its response instead of forcing a bad tool call. That snippet runs in a locked-down sandbox (no filesystem or network access) using the loaded `pandas` DataFrame, and the result gets fed back to the agent to explain — this is what lets Data Buddy handle open-ended questions beyond its built-in tools, without needing a new tool written for every possible request.

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| **Agent orchestration** | LangGraph, LangChain Core |
| **LLM inference** | Groq API, Llama 4 Scout |
| **Data processing** | Pandas, DuckDB, NumPy, SciPy |
| **Visualization** | Plotly |
| **Frontend** | Streamlit |

---

## 📁 Project Structure

```
data-buddy/
├── app.py                # Streamlit frontend — chat UI, file upload, chart rendering
├── backend.py             # LangGraph graph: state, routing, agent node, code-exec fallback
├── tools.py                # The 5 structured tools (SQL, stats, chart, report) + per-session storage
├── utils.py                 # Shared helpers (text extraction, code-block detection, sandbox rules)
├── data_ingestion.py         # CSV/Excel → pandas DataFrame + DuckDB table + schema summary
└── requirements.txt
```

The backend is intentionally split into three files instead of one large one — `tools.py` owns *what the agent can do*, `backend.py` owns *how the agent decides and loops*, and `utils.py` holds small helpers shared by both the backend and frontend.

---

## 🏁 Getting Started

**Prerequisites:** Python 3.10+, a free [Groq API key](https://console.groq.com)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/data-buddy.git
cd data-buddy

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Run it
streamlit run app.py
```

Then open the local URL Streamlit prints, upload a CSV or Excel file, and start asking questions.

---

## ✨ Example Questions

- "Which product generated the highest revenue?"
- "Show me a bar chart of average price by category."
- "Is there a correlation between age and income?"
- "Is there a significant difference in scores between the two groups?"
- "Give me a full report on this dataset with key insights."
- "What percentage of rows have missing values in more than one column?" *(too specific for any built-in tool — this is what triggers the Python fallback)*

---

## 📈 What It Can Actually Do

**SQL Queries** (via DuckDB)
Filtering, aggregation, `GROUP BY`, sorting, multi-condition queries — anything expressible in a single SQL statement against the uploaded table.

**Statistical Analysis**
- Pearson correlation between two numeric columns
- Independent two-sample t-tests (for a numeric column split by a two-category column)
- Descriptive statistics: count, mean, std, min, max, quartiles for numeric columns; value counts for categorical columns

**Visualizations**
Bar, line, scatter, histogram, and box plots — with optional aggregation (mean/sum/count/median) and color grouping.

**AI Reports**
A full-dataset pass: descriptive stats across all numeric columns, missing-value summary, the strongest correlations found, and category breakdowns — synthesized by the LLM into an actual narrative (what stands out, what it might mean), not just a table dump.

**Python Fallback**
For anything the tools above don't cover — the agent writes its own code, which runs in a sandboxed environment with `pandas`, `numpy`, and `scipy.stats` available, but no filesystem or network access.

> Statistical tests beyond correlation/t-test (ANOVA, chi-square, etc.) and chart types beyond the five listed aren't built-in tools yet — but the Python fallback can often produce them anyway when asked directly.

---

## ⚠️ Current Limitations

- Runs on the **free tier of the Groq API** — if the quota is exhausted, requests may temporarily fail until it resets.
- Conversations are **session-based, not persistent** — refreshing the browser starts a clean session by design, so chat history isn't saved across visits.
- One dataset per session — uploading a new file replaces the previous one.
- Occasional tool-calling hiccups are possible with the current model; the app retries automatically but a rare answer may need to be re-asked.

---

## 🚀 Roadmap

- [ ] Downloadable PDF/report export
- [ ] Multiple datasets / joins across tables
- [ ] Persistent chat history (opt-in)
- [ ] Additional statistical tests (ANOVA, chi-square)
- [ ] Database connectors (PostgreSQL, MySQL, Snowflake)
- [ ] Dashboard generation
- [ ] RAG over uploaded documents (PDFs, docs alongside data)
- [ ] Docker deployment
- [ ] FastAPI backend for a production-grade deployment

---

## 👨‍💻 Author

**Aoun Kazmi**

If you found this useful or interesting, a ⭐ on the repo goes a long way.
