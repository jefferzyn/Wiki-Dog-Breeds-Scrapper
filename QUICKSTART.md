# 🚀 Quick Start - Streamlit App

## 1-Minute Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run streamlit_app.py
```

App opens at: `http://localhost:8501`

---

## 🎯 What You Get

### 4 Functional Modules:

1. **🤔 Ask Question**
   - Type any question about dogs
   - See relevant Wikipedia content
   - Get AI-powered answers (or context-only)

2. **📋 Questionnaire**
   - Answer 6 lifestyle questions
   - Get personalized breed recommendations
   - Factors: space, activity, experience, family, grooming, allergies

3. **🔍 Search Breed**
   - Look up specific dog breeds
   - Comprehensive breed information
   - Temperament, exercise, grooming, suitability

4. **📊 Session History**
   - View all Q&A from your session
   - Export as JSON file
   - Timestamped interactions

---

## ⚙️ Configuration (In Sidebar)

| Setting | Default | Notes |
|---------|---------|-------|
| URLs Dir | `data/urls` | Breed URL files |
| Output Dir | `data/qa_outputs` | Log storage |
| Use OpenAI | Auto-detect | Requires `OPENAI_API_KEY` |
| Use HuggingFace | Auto-detect | Requires `HF_TOKEN` |
| Max URLs | 0 (all) | Limit for faster testing |

**Pro Tip:** Use `--limit 20` in browser sidebar to test quickly!

---

## 📊 System Status Display

Sidebar shows:
- ✅ Initialization status
- 📊 Number of documents indexed
- 🎯 Which generator is active
- ✅ Evaluators available (Faithfulness, SAS, Retrieval MRR)

---

## 💾 Output Files

### JSON Logs
**Location:** `data/qa_outputs/qa_log_YYYYMMDD_HHMMSS.json`

**Format:**
```json
{
  "metadata": {
    "generated": "2026-04-13T12:34:56.789...",
    "embedding_model": "all-MiniLM-L6-v2",
    "openai_enabled": false,
    "huggingface_enabled": false
  },
  "qa_pairs": [
    {
      "timestamp": "2026-04-13T12:35:00...",
      "session_type": "Streamlit Query",
      "question": "Which breeds are hypoallergenic?",
      "answer": "..."
    }
  ]
}
```

---

## 🎯 What Evaluated?

### ✅ Faithfulness Evaluator
- Checks if answers are grounded in documents
- Prevents hallucinations
- Ensures answers match retrieved context

### ✅ SAS Evaluator
- Semantic similarity scoring
- Compares answers to ground truth (if available)
- Score: 0 (completely different) → 1 (identical)

### ✅ Retrieval MRR
- Mean Reciprocal Rank metric
- Measures if relevant docs ranked high
- Quality check for retrieval

---

## 🔑 Optional: Use LLM Generators

### OpenAI (GPT-4o-mini)
```bash
export OPENAI_API_KEY="sk-..."
streamlit run streamlit_app.py
```
✅ Enables: Full answer generation with citations

### HuggingFace (Mistral-7B)
```bash
export HF_TOKEN="hf_..."
streamlit run streamlit_app.py
```
✅ Enables: Open-source answer generation

### No API Keys?
**Still works!** Shows retrieved context only (no generation).

---

## ⏱️ Expected Performance

| Task | Time |
|------|------|
| First initialization (all docs) | 10-20 min |
| First initialization (20 docs) | 1-2 min |
| Subsequent initializations | Instant (cached) |
| Per-question answering | 5-10 sec |
| Evaluator run (25 samples) | 2-5 min |

---

## 🐛 Troubleshooting

### App Won't Start
```bash
# Clear Streamlit cache
streamlit cache clear

# Try again
streamlit run streamlit_app.py
```

### System Won't Initialize
1. Check `data/urls/` has breed files
2. Try with `--limit 10` in sidebar
3. Check available disk space

### No Answers Generated
- System shows context-only mode (expected without API keys)
- Set `OPENAI_API_KEY` or `HF_TOKEN` to enable LLM

### Evaluators Not Available
- Install: `pip install haystack-evaluators`
- System works fine without them (will show ⚠️ in sidebar)

---

## 📚 More Info

- **Full Setup**: See `STREAMLIT_SETUP.md`
- **Verification**: See `VERIFICATION_CHECKLIST.md`
- **Architecture**: See `STREAMLIT_SETUP.md` → "System Architecture"

---

## ✨ Features Summary

| Feature | Status | Details |
|---------|--------|---------|
| RAG Pipeline | ✅ | Retrieval + Generation |
| Evaluators | ✅ | Faithfulness, SAS, MRR |
| JSON Logging | ✅ | Auto-save all interactions |
| Questionnaire | ✅ | 6-question lifestyle assessment |
| Breed Search | ✅ | Detailed breed information |
| History Export | ✅ | Download as JSON |
| Error Handling | ✅ | Graceful degradation |
| Responsive UI | ✅ | Mobile-friendly Streamlit |

---

## 🎓 Learn More

### Architecture
```
Browser (Streamlit)
    ↓
Backend API (backend_api.py)
    ↓
Haystack RAG Pipeline (qa_program.py)
    ↓
Retriever + Generator
    ↓
JSON Logs
```

### Evaluator Flow
```
Question
    ↓
RAG Pipeline → Answer
    ↓
Faithfulness Evaluator → "Is answer grounded?" 
SAS Evaluator → "How similar to expected answer?"
Retrieval MRR → "Were relevant docs ranked high?"
```

---

## 🎉 You're All Set!

```bash
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` and start exploring dog breeds! 🐕

**Questions saved to:** `data/qa_outputs/qa_log_*.json`
