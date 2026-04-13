# Command Reference Card - Streamlit Dog Breed QA System

## Quick Commands

### Installation
```bash
# Install all dependencies
pip install -r requirements.txt

# Optional: Install evaluators for advanced metrics
pip install haystack-evaluators
```

### Environment Setup (Optional)
```bash
# Enable OpenAI (GPT-4o-mini) answers
export OPENAI_API_KEY="sk-..."

# Enable HuggingFace (Mistral-7B) answers
export HF_TOKEN="hf_..."
```

### Running the System
```bash
# Start Streamlit app
streamlit run streamlit_app.py

# Browser opens at: http://localhost:8501
```

---

## Streamlit Sidebar Options

### Configuration Section
- **URLs Directory**: Path to breed URL files (default: `data/urls`)
- **Output Directory**: Where logs saved (default: `data/qa_outputs`)
- **Use OpenAI**: Enable GPT-4o-mini (requires API key)
- **Use HuggingFace**: Enable Mistral-7B (requires API key)
- **Max URLs**: 0 = all, >0 = limit (for faster testing)

### Action Buttons
- **🚀 Initialize System**: Load and index documents
- **Status Display**: Shows initialization state and document count
- **Evaluators Status**: Shows which evaluators available

---

## Main Interface Tabs

### 1️⃣ Ask Question
```
Input: User's question about dogs
↓
Processing: Retrieval + Generation (if enabled)
↓
Output: Answer with retrieved documents + save to JSON
```

**Example Questions:**
- "Which dog breeds are best for apartment living?"
- "What breeds are good for first-time owners?"
- "Which breeds have low grooming requirements?"

### 2️⃣ Questionnaire
```
Input: 6-question lifestyle survey
  1. Living Space
  2. Activity Level
  3. Dog Experience
  4. Family/Kids
  5. Grooming Commitment
  6. Allergies
↓
Processing: Answer generation based on preferences
↓
Output: Personalized breed recommendations + save to JSON
```

### 3️⃣ Search Breed
```
Input: Breed name (e.g., "Golden Retriever")
↓
Processing: Comprehensive retrieval + generation
↓
Output: Detailed breed information + save to JSON
```

**Information Provided:**
- Temperament
- Size and weight
- Exercise requirements
- Grooming needs
- Suitability for various living situations
- Best owner types

### 4️⃣ Session History
```
Display: All Q&A from current session
Functions:
  - View full interaction history
  - Expand each entry to see full responses
  - Download session as JSON file
```

---

## JSON Log Structure

### Location
```
data/qa_outputs/qa_log_YYYYMMDD_HHMMSS.json
```

### Format
```json
{
  "metadata": {
    "generated": "2026-04-13T12:34:56.789123",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "openai_enabled": true,
    "huggingface_enabled": false
  },
  "qa_pairs": [
    {
      "timestamp": "2026-04-13T12:35:00.123456",
      "session_type": "Streamlit Query",
      "question": "Which dog breeds are best for apartment living?",
      "answer": "..."
    },
    {
      "timestamp": "2026-04-13T12:36:15.654321",
      "session_type": "Questionnaire",
      "question": "Based on preferences...",
      "answer": "..."
    }
  ]
}
```

### Query JSON Logs
```bash
# Pretty print JSON
jq . data/qa_outputs/qa_log_*.json

# Count Q&A pairs
jq '.qa_pairs | length' data/qa_outputs/qa_log_*.json

# Extract only questions
jq '.qa_pairs[] | .question' data/qa_outputs/qa_log_*.json

# Extract only answers
jq '.qa_pairs[] | .answer' data/qa_outputs/qa_log_*.json

# Filter by session type
jq '.qa_pairs[] | select(.session_type=="Questionnaire")' data/qa_outputs/qa_log_*.json

# Search for specific keyword
jq '.qa_pairs[] | select(.question | contains("apartment"))' data/qa_outputs/qa_log_*.json
```

---

## System Status Indicators

### Sidebar Indicators

| Indicator | Meaning | Action |
|-----------|---------|--------|
| ✅ System Initialized | Ready to use | Proceed with questions |
| ⚠️ Not Initialized | Setup incomplete | Click "Initialize System" |
| ✅ Evaluators Available | All 3 working | Quality metrics enabled |
| ⚠️ Evaluators Not Available | Missing package | Optional: `pip install haystack-evaluators` |

### Response Confidence
- ✅ **Confident Response**: Grounded in retrieved documents
- ❌ **Error Response**: Check error message and retry

---

## System Health Checks

### Verify Installation
```bash
# Check Streamlit
streamlit --version

# Check Haystack
python -c "from haystack import Pipeline; print('Haystack OK')"

# Check Evaluators
python -c "from haystack.components.evaluators import FaithfulnessEvaluator; print('Evaluators OK')"
```

### Test Backend
```bash
python -c "
from backend_api import QABackend
backend = QABackend()
print(backend.get_status())
print(backend.get_evaluators_status())
"
```

### Check Logs Created
```bash
# List all logs
ls data/qa_outputs/qa_log_*.json

# Show latest log
cat data/qa_outputs/qa_log_*.json | jq .
```

---

## Performance Tuning

### For Faster Testing
1. In sidebar, set "Max URLs" to 20-50
2. Click "Initialize System"
3. Should complete in 1-2 minutes

### For Complete System
1. Set "Max URLs" to 0 (all)
2. Click "Initialize System"
3. Will take 10-20 minutes (one-time)
4. Subsequent runs use cache (fast)

### For Production
1. Keep all defaults
2. Set API keys (OpenAI/HuggingFace)
3. Monitor evaluator scores
4. Archive logs periodically

---

## Advanced Usage

### Batch Processing (Command Line)
```bash
python batch_qa_processor.py --limit 50
# Processes first 50 questions from dog_breed_questions.txt
# Results in data/qa_outputs/qa_log_[timestamp].json
```

### Using Backend Directly (Python)
```python
from backend_api import QABackend

# Initialize
backend = QABackend(use_openai=True)
backend.initialize(limit=20)

# Ask question
response = backend.answer_question("Best breeds for families?")
print(response.answer)
print(f"Documents retrieved: {len(response.retrieved_docs)}")

# Check status
print(backend.get_status())
```

### Direct Haystack Usage
```python
from qa_program import DogBreedQA

# Initialize
qa = DogBreedQA(use_openai=True)
qa.initialize()

# Ask question
result = qa.ask("Which breeds are hypoallergenic?")

# Get formatted answer
answer = qa.get_answer("Which breeds are hypoallergenic?")

# Evaluate
eval_results = qa.evaluate_rag_pipeline(
    questions=["..."],
    ground_truth_answers=["..."],
    num_samples=25
)
```

---

## Troubleshooting Checklist

- [ ] All dependencies installed: `pip list | grep haystack`
- [ ] Streamlit running: Check console for no errors
- [ ] Browser at localhost:8501: Page loads without 404
- [ ] Sidebar shows configuration: All fields visible
- [ ] Initialize button works: No errors in console
- [ ] System initialized: Sidebar shows ✅ status
- [ ] Evaluators available: Check sidebar indicator
- [ ] Can ask questions: Get answers with documents
- [ ] JSON logs created: Check `data/qa_outputs/`
- [ ] Can export history: Download button works

---

## Files Reference

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main Streamlit application |
| `backend_api.py` | Backend API wrapper for clean integration |
| `qa_program.py` | Haystack RAG pipeline implementation |
| `batch_qa_processor.py` | Batch processing for large document sets |
| `requirements.txt` | All Python dependencies |
| `data/urls/` | Breed URL files for indexing |
| `data/qa_outputs/` | JSON logs saved here |
| `data/dog_breeds.json` | Cached breed list |
| `QUICKSTART.md` | 1-minute setup guide |
| `STREAMLIT_SETUP.md` | Detailed Streamlit guide |
| `VERIFICATION_CHECKLIST.md` | Full technical verification |
| `SYSTEM_READY_REPORT.md` | Complete system verification report |

---

## Key Metrics

### Evaluators
- **Faithfulness**: 0-1 scale (higher = more grounded)
- **SAS**: 0-1 scale (higher = more similar to expected)
- **MRR**: 0-1 scale (higher = better ranking of relevant docs)

### Performance
- Initialization: 10-20 min (all docs) or 1-2 min (limit 20)
- Per-query: 5-10 seconds
- Memory: ~2GB for 500 documents
- Max throughput: Depends on API rate limits

---

## Keyboard Shortcuts (Streamlit)

| Key | Action |
|-----|--------|
| `r` | Rerun script |
| `c` | Clear cache |
| `s` | Settings |
| `?` | Help |

---

## Version Information

```
Haystack: >=2.0.0
Streamlit: >=1.28.0
Sentence-Transformers: >=2.2.0
Python: 3.8+
```

---

## Support Resources

1. **Documentation**: See `.md` files in project root
2. **Haystack Docs**: https://docs.haystack.deepset.ai
3. **Streamlit Docs**: https://docs.streamlit.io
4. **GitHub Issues**: Open issue in repository

---

**Last Updated**: April 13, 2026
**Status**: ✅ All systems operational
