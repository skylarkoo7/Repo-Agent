# 🛠 Open Source Repo Agent (AgentStack + Multi-Retrieval Query)

This project is an **AgentStack-powered AI pipeline** that assists in managing **GitHub issues** efficiently. It uses **Retrieval-Augmented Generation (RAG) with Multi-Retrieval Queries** to analyze GitHub issues, extract relevant information, and update issues with related discussions.

## 🚀 Features

- **Retrieves and processes GitHub issues** from a repository.
- **Uses Multi-Retrieval Queries** to improve similarity search by reformulating queries from different perspectives.
- **Stores issues in a vector database (Chroma)** for efficient retrieval.
- **Extracts issue URLs** and **adds comments** to GitHub issues with related discussions.
- **Runs as a structured pipeline using `agentstack`**.

## 🛠 How It Works

The agent follows a structured pipeline:

### 1️⃣ `create_vector_db` (Task)
- Retrieves all issues from the GitHub repository.
- Stores them in ChromaDB for efficient retrieval.

### 2️⃣ `full_rag_chain` (Agent)
- Uses Multi-Retrieval Query Generation to generate alternative versions of the user query.
- Reformulates queries from different perspectives to enhance similarity-based retrieval.
- Retrieves the most relevant GitHub issues by overcoming distance-based limitations.
- Generates an AI-assisted response.

### 3️⃣ `extract_issue_urls` (Task)
- Extracts relevant issue links from the generated response.
- Adds a comment to the corresponding GitHub issue.

### 4️⃣ **Pipeline Execution (`run()`)**
- Defines and executes the entire workflow using `agentstack`.
- Ensures that each step updates the state correctly.

---

## 🔍 Multi-Retrieval Query for Improved Similarity Search

### What is Multi-Retrieval Query?

Instead of using a single search query, the Multi-Retrieval Query method generates five alternative perspectives of the user's query to:

- Improve similarity-based search results.
- Reduce distance-based search errors.
- Ensure better document retrieval.

### Example of Multi-Retrieval Query Reformulation

#### **Input Query:**
```
"How do I extract a datetime from a timestamp?"
```

#### **Generated Alternative Queries:**
1. "Convert timestamp to datetime format in Python."
2. "How can I parse timestamps into datetime objects?"
3. "Best ways to retrieve datetime from a timestamp."
4. "Extract date and time from a given timestamp."
5. "Methods to transform a timestamp into a readable date."

These alternative queries increase retrieval accuracy, helping the AI find more relevant GitHub issues.

---

## 📌 Example Output

When the agent runs, it automatically adds a comment to the GitHub issue:

```
✅ Successfully added a comment to Issue #123:
The issue titled "How do I extract a datetime from a timestamp?" can be found at:
https://github.com/YOUR_REPO/issues/456
```

---

## 🔍 Troubleshooting

### 1️⃣ GitHub Authentication Issues?
Make sure your `GITHUB_TOKEN` is correct and has repo permissions.

### 2️⃣ LangChain Deprecation Warnings?
Use updated imports:
```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
```

### 3️⃣ State Update Errors?
Ensure that each task/agent returns an updated state:
```python
return {**state, "retrievelink": response_text}
```

---

## 🛠 Technologies Used

- **Python 3.9+**
- **AgentStack**
- **LangChain**
- **ChromaDB**
- **GitHub API (PyGitHub)**
- **Multi-Retrieval Query Generation**

---

## 📬 Contact

For questions or contributions, feel free to reach out:

- **GitHub:** [Neel Harsola](https://github.com/skylarkoo7)
- **Email:** work.neelharsola@gmail.com

---


