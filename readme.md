# RAG Fundamentals Project

A Retrieval-Augmented Generation (RAG) pipeline built with Python, ChromaDB, and Ollama to answer user queries accurately based on local document context.

---

## Requirements & Quickstart

### Prerequisites
* **Python 3.10+**
* **Poetry** (dependency management)
* **Ollama** installed and running locally

### Installation & Setup

1. **Clone the repository and install dependencies:**
   ```bash
poetry install
```
Ensure Ollama is running and pull your models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```
Run the Application:

```bash
poetry run python main.py
```

### Architecture & Technical Choices
1. Vector Database: ChromaDB
Database: ChromaDB (Persistent Storage)

Why: ChromaDB provides a lightweight, serverless, embedded vector store that requires zero external server infrastructure. Its native Python integration makes it fast and efficient for persistent local vector indexing and query similarity retrieval.

2. Embedding Model: Ollama Embeddings
Model: Ollama Local Embedding Model (e.g., nomic-embed-text / all-minilm)

Why: Local embeddings keep data execution entirely on-device, ensuring zero API costs, reduced network latency, and improved privacy compliance. They generate dense vector representations optimized for cosine/L2 distance search in ChromaDB.

3. Chunking Strategy: Document-Level / Paragraph Chunking
Strategy: File-based structural chunking (or single-document chunking mapped per file).

Why: Retains structural coherence and full context for short-to-medium IT procedures and policy files. Keeping administrative procedures intact ensures the generation model receives complete prerequisites, step-by-step sequences, and security rules without semantic fragmentation across boundaries.

### Step 7: App Testing & Evaluation
**Question 1 (In Context - Backup Process)**
Question: How many steps are to backup files?
```
Retrieved Chunks:

[Chunk 1] Source: 006_Creating_a_Backup_of_Important_Files.txt | Distance: 0.5275
    
[Chunk 2] Source: 002_Resetting_a_Forgotten_PIN.txt | Distance: 0.9400
    
[Chunk 3] Source: 009_Resetting_a_Jammed_Printer.txt | Distance: 0.9795
    
[Chunk 4] Source: 004_Troubleshooting_Issues_with_Microsoft_Office.txt | Distance: 1.0518
    
[Chunk 5] Source: 008_Setting_Up_a_Secure_Wireless_Network.txt | Distance: 1.0931
```

**Final Answer:** 
```
There are 6 steps to backup files:

Identify the important files that need to be backed up.

Choose a backup method (external hard drive, cloud backup, network share, or USB drive).

Connect the backup device to your computer.

Copy the selected files to the backup device.

Verify that the backup device is complete and can be opened.

Schedule regular backups to ensure that the files are always up to date.
```

**Question 2 (In Context - PIN Reset)**
Question: How can i reset my pin?
```
Retrieved Chunks:

[Chunk 1] Source: 002_Resetting_a_Forgotten_PIN.txt | Distance: 0.3695

[Chunk 2] Source: 009_Resetting_a_Jammed_Printer.txt | Distance: 0.8278

[Chunk 3] Source: 004_Troubleshooting_Issues_with_Microsoft_Office.txt | Distance: 1.0545

[Chunk 4] Source: 007_Troubleshooting_Issues_with_Company-Issued_Tablets.txt | Distance: 1.0589

[Chunk 5] Source: 003_Configuring_VPN_Access_for_Remote_Workers.txt | Distance: 1.0704
```

**Final Answer:**
```
To reset your PIN, follow these steps:
    
Go to the company's intranet homepage and click on the "IT Support" link at the top right corner of the page.
    
Click on the "Self-Service" tab and then select "PIN Reset" from the drop-down menu.
    
You will be redirected to the PIN Reset Tool login page.
    
Enter your company username and password in the required fields.
    
Click the "Login" button to authenticate.
    
You will be prompted to answer your security question. Enter your answer in the required field.
    
Click the "Next" button to proceed.
    
Enter a new PIN in the required field. The PIN must be at least 8 characters long and contain a mix of uppercase and lowercase letters, numbers, and special characters.
    
Re-enter your new PIN in the confirmation field to ensure accuracy.
    
Click the "Reset PIN" button to complete the process.
    
Note: You will be asked to confirm your new PIN before it is applied.
```

**Question 3 (In Context - Printer Troubleshooting)**
Question: How can i fixed my printer it is jammed?
```
Retrieved Chunks:

[Chunk 1] Source: 009_Resetting_a_Jammed_Printer.txt | Distance: 0.3023

[Chunk 2] Source: 004_Troubleshooting_Issues_with_Microsoft_Office.txt | Distance: 0.9010

[Chunk 3] Source: 007_Troubleshooting_Issues_with_Company-Issued_Tablets.txt | Distance: 0.9362

[Chunk 4] Source: 002_Resetting_a_Forgotten_PIN.txt | Distance: 0.9806

[Chunk 5] Source: 006_Creating_a_Backup_of_Important_Files.txt | Distance: 1.0469
```
**Final Answer:**
```
To fix a jammed printer, you can try the following steps:
    
Turn off the printer to prevent any further damage or paper jams.
    
Open the printer's access panel to inspect the internal mechanisms. Common locations include the top, front, or back of the printer.
    
Carefully pull out any visible paper scraps or torn pieces.
    
Check the paper path for any obstructions or debris. Remove any dust, dirt, or other blockages that may be contributing to the jam.
    
Realign the paper tray if it was dislodged during the jam.
    
Check and clean the print head to ensure it is free from blockages or dried ink.
    
Reassemble the printer in the reverse order of how you disassembled it.
    
Power on the printer and test it by printing a test page or a document. If the printer is still experiencing issues, refer to the manufacturer's troubleshooting guide or contact the IT helpdesk for further assistance.
```
**Question 4 (Out of Context - Unhandled Topic)**
Question: What is the company policy on remote expense reimbursements for travel?
```
Retrieved Chunks:
No matching chunks found.
```
**Final Answer:**
```
I don't have enough information for it, sorry!
```

**Question 5 (Out of Topic)**
Question: What is 2+2?
```
Retrieved Chunks:
No matching chunks found.
```
**Final Answer:**
```
I don't have enough information for it, sorry!
```

##### Step 8: Reflection

Building this baseline RAG application provided practical insights into vector store retrieval, local embedding performance, and prompt grounding.

What Worked Well:
The integration between ChromaDB and nomic-embed-text performed exceptionally well. In-context queries achieved low distance metrics for top chunks (e.g., 0.3023 for printer jams and 0.3695 for PIN resets). This allowed llama3.2:1b to reliably extract precise, step-by-step IT procedures without hallucinating extra steps or losing document context.

Challenges Encountered:
Preventing the model from answering out-of-context queries was much harder than expected. Initially, when asked about travel reimbursements, vector retrieval fetched loosely related chunks (distance > 0.75), causing the LLM to invent a plausible policy. Resolving this required implementing distance thresholding in the retriever and adding explicit fallback logic to safely decline answering when no relevant chunks were returned.

Future Improvements (Advanced RAG):
To enhance the application further, I plan to implement a Cross-Encoder Re-Ranking stage. As seen in the test output, lower-ranked chunks (ranks 2–5) regularly yielded high distance scores (> 0.90). Implementing a re-ranker will re-score retrieved contexts, filtering out noisy chunks before they reach the LLM to improve generation latency and response precision.
