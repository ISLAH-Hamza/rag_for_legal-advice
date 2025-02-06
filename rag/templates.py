

QUESTION_TEMPLATE = """
Answer the following question: {question}
based only on the provided context:
{context}

The answer must be concise, accurate, and written in French.
If the context does not provide relevant information, respond with: "Information non disponible dans le contexte fourni."
"""





QUERY_TRANSLATION = """
You are a legal assistant tasked with breaking down a complex legal question into specific sub-questions for retrieval.

Objective: Generate three distinct and relevant sub-questions based on: {question}.

Each query must:
1. Address a specific aspect of the legal issue.
2. Be concise and written in French.
3. Assist in retrieving relevant information.

Output exactly 2 queries, each on a separate line:
"""






IS_IT_LEGAL_QUESTION = """
You are an AI trained to classify questions as either **legal** or **general**.

### **Instructions:**
- If the question relates to **laws, contracts, regulations, legal rights, court cases, legal procedures, or legal disputes**, return **True**.
- If the question is about **everyday topics, opinions, science, general knowledge, health, or personal matters**, return **False**.

### **Examples:**
 **Legal Questions (Return True)**
- "What are the legal requirements for starting a business?"
- "Can my landlord evict me without notice?"
- "What are my rights if I get fired unfairly?"

**General Questions (Return False)**
- "What is the capital of France?"
- "How do I cook pasta?"
- "What is the meaning of life?"
- "Who won the last World Cup?"

### **Question to Classify:**
**{question}**

### **Output (Only return True or False):**
"""
