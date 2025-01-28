import os
from langchain.load import dumps, loads
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from api import *
from pathlib import Path
import warnings


warnings.filterwarnings("ignore")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing! Check api.py file.")

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

Output exactly 3 queries, each on a separate line:
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

class RAG:
    
  def __init__(self,llm):
    self.llm=llm
    self.retriever=None


  def indexing(self,source_path,splitter):
    path=Path(source_path)
    if path.is_dir():
      vectors_db=Chroma(persist_directory=source_path, embedding_function=OpenAIEmbeddings())
    else:
      loader=PyPDFLoader(source_path)
      docs=loader.load()
      docs=splitter.split_documents(docs)
      vectors_db=Chroma.from_documents(docs, OpenAIEmbeddings(),persist_directory=self.vector_db_path)
      
    self.retriever=vectors_db.as_retriever(k=4)  

  def query_translation(self,query):
      prompt=PromptTemplate(template=QUERY_TRANSLATION,input_variables=["question"]) | self.llm
      questions=prompt.invoke({"question":query})
      results=[]
      for question in questions.split("\n"):
        results.append(self.retriever.invoke(question))
      relevant_docs=self.reciprocal_rank_fusion(results)
      return '\n'.join([doc.page_content for doc,score in relevant_docs])
    
  def generate(self,question):
    
    is_question_chain=PromptTemplate(template=IS_IT_LEGAL_QUESTION,input_variables=['question']) | self.llm
    r=is_question_chain.invoke({'question':question}).strip()
    if r =='True':
      context=self.query_translation(question)
      prompt=PromptTemplate(template=QUESTION_TEMPLATE,input_variables=["question","context"]) | self.llm
      return prompt.invoke({"question":question,"context":context}).strip()
    else:
      response_chain=PromptTemplate(template="""respod to this from your knwoledge {question}""",input_variables=["question"]) | self.llm
      return response_chain.invoke({'question':question})
  # tool
  def reciprocal_rank_fusion(slef,results: list[list], k=10):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [ (loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

    return reranked_results
  



