from langchain.load import dumps, loads
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
from rag.templates import *

warnings.filterwarnings("ignore")


class RAG:
    def __init__(self, llm):
        """
        Initialise la classe RAG.
        
        Paramètres :
        - llm : Modèle de langage utilisé pour générer des réponses.
        """
        self.llm = llm
        self.retriever = None
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


    def indexing(self, file_path, db_path):
        """
        Indexe un fichier PDF et stocke les embeddings dans une base de données vectorielle.
        
        Paramètres :
        - file_path (str) : Chemin du fichier PDF à indexer.
        - db_path (str) : Chemin de la base de données où stocker/charger les embeddings.
        
        Lève une exception si le chemin de la base de données n'est pas fourni.
        """
        if db_path == "None":
            raise ValueError("Veuillez fournir un chemin de base de données pour stocker/charger les embeddings.")
        
        if file_path is None:
            vectors_db = Chroma(persist_directory=db_path, embedding_function=OpenAIEmbeddings())
        else:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            docs = self.splitter.split_documents(docs)
            vectors_db = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory=db_path)
        
        self.retriever = vectors_db.as_retriever(k=3)


    def query_translation(self, query):
        """
        Traduit une requête et récupère les documents pertinents.
        
        Paramètres :
        - query (str) : La requête utilisateur.
        
        Retourne :
        - str : Contenu des documents pertinents fusionnés.
        """
        prompt = PromptTemplate(template=QUERY_TRANSLATION, input_variables=["question"]) | self.llm
        questions = prompt.invoke({"question": query})
        results = []
        
        for question in questions.split("\n"):
            results.append(self.retriever.invoke(question))
        
        relevant_docs = self.reciprocal_rank_fusion(results)
        return '\n'.join([doc.page_content for doc, score in relevant_docs])


    def generate(self, question):
        """
        Génère une réponse basée sur une question donnée.
        
        Paramètres :
        - question (str) : La question posée par l'utilisateur.
        
        Retourne :
        - str : Réponse générée par le modèle de langage.
        """
        is_question_chain = PromptTemplate(template=IS_IT_LEGAL_QUESTION, input_variables=['question']) | self.llm
        r = is_question_chain.invoke({'question': question}).strip()
        
        if r == 'True':
            context = self.query_translation(question)
            prompt = PromptTemplate(template=QUESTION_TEMPLATE, input_variables=["question", "context"]) | self.llm
            return prompt.invoke({"question": question, "context": context}).strip()
        else:
            response_chain = PromptTemplate(template="""Répondez à cette question selon vos connaissances : {question}""", input_variables=["question"]) | self.llm
            return response_chain.invoke({'question': question}).strip()


    def reciprocal_rank_fusion(self, results: list[list], k=10):
        """
        Effectue une fusion des résultats de recherche en utilisant la méthode Reciprocal Rank Fusion (RRF).
        
        Paramètres :
        - results (list[list]) : Liste des résultats de recherche.
        - k (int) : Facteur de pondération utilisé pour la fusion (par défaut : 10).
        
        Retourne :
        - list : Liste triée des documents rerankés avec leurs scores.
        """
        fused_scores = {}
        
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
        
        reranked_results = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
        return reranked_results
