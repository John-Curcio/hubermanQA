import constants
import os
import argparse
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
import tiktoken
import re


os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-model", "--model", help="model name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("-temperature", "--temperature", help="temperature", type=float, default=0)
    parser.add_argument("-from_scratch", "--from_scratch", help="train from scratch", 
                        action="store_true", default=False)
    return parser


class TranscriptLoader(object):

    def __init__(self, path, chunk_size=1000, 
                 chunk_overlap=None, model_name="gpt-3.5-turbo"):
        self.path = path
        self.chunk_size = chunk_size
        if chunk_overlap is None:
            chunk_overlap = chunk_size // 2
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.markdown_text = None
        self.texts = None
        self.title = None
        self.metadatas = None

    def count_tokens(self, text):
        """
        Count the number of tokens in the text.
        """
        return len(self.encoding.encode(text))

    def parse_text(self):
        """
        Load the text from the markdown file, parse it into 
        individual chunks, and parse metadata for each chunk.
        Return list of Documents.
        """
        with open(self.path, 'r') as f:
            text = f.read()
        self.markdown_text = text
        self.texts = re.findall(r'</summary>(.*?)</details>', 
                                self.markdown_text, re.DOTALL)
        match = re.search(r'\*\*(.*?)\*\*', text)
        if match:
            self.title = match.group(1)
        else:
            self.title = ""
        chunk_summaries = re.findall(r'<summary>(.*?)</summary>', 
                                     self.markdown_text, re.DOTALL)
        self.metadatas = [{
                            "title": self.title,
                            "summary": summary,
                            "source": f"{self.title} {summary}"
                          }
                          for summary in chunk_summaries]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_size // 2,
            length_function=self.count_tokens,
        )
        return splitter.create_documents(self.texts, self.metadatas)
        

class DocSearch(object):
    """
    Given a path to a folder of markdown files, load the files,
    parse them into chunks, and index them for search.
    """
    def __init__(self, path, chunk_size=1000, db_persist_dir="db", from_scratch=False):
        self.path = path
        self.chunk_size = chunk_size
        self.db_persist_dir = db_persist_dir
        self.from_scratch = from_scratch
        self.documents = []
        self.docstore = None
        self.vectorstore = None
        self.embeddings = None
        self.splitter = None
        self.searcher = None

    def load_documents(self):
        """
        Load the documents from the markdown files.
        """
        self.documents = []
        for filename in os.listdir(self.path):
            if filename.endswith(".md"):
                loader = TranscriptLoader(os.path.join(self.path, filename))
                loader.parse_text()
                self.documents.extend(loader.parse_text())
        print(f"Loaded {len(self.documents)} documents.")
    
    def index_documents(self):
        """
        Index the documents for search.
        """
        embeddings = OpenAIEmbeddings()
        # try loading the persisted database from disk
        if (not self.from_scratch and self.db_persist_dir and 
                os.path.exists(self.db_persist_dir)):
            self.db = Chroma(persist_directory=self.db_persist_dir,
                             embedding_function=embeddings)
            return
        # otherwise, index the documents
        if len(self.documents) == 0:
            self.load_documents()
        self.db = Chroma.from_documents(self.documents, embeddings, 
            persist_directory=self.db_persist_dir)
        if self.db_persist_dir:
            self.db.persist()

    def search(self, query, k=4, search_type="mmr"):
        """
        Search the documents for the query.
        Parameters:
            query: str
                The query to search for.
            n: int
                The number of results to return.
            search_type: str
                The type of search to perform. Options are:
                    "mmr": Maximal Marginal Relevance
                    "similarity": Similarity search
        """
        if self.db is None:
            self.index_documents()
        return self.db.search(query, k=k, search_type=search_type)


class QA(object):

    def __init__(self, docsearch: DocSearch, model_name="gpt-3.5-turbo",
                 temperature=0, max_tokens_limit=4097, verbose=False):
        self.docsearch = docsearch
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            # OpenAI(
            ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
            ), 
            chain_type="stuff",
            retriever=docsearch.db.as_retriever(),
            reduce_k_below_max_tokens=True,
            max_tokens_limit=max_tokens_limit,
            verbose=True,
        )
        self.verbose = verbose

    def answer(self, question):
        """
        Answer a question.
        """
        callbacks = []
        if self.verbose:
            callbacks = [StdOutCallbackHandler()]
        return self.chain({
            "question": question,
        }, callbacks=callbacks)
    
if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    docsearch = DocSearch("md_transcripts", chunk_size=480, 
                          db_persist_dir="db", from_scratch=args.from_scratch)
    # docsearch.load_documents()
    docsearch.index_documents()
    qa = QA(docsearch, model_name=args.model, 
            temperature=args.temperature, verbose=args.verbose)
    while True:
        try:
            text = input("Huberman awaits your question: ")
            result = qa.answer(text)
            print(result["answer"])
            print(result["sources"])
        except KeyboardInterrupt:
            break

