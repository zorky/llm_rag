import chromadb
from langchain.prompts import ChatPromptTemplate

from datetime import datetime
from langchain_community.vectorstores.chroma import Chroma

from utils import logger
from utils.constants import CHROMA_DB, CHROMA_COLLECTION, PROMPT_TEMPLATE, LOGGER
from utils.duration_decorator import measure_time
from utils.model_embeddings import get_embeddings_model

def _init_logger():
    logger.setup_logging()
    log = logger.get_logger(LOGGER)
    return log

@measure_time
def _chroma_embedding():
    """
    Charge le modèle de vecteurs de Chroma
    """
    embedding_model = get_embeddings_model()
    chroma = Chroma(persist_directory=CHROMA_DB, embedding_function=embedding_model)
    return chroma

@measure_time
def query_rag(question, n_results=3):
    chroma = _chroma_embedding()
    results = chroma.similarity_search_with_score(question, n_results)
    # print(results)
    if len(results) == 0:
        print(f"Impossible de trouver des résultats correspondants.")
        prompt = ""
    else:
        # sorted_result = sorted(
        #     results,
        #     key=lambda x: (-x[1], -datetime.fromisoformat(x[0].metadata['timestamp']).timestamp())
        # )
        # context_text = (
        #     "\n\n".join([f"{doc.page_content}\n\nScore: {score}\n\nMetadata: {doc.metadata}"
        #                  + "\n\n--------------------------------------------------------------------------------------"
        #                  for doc, score in sorted_result])
        # )
        context_text = (
            "\n\n".join([f"* Score: {score}\n\n* Metadata: {doc.metadata}\n\n* Document:\n\n{doc.page_content}\n\n"
                         + "\n\n--------------------------------------------------------------------------------------"
                         for doc, score in results])
        )
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=question)

    return prompt

if __name__ == '__main__':
    _init_logger()
    question = "Quels sont les points clés du document ?"
    response = query_rag(question)
    print(response)
