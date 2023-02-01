from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 512

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index

def ask_the_journal():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')

    RETRIEVE_DOCUMENTS = 0
    SUMMARIZE_DOCUMENTS = 1

    state = RETRIEVE_DOCUMENTS 

    while True:
        if (state == RETRIEVE_DOCUMENTS):
            query = input("What kinds of documents do you want to extract? ")
            
            response = index.query(query, similarity_top_k=50, verbose=True, response_mode="no_text")

            retrieved_docs = []
            for sn in response.source_nodes:
                retrieved_docs.append(readers.Document(sn.source_text))

            state = SUMMARIZE_DOCUMENTS

            index2 = GPTListIndex(retrieved_docs)

            while state == SUMMARIZE_DOCUMENTS:
                query = input("What do you want to now about yourself? ")

                if(query == "exit"):
                    state = RETRIEVE_DOCUMENTS
                    break

                response = index2.query(query, response_mode="compact", verbose=True)
                print(response.response)



if __name__ == '__main__':
    ask_the_journal();
