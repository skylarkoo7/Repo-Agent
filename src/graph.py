from typing import Annotated
from typing_extensions import TypedDict

from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
import os
from github import Github
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import agentstack
from operator import itemgetter
import json
import re


class State(TypedDict):
    # inputs: dict[str, str]
    # messages: Annotated[list, add_messages]
    """
    Represents the state of our graph.
    """
    question: str                     # User's question
    generation: str                    # LLM-generated response
    retrievelink:str           # Web search flag (true/false)
    retriever_db:str            # Retrieved documents
    issue_url: str 

class OpensourcerepoGraph:



    @agentstack.task
    def extract_issue_urls(self, state):
        """Extracts issue URLs and updates the GitHub issue with a reference link."""
        github_token = os.getenv("GITHUB_TOKEN")
        
        if not github_token:
            raise ValueError("❌ Error: Missing GitHub token. Ensure GITHUB_TOKEN is set.")

        g = Github(github_token)

        def extract_issue_number(response_text):
            match = re.search(r'https://github.com/[^/]+/[^/]+/issues/(\d+)', response_text)
            return int(match.group(1)) if match else None

        def update_issue(issue_number, response_text):
            repo_owner = os.getenv("repo_ownerfenil")
            repo_name = os.getenv("repo_namefenil")
            update_issue_number=extract_issue_number(state.get("issue_url"))
            if not repo_owner or not repo_name:
                raise ValueError("❌ Error: Missing repo_owner or repo_name.")

            try:
                repo = g.get_repo(f"{repo_owner}/{repo_name}")
                issue = repo.get_issue(number=update_issue_number)
                existing_body = issue.body if issue.body else ""

                print(f"Existing issue body:\n{existing_body}")

                # Prevent duplicate updates
                if response_text in existing_body:
                    print(f"Issue #{issue_number} already contains the reference link. No update needed.")
                    return

                # Construct the comment content
                comment_text = f"""
    Related Issue:
    The issue similar issue can be found at the following URL: {state["retrievelink"]}
    """

                # Add a comment instead of modifying the issue body
                issue.create_comment(comment_text)
                print(f"✅ Successfully added a comment to Issue #{issue_number}.")

            except Exception as e:
                print(f"❌ Error updating issue #{issue_number}: {e}")

        # Extract issue number from response text
        response_text = state.get("retrievelink", "")
        issue_number = extract_issue_number(response_text)

        if issue_number:
            update_issue(issue_number, response_text)
        else:
            print("No issue link found in response.")

        return {**state, "updated_issue": issue_number}



    @agentstack.task
    def create_vector_db(self, state):
        repo_owner = os.getenv("repo_owner")
        repo_name = os.getenv("repo_name")
        github_token = os.getenv("GITHUB_TOKEN")

        if not repo_owner or not repo_name or not github_token:
            raise ValueError("Missing required environment variables: repo_owner, repo_name, or GITHUB_TOKEN")

        g = Github(github_token)
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        issues = repo.get_issues(state="all")  # Fetch all issues (open & closed)
        
        issue_docs = [
            Document(page_content=f"Issue Title: {issue.title}\n\n{issue.body}", 
                    metadata={"url": issue.html_url})
            for issue in issues
        ]
        
        if not issue_docs:
            print("No issues found in the repository.")
            return {**state, "retriever_db": None}

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, chunk_overlap=50
        )
        
        # Make splits
        splits = text_splitter.split_documents(issue_docs)

        # Create vector database
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        retriever = vectorstore.as_retriever()
        
        return {**state, "retriever_db": retriever}



    @agentstack.agent
    def full_rag_chain(self, state):
        
        def get_unique_union(documents: list[list]):
            """ Unique union of retrieved docs """
            
            def document_to_dict(doc):
                """Convert LangChain Document object to a JSON-serializable dictionary."""
                return {
                 "page_content": doc.page_content,
                    "metadata": doc.metadata
                }

            # Flatten list of lists, and convert each Document to string
            flattened_docs = [json.dumps(document_to_dict(doc)) for sublist in documents for doc in sublist]
            # Get unique documents
            unique_docs = list(set(flattened_docs))
            # Return
            return [Document(**json.loads(doc)) for doc in unique_docs]

        # Multi Query: Different Perspectives
        question = state["question"]
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. 

        Original question: {question}"""

        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_perspectives 
            | ChatOpenAI(temperature=0) 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

        # Ensure retriever is valid
        retriever = state.get("retriever_db")
        if retriever is None:
            raise ValueError("Retriever not found in state. Ensure retriever_db is correctly initialized.")

        # Retrieval pipeline
        retrieval_chain = generate_queries | retriever.map() | get_unique_union
        docs = retrieval_chain.invoke({"question": question})

        if not docs:
            return {"messages": ["No relevant documents found."]}

        print(f"Retrieved {len(docs)} documents.")

        # RAG Answer Generation
        answer_template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(answer_template)
        llm = ChatOpenAI(temperature=0)

        final_rag_chain = ( 
            {"context": retrieval_chain, "question": itemgetter("question")} 
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response_text = final_rag_chain.invoke({"question": question})

        return {**state, "retrievelink": response_text}

   
   
    def run(self, inputs: dict):
        tools = ToolNode([])
        self.graph = StateGraph(State)
        
                
        self.graph.add_node("create_vector_db", self.create_vector_db)
        self.graph.add_node("full_rag_chain", self.full_rag_chain)
        self.graph.add_node("extract_issue_urls", self.extract_issue_urls)
        
        self.graph.add_edge(START, "create_vector_db")
        self.graph.add_edge("create_vector_db", "full_rag_chain")
        self.graph.add_edge("full_rag_chain", "extract_issue_urls")
        self.graph.add_edge("extract_issue_urls", END)
        

        # self.graph.add_node("tools", tools)
        
        # self.graph.add_conditional_edges("opensourcerepo", tools_condition)
        # self.graph.add_edge("tools", "opensourcerepo")

        app = self.graph.compile()
            
        # result_generator=app.stream(initial_state)
        # print(result_generator)
        
        # final_state=app.invoke({'inputs':inputs})
        # print(final_state)
        
        result_generator = app.stream(inputs)
        print(result_generator)
        
        for message in result_generator:
            print("\n🔹 Agent Output:")
            for key, item in message.items():
                for m in item.get("messages", []):
                    print(f"{key}: {m.content}")
        print("\n🔹 Final State:", inputs)
