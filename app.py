import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

st.set_page_config(page_title="🌍 Mauritanie AI Observatory", page_icon="🌍", layout="centered")

st.markdown("""
<style>
    .title-box {
        background: linear-gradient(135deg, #1a5276, #2ecc71);
        color: white; padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    }
    .info-box {
        background: #eafaf1;
        border-left: 4px solid #2ecc71;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>🌍 Mauritanie AI Observatory</h1>
    <p>Agent IA expert sur la Mauritanie — RAG + LangGraph + Groq</p>
</div>
""", unsafe_allow_html=True)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    rag_context: str
    reponse: str

@st.cache_resource
def init_agent():
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, api_key=groq_api_key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = [
        Document(page_content="La population de la Mauritanie est d'environ 4.7 millions en 2023.", metadata={"source": "population"}),
        Document(page_content="Le PIB par habitant de la Mauritanie est d'environ 1700 USD en 2023.", metadata={"source": "gdp"}),
        Document(page_content="Le taux d'alphabétisation en Mauritanie est d'environ 67% en 2023.", metadata={"source": "literacy"}),
        Document(page_content="L'espérance de vie en Mauritanie est d'environ 65 ans en 2023.", metadata={"source": "health"}),
        Document(page_content="Nouakchott est la capitale et la plus grande ville de Mauritanie avec 1.2 million d'habitants.", metadata={"source": "geography"}),
        Document(page_content="L'économie mauritanienne repose sur l'extraction minière, la pêche et l'élevage.", metadata={"source": "economy"}),
        Document(page_content="Le taux de croissance du PIB de la Mauritanie était de 5.2% en 2022.", metadata={"source": "growth"}),
        Document(page_content="La Mauritanie possède des réserves importantes de fer, de cuivre et de pétrole offshore.", metadata={"source": "resources"}),
        Document(page_content="La Mauritanie est un pays d'Afrique de l'Ouest, bordé par le Maroc, l'Algérie, le Mali et le Sénégal.", metadata={"source": "geography2"}),
        Document(page_content="La monnaie officielle de la Mauritanie est l'Ouguiya (MRU).", metadata={"source": "currency"}),
        Document(page_content="La langue officielle de la Mauritanie est l'arabe. Le français est largement utilisé.", metadata={"source": "language"}),
        Document(page_content="La Mauritanie est membre de la Ligue arabe et de l'Union africaine.", metadata={"source": "politics"}),
        Document(page_content="Le secteur minier représente environ 25% du PIB mauritanien, grâce au fer de Zouerate.", metadata={"source": "mining"}),
        Document(page_content="La pêche est un secteur clé, avec une des zones les plus riches d'Afrique.", metadata={"source": "fishing"}),
        Document(page_content="Le taux de pauvreté en Mauritanie est d'environ 31%.", metadata={"source": "poverty"}),
        Document(page_content="La Mauritanie a découvert du gaz naturel offshore (Grand Tortue Ahmeyim) partagé avec le Sénégal.", metadata={"source": "gas"}),
        Document(page_content="Le désert du Sahara couvre environ 75% du territoire mauritanien.", metadata={"source": "geography3"}),
        Document(page_content="Nouadhibou est la deuxième ville de Mauritanie et le principal port du pays.", metadata={"source": "cities"}),
        Document(page_content="La Mauritanie a une superficie de 1 030 700 km².", metadata={"source": "size"}),
        Document(page_content="Le fleuve Sénégal forme la frontière sud de la Mauritanie.", metadata={"source": "river"}),
    ]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="/tmp/chroma_db",
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
Tu es un assistant expert sur la Mauritanie.
Tu réponds toujours en français, de façon claire et précise.
Utilise UNIQUEMENT le contexte suivant pour répondre.
Si la réponse n'est pas dans le contexte, dis-le honnêtement.

Contexte :
{context}

Question : {question}

Réponse :
""")

    def rag_agent(state: AgentState):
        docs = retriever.invoke(state["question"])
        context = "\n".join([doc.page_content for doc in docs])
        return {"rag_context": context}

    def response_agent(state: AgentState):
        messages = prompt.format_messages(context=state["rag_context"], question=state["question"])
        response = llm.invoke(messages)
        return {"reponse": response.content}

    graph = StateGraph(AgentState)
    graph.add_node("rag", rag_agent)
    graph.add_node("response", response_agent)
    graph.set_entry_point("rag")
    graph.add_edge("rag", "response")
    graph.add_edge("response", END)
    return graph.compile()

agent = init_agent()

st.markdown('<div class="info-box">💡 Clique sur une question ou écris la tienne :</div>', unsafe_allow_html=True)

suggestions = ["Quelle est la capitale ?", "Quelle est l'économie ?", "Population ?",
               "Ressources naturelles ?", "Quelle est la monnaie ?", "Taux d'alphabétisation ?"]

cols = st.columns(3)
for i, s in enumerate(suggestions):
    with cols[i % 3]:
        if st.button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state["suggestion"] = s

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Bonjour ! Je suis votre assistant expert sur la Mauritanie. Posez-moi vos questions !"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Posez votre question sur la Mauritanie...")

if "suggestion" in st.session_state and not question:
    question = st.session_state.pop("suggestion")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("🤔 Recherche en cours..."):
            result = agent.invoke({"messages": [], "question": question, "rag_context": "", "reponse": ""})
        st.write(result["reponse"])
    st.session_state.messages.append({"role": "assistant", "content": result["reponse"]})

st.markdown("---")
st.markdown("<center><small>🌍 Mauritanie AI Observatory — LangChain + LangGraph + ChromaDB + Groq</small></center>", unsafe_allow_html=True)
