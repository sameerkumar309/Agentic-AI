import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from typing import List, Dict

# Load API keys
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="gemma2-9b-it")

# Define State Class
class State(TypedDict):
    user_requirements: str
    user_stories: List[str]
    PO_review_feedback: str
    PO_approved: bool
    design_docs: Dict[str, Dict[str, str]]
    design_review_feedback: str
    design_review_approval: bool

# Streamlit UI
st.title("User Story & Design Document Generator")

# User Input for Requirements
requirements = st.text_area("Enter Project Requirements:")
if st.button("Generate User Stories"):
    userstory_prompt = PromptTemplate(
        template="""You are an expert software developer. Based on the following requirement:
        "{user_requirements}", generate exactly 5 well-structured user stories.
        
        Format:
        1. As a <user>, I want to <action> so that <benefit>.
        
        Provide only the user stories, nothing else.""",
        input_variables=["user_requirements"]
    )
    userstory_chain = userstory_prompt | llm | StrOutputParser()
    response = userstory_chain.invoke({"user_requirements": requirements})
    user_stories = response.strip().split("\n")
    st.session_state.user_stories = user_stories
    st.write("### Generated User Stories:")
    st.write("\n".join(user_stories))

# Product Owner Review
if "user_stories" in st.session_state:
    st.write("## Product Owner Review")
    approval = st.radio("Approve the user stories?", ["Approved", "Needs Improvement"])
    feedback = ""
    if approval == "Needs Improvement":
        feedback = st.text_area("Provide review remarks:")
    if st.button("Submit Review"):
        st.session_state.PO_approved = approval == "Approved"
        st.session_state.PO_review_feedback = feedback

# Design Document Generation
if "PO_approved" in st.session_state and st.session_state.PO_approved:
    st.write("## Generate Design Documents")
    if st.button("Generate Functional & Technical Docs"):
        functional_prompt = PromptTemplate(
            template="""Provide a detailed functional design document for the following user stories:
            {user_stories}

            Format:
            - Purpose
            - Features
            - User Flow
            - Expected Outcomes""",
            input_variables=["user_stories"]
        )
        functional_chain = functional_prompt | llm | StrOutputParser()
        functional_response = functional_chain.invoke({"user_stories": "\n".join(st.session_state.user_stories)})

        technical_prompt = PromptTemplate(
            template="""Provide a detailed technical design document for the following user stories:
            {user_stories}

            Format:
            - Technology Stack
            - System Architecture
            - Database Design
            - APIs and Integrations""",
            input_variables=["user_stories"]
        )
        technical_chain = technical_prompt | llm | StrOutputParser()
        technical_response = technical_chain.invoke({"user_stories": "\n".join(st.session_state.user_stories)})

        st.session_state.design_docs = {"functional": functional_response, "technical": technical_response}
        st.write("### Functional Design Document:")
        st.write(functional_response)
        st.write("### Technical Design Document:")
        st.write(technical_response)
