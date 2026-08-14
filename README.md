# Zepto Support Assistant

A small GenAI support assistant built with LangGraph, ChromaDB, Sentence Transformers, Pydantic, and FastAPI.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline for Zepto support-policy questions.

The graded baseline uses deterministic offline mock LLM behavior with `MOCK_LLM` unset or set to `1`. No API key or external LLM service is required.

## Architecture

The pipeline follows:

```text
Documents
   ↓
Ingestion / Chunking
   ↓
Sentence Transformers Embeddings
   ↓
ChromaDB (zepto_policies)
   ↓
LangGraph Intent Router
   ↓
 ┌─────────────────────────────┐
 │                             │
Policy Question          General Question
 │                             │
 ↓                             ↓
retrieve_and_answer       direct_answer
 │                             │
 ↓                             ↓
Grounded response        Fixed mock response
 │                             │
 └──────────────┬──────────────┘
                ↓
       Pydantic validation
                ↓
          FastAPI /ask