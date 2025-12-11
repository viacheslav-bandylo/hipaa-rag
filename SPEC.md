# Software Requirements Specification (SRS): HIPAA RAG Solution

## 1. Introduction

### 1.1 Purpose
The primary objective of this project is to build a Retrieval-Augmented Generation (RAG) solution designed to effectively process, query, and retrieve information from HIPAA documentation. The system aims to provide accurate, context-informed responses to user queries by sourcing data directly from a provided PDF document.

### 1.2 Scope
* **In-Scope:** The system shall focus exclusively on processing a single combined PDF file containing HIPAA Parts 160, 162, and 164.
* **Out-of-Scope:**
    * Processing of any table-based information contained within the PDF.
    * Processing of any picture-based information contained within the PDF.
    * Multi-document ingestion or complex document management outside the core provided file.

## 2. System Architecture & Technology Stack

The solution must be orchestrated using **Docker Compose**. The system shall be composed of the following mandatory components:

### 2.1 Components
* **Large Language Model (LLM):** The system shall utilize OpenAI or Anthropic Cloud APIs for text generation.
* **Database:** The system shall use **PostgreSQL** for data and metadata storage.
* **Backend API:** The backend shall be developed as an **Async FastAPI** service.
* **Frontend UI:** The user interface shall be a Simple Chat UI built with Gradio, Dash, or a similar UI builder.
* **Reverse Proxy & Access:** The system shall utilize **Nginx** as a reverse proxy and **TryCloudflare** or **ngrok** to provide a secure tunnel for public access.

### 2.2 Architecture Design
The system shall follow a containerized architecture where the User interacts with the Chat UI via a secure tunnel and Nginx reverse proxy. The Chat UI communicates with the Backend API, which orchestrates interactions between the PostgreSQL database (containing the indexed HIPAA documents) and the external AI Model.

## 3. Functional Requirements

### 3.1 Document Ingestion and Indexing
* **REQ-ING-01:** The system shall parse the HIPAA documentation provided as a single combined PDF.
* **REQ-ING-02:** The system shall retain the document's outline and references to subdivisions (e.g., sub-paragraphs, sections) during parsing.
* **REQ-ING-03:** The system shall split text-based knowledge between paragraphs and sub-paragraphs, preserving context where paragraphs refer to other paragraphs.
* **REQ-ING-04:** The system shall efficiently index the parsed structure to facilitate targeted information retrieval.

### 3.2 Querying and Retrieval
* **REQ-QRY-01:** The system shall accept natural language questions submitted by the user.
* **REQ-QRY-02:** The system shall provide accurate answers derived *strictly* from the provided HIPAA documentation.
* **REQ-QRY-03:** The system shall support broader user inquiries, such as requests for lists of inquiry-related HIPAA paragraphs (classified as a "Nice-to-have" feature).

### 3.3 Citation and Quoting
* **REQ-CIT-01:** When a user explicitly requests the full text of a requirement or policy, the system shall directly quote the exact content.
* **REQ-CIT-02:** The system shall precisely reference the original source sub-paragraph or section number for any quoted content.
* **REQ-CIT-03:** The system shall optionally provide structured lists of all reference sub-paragraphs related to a user inquiry.

## 4. Non-Functional Requirements & Constraints

### 4.1 Security
* **REQ-SEC-01:** The system shall implement basic encryption using HTTPS via the secure tunnel/proxy (e.g., TryCloudflare/ngrok).
* **REQ-SEC-02:** The system shall **not** implement user management, registration, authentication, or role-based permissions.
* **REQ-SEC-03:** The system is **not** required to implement HIPAA compliance infrastructure, alignment protocols, or security audits.

### 4.2 Performance and Scalability
* **REQ-PERF-01:** The system shall be designed for low usage and minimal concurrency (maximum of a few simultaneous users).
* **REQ-PERF-02:** Scalability, high availability, clustering, and performance optimizations are explicitly excluded from the scope.

### 4.3 Operational Constraints
* **REQ-OPS-01:** The system shall not implement any analytics, usage tracking, logging, or telemetry.
* **REQ-OPS-02:** The code must follow best practices for Python development and be robust and clear.

## 5. Evaluation Criteria

The success of the solution will be evaluated based on the following indicators:

* **Accuracy:** The ability to deliver correct responses that reflect the factual content of the provided documentation.
* **Attribution:** The precision in providing exact quotes and clear references to source sections/subparagraphs.
* **Code Quality:** The robustness, clarity, and adherence to Python best practices within the codebase.
* **Capabilities:** The successful implementation of reference gathering and exact quoting features.