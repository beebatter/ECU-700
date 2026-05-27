# Code Challenge: The "ME Engineering Assistant" Agent

## Objective

Design, build, and package a multi-tool AI agent that can assist our engineers by answering questions about different Electronic Control Unit (ECU) specifications. The agent must be able to intelligently select the appropriate documentation to consult based on the user's query.

This challenge will assess your ability to design and implement robust RAG and agentic workflows, and to package a solution for production on Databricks using MLOps best practices.

## Scenario

An engineer at "ME" needs to quickly cross-reference information from two different product lines: the **ECU-700 Series** (older, documented in Markdown files) and the **ECU-800 Series** (newer, documented in Markdown files).

The goal is to build an agent that, when asked a question like *"What is the maximum operating temperature for the ECU-800b?"* or *"Compare the CAN bus speed of the ECU-750 and the ECU-850"*, can autonomously retrieve and synthesize the necessary information to provide a coherent answer.

## Technical Stack & Constraints

* **Language:** Python
* **Core Logic:** LangChain & LangGraph
* **Vector Storage:** An in-memory vector store (e.g., FAISS, ChromaDB)
* **Models & Environment:** You will be provided access to our LLM and Embedding models on a Databricks workspace
* **Packaging & Deployment:** The final solution must be packaged using **Databricks Asset Bundles (DABs)** and the agent logged to **MLflow**
* **Strategy:** Adhere to "everything-as-code" and "python-package-first" principles. The final solution should not be a monolithic notebook, but a proper installable python package
* **ME ECU Engineering Platform:** You have access to our [BIOS repository](https://github.boschdevcloud.com/bios-eco-mde/ai-platform/) and [Documentation](https://pages.github.boschdevcloud.com/bios-eco-mde/ai-platform/) with examples, templates, and patterns

---

## Challenge Structure & Timeline

**Total Time:** 8-10 hours over **10 days**

This challenge uses a **tiered evaluation approach** that allows you to demonstrate competency across different aspects of senior AI engineering while providing multiple paths to success.

### **Tier 1: Core AI/ML Engineering (60% of evaluation)**
*Essential skills - must be demonstrated for passing grade*

**Requirements:**
- Functional multi-source RAG system with ECU-700/800 document retrieval
- Working LangGraph agent with intelligent query routing
- Basic MLflow model logging with predict() method
- Clear architectural documentation and design rationale

**Success Criteria:**
- Agent correctly answers 8/10 predefined test queries
- Response time <10 seconds per query
- Code passes basic quality checks (pylint score > 85%)

**Recommended Time:** 5-6 hours

### **Tier 2: Production & MLOps Excellence (30% of evaluation)**
*Production readiness - distinguishes senior from mid-level candidates*

**Requirements:**
- Complete DAB packaging with working databricks.yml
- Automated deployment job that builds and logs the model
- Comprehensive testing and validation strategy (documented)
- Performance monitoring and error handling

**Success Criteria:**
- DAB deploys successfully in provided workspace
- MLflow model serves predictions via REST API
- Error handling covers common failure modes

**Recommended Time:** 2-3 hours

### **Tier 3: Innovation & Leadership (10% of evaluation)**
*Advanced capabilities demonstrating technical leadership*

**Requirements (Choose 1-2):**
- Implemented evaluation framework with MLflow evaluation
- Human-in-the-loop mechanisms for low-confidence scenarios
- Scalability strategy with detailed implementation plan
- Advanced agent behaviors (multi-step reasoning, tool use)

**Success Criteria:**
- Custom evaluation metrics implemented and logged
- Advanced features demonstrate clear business value
- Scalability analysis includes concrete implementation steps

**Recommended Time:** 1-2 hours

---

## Core Deliverables

Your final submission must be a well-structured project organized as a **Databricks Asset Bundle (DAB)** that includes:

### **1. Custom MLflow Model**
The complete LangGraph agent, packaged as a custom `mlflow.pyfunc` model with a `predict` method that accepts user queries and returns agent responses.

### **2. Deployment Job**
A `databricks.yml` file defining a Databricks Job that builds necessary resources and logs the MLflow model to the Tracking Server.

### **3. Comprehensive Documentation**
A project `README.md` that serves as the single source of truth, including:

- **Architectural Design:** High-level design and key decisions (chunking strategy, agent graph structure, etc.)
- **Setup & Deployment:** Clear instructions for using the Databricks Asset Bundle
- **Testing & Validation Strategy:** Conceptual framework for production validation, including:
  - Proposed evaluation metrics and automated testing approaches
  - Domain expertise validation methods (e.g., using MLflow evaluation with subject matter expert judgments, golden datasets, or domain-specific benchmarks)
  - Strategies for continuous validation and monitoring of agent performance in production
- **Limitations & Future Work:** Discussion of approach limitations and potential improvements

---

## What We Provide

### **Infrastructure & Access**
- Databricks workspace with LLM and Embedding model permissions
- Small dataset of sample documents (`ECU-700-manual.md`, `ECU-800a.md`, `ECU-800b.md`)

### **Resources & Support**
- **Full Repository Access:** Complete BIOS platform repository with working examples and templates
- **Documentation:** Step-by-step Databricks workspace configuration guides
- **Technical Support:** MS Teams or Email for tooling/environment questions

### **Technical Fallback Options**
- **Vector Store Issues:** Since documents are relatively small, you can implement a fallback strategy that passes document content directly as context to the LLM
- **DAB Deployment Problems:** Manual MLflow logging with detailed deployment plan is acceptable
- **Alternative Frameworks:** If LangGraph proves challenging, alternative agent frameworks are allowed with proper justification

---

## Evaluation Criteria

### **Technical Excellence**
1. **Functionality & Robustness:** Does the agent correctly answer different types of queries? How does it handle edge cases?
2. **Code Quality:** Is the solution well-designed, modular, and maintainable? We value clean code and sound software engineering principles.

### **Production Readiness**
3. **MLOps Integration:** How effectively have you used MLflow and Databricks Asset Bundles? Your ability to deliver a deployable, versioned artifact is a key indicator of seniority.
4. **Deployment Automation:** Does your solution demonstrate production-ready deployment practices?

### **Strategic Thinking**
5. **Architectural Decision-Making:** Your documentation should demonstrate informed design trade-offs and clear articulation of your approach.
6. **Testing Strategy:** How comprehensive and practical is your proposed validation framework?

### **Communication**
7. **Final Presentation:** 45-minute code review and discussion where you present your solution, rationale, and lessons learned.

---

## Bonus Challenges (Optional)

Choose one or more to demonstrate advanced capabilities:

- **Comprehensive Evaluation Framework:** Implement automated testing with predefined engineering questions, domain expertise validation using MLflow evaluation, and performance metrics logging integrated into your Databricks Job.

- **Human-in-the-Loop Integration:** Incorporate mechanisms for handling low-confidence scenarios with human oversight.

- **Enterprise Scalability:** Develop and document a detailed strategy for scaling the solution to handle thousands of documents with frequent updates.

---

## Getting Started

1. **Repository Access:** Clone the BIOS platform repository and review existing DAB examples
2. **Environment Setup:** Follow the workspace configuration guide in the platform documentation
3. **Architecture Planning:** Design your approach using provided patterns and templates
4. **Iterative Development:** Build incrementally across the three tiers
5. **Documentation:** Maintain clear documentation throughout development

## Support & Questions

- **Technical Issues:** Contact us via MS Teams or Email for environment/tooling questions
- **Clarifications:** Don't hesitate to ask about requirements or expectations
- **Resources:** Leverage the full BIOS repository - it's designed to accelerate your development

---

**Timeline:** You have **10 days** to complete this challenge. We recommend distributing your 8-10 hours across the timeline to allow for iterative development and refinement.

**Submission:** Provide repository access and ensure your DAB is deployable in the provided Databricks workspace.

Good luck! We're excited to see your approach to building production-ready AI engineering solutions.
