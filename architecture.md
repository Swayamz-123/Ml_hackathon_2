# Hallucination Detection Pipeline Architecture

This diagram visually details the end-to-end structure of your hallucination detection system. You can copy the code below or view the rendered version directly in your preferred Markdown editor that supports Mermaid diagrams (like GitHub or modern VS Code extensions).

```mermaid
flowchart TD
    %% Custom Styles
    classDef input fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#ecf0f1
    classDef phase fill:#2980b9,stroke:#3498db,stroke-width:2px,color:#ffffff
    classDef feature fill:#e67e22,stroke:#f39c12,stroke-width:2px,color:#ffffff
    classDef model fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#ffffff
    classDef output fill:#8e44ad,stroke:#9b59b6,stroke-width:3px,color:#ffffff
    classDef subbox fill:transparent,stroke:#95a5a6,stroke-width:1px,stroke-dasharray: 5 5

    %% Data Input
    subgraph Input ["Data Input"]
        A("Text A<br/>(Candidate 1)"):::input
        B("Text B<br/>(Candidate 2)"):::input
    end

    Input --> P1

    %% Phase 1
    subgraph Phase1 ["Phase 1: Unsupervised Language Learning"]
        P1("Pooled Corpus Preprocessing"):::phase
        T1["BPE Tokenizer<br/>(Trained from scratch)"]:::phase
        T2["Kneser-Ney LM<br/>(Trigram on Clean Texts)"]:::phase
        T3["TF-IDF Vectorizer<br/>(Fitted on Full Corpus)"]:::phase
        
        P1 --> T1 & T2 & T3
    end

    T1 & T2 & T3 --> P2

    %% Phase 2
    subgraph Phase2 ["Phase 2: Feature Engineering (33 Dimensions)"]
        P2("Pairwise Feature Extraction"):::feature
        
        F1["1. Linguistic<br/>(24 Deltas & Means)"]:::feature
        F2["2. Cross-Text<br/>(4 Jaccards: Entity/Numeric)"]:::feature
        F3["3. Perplexity<br/>(4 LM Overlap Scores)"]:::feature
        F4["4. Semantic<br/>(1 TF-IDF Cosine Score)"]:::feature
        
        P2 -.-> F1 & F2 & F3 & F4
    end

    F1 & F2 & F3 & F4 --> P3

    %% Phase 3
    subgraph Phase3 ["Phase 3: Ensemble Classification"]
        P3("Train & Infer Classifiers<br/>(Repeated Stratified 5-Fold)"):::model
        
        M1["LightGBM<br/>(Gradient Boosted Trees)"]:::model
        M2["Support Vector Machine<br/>(RBF Kernel)"]:::model
        M3["Logistic Regression<br/>(L2 Regularization)"]:::model

        P3 -.-> M1 & M2 & M3
        M1 & M2 & M3 -.-> P3_Avg{"Arithmetic Mean<br/>of Probabilities"}:::model
    end

    P3_Avg --> P4

    %% Phase 4
    subgraph Phase4 ["Phase 4: Swap Consistency / Symmetrisation"]
        P4("Order-Invariant Execution"):::phase
        
        S1["Predict Forward: P(A, B)"]:::phase
        S2["Predict Swapped: P(B, A)"]:::phase
        
        P4 -.-> S1 & S2
        S1 & S2 -.-> P4_Avg{"Symmetrised Avg<br/>0.5 * (P_fwd + (1 - P_swap))"}:::phase
    end

    P4_Avg --> P5

    %% Phase 5
    subgraph Phase5 ["Phase 5: Conservative Pseudo-Labelling"]
        P5{"Is test pair confidence<br/>extremely high?<br/>(p < 0.02 or p > 0.98)"}:::feature
        
        P5_Yes["Add as Pseudo-Label"]:::feature
        P5_Retrain["Retrain Ensemble<br/>From Scratch"]:::feature
        
        P5 -- Yes --> P5_Yes --> P5_Retrain
    end

    P5_Retrain --> Final
    P5 -- No --> Final

    %% Output
    Final((("Final Prediction<br/>(Class 1 or 2)"))):::output
```
