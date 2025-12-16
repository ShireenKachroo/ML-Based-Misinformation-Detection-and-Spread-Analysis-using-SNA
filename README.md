# ML-Based Misinformation Detection and Spread Analysis using SNA

<br>

## Project Overview
The **Misinformation Detection & Propagation Analysis** project develops a machine learning-based system integrated with **Social Network Analysis (SNA)**. It performs two core functions:

1. **Classifying information** as real or misinformation using NLP and ML  
2. **Simulating the spread** of misinformation using an **SIR epidemic model** to study network-level diffusion patterns  

This combination provides actionable insights into **how misinformation originates, spreads, and can be mitigated** in online social networks.

---

## Motivation
Misinformation spreads faster than real news due to emotional appeal and rapid sharing. Most users cannot easily distinguish credible from false information, and the structural dynamics of online networks (hubs, bridges, and communities) amplify the spread. Existing solutions typically only classify misinformation, lacking simulation of propagation or identification of super-spreaders.  

This project bridges that gap by combining **ML-based detection** with **SNA-based propagation simulation**, providing a comprehensive, research-oriented analytical tool.

---

## Features
- **Fake News Detection**
  - Text preprocessing: HTML cleaning (BeautifulSoup), regex filtering, stopword removal, sentiment extraction (TextBlob)  
  - Feature Engineering: TF–IDF and BERT embeddings  
  - Classifiers: Logistic Regression, Linear SVM, Passive Aggressive Classifier, Random Forest, and BERT-based deep learning model  
  - Achieves robust performance on noisy and evolving datasets (~95.6% accuracy with ensemble)  

- **Propagation Simulation**
  - Network modeling using **Barabási–Albert scale-free networks**  
  - Simulation using **SIR (Susceptible–Infected–Recovered) epidemic model**  
  - Tracks misinformation flow across hubs, bridges, and communities  
  - Visualizes peak infection, spread over time, and recovery phases  

- **Analytical Insights**
  - Identifies super-spreader nodes
  - Demonstrates impact of network structure on misinformation diffusion  

---

## Technologies Used

| Component           | Tools / Libraries                                 |
| ------------------- | ------------------------------------------------- |
| Language            | Python                                       |
| Data Processing     | Pandas, NumPy                                     |
| NLP & Feature Eng.  | BeautifulSoup, Regex, TextBlob, TF–IDF, BERT embeddings |
| ML Models           | scikit-learn, PyTorch (BERT with AdamW)          |
| Network Analysis    | NetworkX                                          |
| Visualization       | Matplotlib, Seaborn                               |
| Version Control     | Git & GitHub                                      |

---

## Methodology

### 1. Text Preprocessing
- HTML cleaning with BeautifulSoup  
- Regex-based text cleaning  
- Tokenization, stopword removal, sentiment extraction (TextBlob)  
- Feature engineering: TF–IDF vectors and BERT embeddings  

### 2. Machine Learning Classification
- Traditional ML models: Logistic Regression, Linear SVM, Passive Aggressive, Random Forest  
- Deep Learning: BERT-based classifier with PyTorch & AdamW optimization  
- Fine-tuned hyperparameters and thresholds to achieve robust classification  

### 3. Network Graph Construction
- Users modeled as nodes; interactions/shares as edges  
- Scale-free network constructed via **Barabási–Albert model**  
- Identified **hubs** (high-degree nodes) and **bridges** (high betweenness centrality)  

### 4. SIR Spread Simulation
- Initialized nodes with S/I/R states  
- Applied infection and recovery probabilities per time step  
- Simulated misinformation spread over the network  
- Tracked the number of susceptible, infected, and recovered nodes at each step  

---

## Evaluation

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 94.7% | 
| Linear SVM | 94.5% |
| Random Forest | 95.4% | 
| Ensemble| 95.6% | 

---

## Findings & Insights

### Machine Learning
- Soft Voting Ensemble achieved **highest accuracy and generalization**  
- Robust on unseen data: detects fabricated events, political claims, and social exaggerations  
- BERT validated **semantic complexity**

### Network Propagation
- **Hubs**: cause rapid, localized amplification  
- **Bridges**: enable cross-community spread  
- BA network structure produced realistic propagation patterns  
- Real-time simulations reliable up to ~500 nodes; larger networks slower without optimization  

---

## Conclusion
- End-to-end framework combining ML-based detection and SIR-based propagation modeling  
- Soft Ensemble is the most effective classifier  
- Node roles (hubs & bridges) determine spread potential  
- SIR model provides **realistic approximation of misinformation flow**  
- Educational and analytical tool: visual simulations + metrics support experimentation and teaching  

---

## Future Work
1. **Optimized Transformer Models**: DistilBERT, ALBERT, ONNX acceleration for real-time deployment  
2. **Integration of Real Social Media Data**: Twitter, Reddit, Facebook public data for realistic spread patterns  
3. **Multi-Source Spread Simulations**: concurrent misinformation sources and coordinated influence operations  
4. **Performance & Scalability Enhancements**: GPU rendering, multiprocessing, caching, optimized graph libraries  
5. **Alternative Diffusion Models**: SEIR, Independent Cascade, Linear Threshold, SIHR  
6. **Real-Time Data Stream Integration**: live monitoring, dynamic embeddings, automatic graph updates  

---
## References

1. **Fake News Dataset:** [Clément Bisaillon – Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download)  
   > Used for training and evaluating the misinformation detection models.

2. **Rastogi, A. & Bansal, S. (2022).** Disinformation Detection on Social Media: An Integrated Approach.  
   > Provides methods combining linguistic features, sentiment, and propagation cues for misinformation classification. Strong foundation for ML classification part of this project.

3. **IEEE Access (2023).** New Techniques for Limiting Misinformation Propagation.  
   > Introduces interventions like edge-removal and influence-limiting strategies. Useful for designing SNA-based simulation strategies and understanding network-level diffusion.


## Getting Started

### Clone the Repository
Open your terminal and run:

```bash
git clone https://github.com/your-username/Misinformation-Detection.git
cd Misinformation-Detection


