# Comparative Analysis and Improvement of Chatbot Models

This repository contains a Jupyter Notebook documenting the evaluation and improvement of two chatbot models: a **Bag-of-Words (BoW) classifier** and a more sophisticated **Transformer-based model**. The project demonstrates the end-to-end workflow of benchmarking, analysing and iteratively improving conversational AI systems.

---

## Project Overview
The goal of this project is to explore and compare different approaches to intent classification and response generation in chatbots:

1. **Bag-of-Words Model** – A simple model using frequency-based text representation.
2. **Transformer Model** – A more sophisticated architecture leveraging contextual embeddings.

Key aspects of this work include:
- Designing benchmark tests to fairly compare models.
- Evaluating intent classification accuracy, confidence, and response relevance.
- Data pre-processing
- Visualizing comparative results with clear metrics.
- Iteratively improving the Transformer model to close the performance gap.
- Evolving the test design to explore contrasts in more depth

---

## Key Features
- **Intent Recognition**: Both models were trained on a curated intents dataset.
- **Benchmarking Framework**: Identical prompts were used to assess model performance.
- **Quantitative Metrics**: Accuracy, confidence scores, relevance ratings, and response times.
- **Visual Analysis**: Tables and plots for side-by-side comparisons.
- **Error Analysis & Improvements**: Identifying failure cases and refining the Transformer model.

---

## Results Summary
- The **Bag-of-Words model** achieved strong baseline accuracy with near-perfect classification on most prompts.
- The **Transformer model** initially underperformed but showed potential in understanding subtle differences in meaning and context between user inputs.
- Iterative improvements narrowed the performance gap, showcasing how modern NLP architectures can be optimized.

---

## Technologies Used
- **Python 3**
- **Jupyter Notebook**
- **Natural Language Processing (NLP)**
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Models**: Bag-of-Words classifier, Transformer-based model

---

## Repository Structure
```
├── Comparative Analysis and Improvement of Chatbot Models.ipynb   # Main notebook
├── intents.json                                                   # Text file of intents
├── README.md                                                      # Project documentation (this file)
├── BOW Model                                                      # Files for bag-of-words model
├── Transformer Model                                              # Files for transformer model
├── Improved Transformer Model                                     # Improvements made to the transformer model
├── Saved Graphics                                                 # Graphics displayed in the notebook
├── Saved Tables                                                   # Tables of data used in graphics
```

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   ```
2. Navigate to the project folder:
   ```bash
   cd <your-repo-name>
   ```
4. Make sure the notebook, data sets and intents.json are placed in the same folder for smooth execution (you will need to move the tables from the "Saved Tables" folder). The models should run in their seperate folders but if you would like to re-run the batch-tests you will need to manually move the tables produced into the same folder as the notebook to use        your own data. Note: The saved graphics included in this repo are for illustration only and are not required to re-run the notebook.
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Comparative Analysis and Improvement of Chatbot Models.ipynb"
   ```

---

## What This Project Demonstrates
This project highlights:
- Strong understanding of **NLP fundamentals** and chatbot architectures.
- Ability to **design evaluation frameworks** and conduct rigorous model comparisons.
- Skills in **data analysis, visualization, and model iteration**.
- A mindset for **continuous improvement and error analysis**.

---

## Contact
If you're interested feel free to connect:

- **Name:** [Stanley Poley]
- **LinkedIn:** [https://www.linkedin.com/in/stanley-poley/]
- **Email:** [stanley.poley@gmail.com]

---
