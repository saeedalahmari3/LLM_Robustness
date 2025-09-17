# Large Language Models Robustness Against Perturbation

Saeed S. Alahmari (Najran University-Najran-Saudi Arabia), Lawerence Hall (University of South Florida-Tampa-FL-United States), Peter R. Mouton (SRC Biosciences, Tampa-FL-United States), Dmitry Goldgof (University of South Florida-Tampa-FL-United States)

Large Language Models (LLMs) have demonstrated impressive performance across various natural language processing (NLP) tasks, including text summarization, classification, and generation. Despite their success, LLMs are primarily trained on curated datasets that lack human-induced errors, such as typos or variations in word choice. As a result, LLMs may produce unexpected outputs when processing text containing such perturbations. In this paper, we investigate the resilience of LLMs to two types of text perturbations: typos and word substitutions. Using two public datasets, we evaluate the impact of these perturbations on text generation using six state-of-the-art models, including GPT-4o and LLaMA3.3-70B. Although previous studies have primarily examined the effects of perturbations in classification tasks, our research focuses on their impact on text generation. The results indicate that LLMs are sensitive to text perturbations, leading to variations in generated outputs, which have implications for their robustness and reliability in real-world applications.

---


![Placeholder Figure](https://via.placeholder.com/600x400.png?text=Your+Figure+Here)

*Figure 1: A brief description of the figure and what it illustrates.*

---

## How to Run the Code

### Prerequisites

Below are the main libraries needed to run the code

- Python 3.8
- NumPy
- Pandas
- Ollama 0.4.7
- openai 1.60
- rouge 1.0.1
- sentence-transformers 3.2.1
- together 1.3.14

### Installation

To run the code, follow the following steps. 

1.  Clone the repository:
    ```bash
    git clone https://github.com/saeedalahmari3/LLM_Robustness.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd LLM_Robustness
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

[Explain how to run your code with examples. Include any command-line arguments or configuration options.]

To run the main script, use the following command:

```bash
python eval_robustness.py --task eval --path2CSV [path2CSVfile]
```
The CSVFile should include the original text in one column and the perturbed text in the other column. 

```bash
python eval_robustness.py --task infer --ori_json_path [path2ori_json] --trans_json_path [path2trans_json]
```

where the JSON files are the dataset shown in the data directory.
---

## Contributing

[If you are open to contributions, explain how others can contribute to your project. You can include information on reporting bugs, suggesting features, or submitting pull requests.]

---

## License

[Specify the license for your project. For example, MIT License, Apache 2.0, etc.]

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
