<h1 align="center">	&#128513; MoE</h1>
<h4 align="center">GRIN: <em>GR</em>adient-<em>IN</em>formed MoE</h4>
<p align="center">
<a href="https://huggingface.co/microsoft/GRIN-MoE">Hugging Face</a>&nbsp | &nbsp <a href="https://github.com/microsoft/GRIN-MoE/blob/main/GRIN_MoE.pdf"> Tech Report</a>&nbsp | &nbsp  <a href="https://github.com/microsoft/GRIN-MoE/blob/main/LICENSE">License</a>&nbsp  | &nbsp <a href="https://github.com/microsoft/GRIN-MoE">Github</a> &nbsp | &nbsp <a href="https://github.com/microsoft/GRIN-MoE/tree/main#usage">Get Started</a>&nbsp
<br>

- With **only 6.6B** activate parameters, GRIN MoE achieves **exceptionally good** performance across a diverse set of tasks, particularly in coding and mathematics tasks.

- GRIN uses **SparseMixer-v2** to estimate the gradient related to expert routing, while the conventional MoE training treats expert gating as a proxy for the gradient estimation. 

- GRIN scales MoE training with **neither expert parallelism nor token dropping**, while the conventional MoE training employs expert parallelism and deploys token dropping.

## Intended Uses

### Primary Use Cases

The model is intended for commercial and research use in multiple languages. The model provides uses for general purpose AI systems and applications which require:

1) Memory/compute constrained environments
2) Latency bound scenarios
3) Strong reasoning (especially code, math and logic)

Our model is designed to accelerate research on language and multimodal models, for use as a building block for generative AI powered features. 

### Use Case Considerations

Our models are not specifically designed or evaluated for all downstream purposes. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fariness before using within a specific downstream use case, particularly for high risk scenarios. Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case.

***Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the license the model is released under.*** 

## Usage

### Command-line Demo

The simpliest way to inference with GRIN-MoE is to run the demo as below, which would setup environment, download model weight, and run inference for a math question. 

```bash
# This script is available at `https://github.com/microsoft/GRIN-MoE/blob/main/demo/demo.sh` and requires docker to run.
curl -s https://raw.githubusercontent.com/microsoft/GRIN-MoE/main/demo/demo.sh | bash -s 
```

### Interactive Demo

Run the following command to play with the model with more questions and customized inputs, which would launch a jupyter notebook at `localhost:8887`. 
```bash
# This script requires docker to run.
docker run --gpus all -p 8887:8887 --rm -it nvcr.io/nvidia/pytorch:24.08-py3 /bin/bash -c 'git clone https://github.com/microsoft/GRIN-MoE.git && jupyter notebook --port 8887 --notebook-dir GRIN-MoE/demo'
```

## Benchmarks

To understand the capabilities, we compare GRIN MoE with a set of models over a variety of benchmarks using our internal benchmark platform. At the high-level overview of the model quality on representative benchmarks:

### Popular Benchmarks

|               | GRIN MoE (16x3.8B) | Mixtral (8x7B) | Mixtral (8x22B) | Llama3 (8B) | Llama3 (70B) | GPT3.5 | GPT4o |
|---------------|-----------|---------|---------|--------|--------|--------|-------|
| MMLU          | 79.4      | 70.5    | 76.2    | 66.5   | 80.2   | 71.4   | 86.9  |
| HellaSwag     | 83.7      | 70.4    | 79.0    | 71.1   | 82.6   | 78.8   | 91.7  |
| ANLI          | 60.6      | 55.2    | 65.2    | 57.3   | 68.3   | 58.1   | 75.7  |
| GSM-8K        | 90.4      | 64.7    | 83.8    | 77.4   | 93.5   | 78.1   | 93.8  |
| Math          | 58.9      | 11.1    | 41.8    | 28.2   | 51.2   | 45.3   | 67.8  |
| MedQA         | 70.4      | 62.2    | 67.9    | 60.5   | 78.5   | 63.4   | 88.9  |
| AGIEval       | 48.2      | 45.2    | 54.0    | 42.0   | 56.9   | 48.4   | 37.6  |
| TriviaQA      | 73.9      | 78.5    | 82.2    | 67.7   | 84.5   | 85.8   | 66.0  |
| Arc-C         | 92.0      | 87.3    | 91.3    | 82.8   | 93.0   | 87.4   | 97.0  |
| Arc-E         | 98.0      | 95.6    | 96.9    | 93.4   | 98.2   | 96.3   | 99.0  |
| PIQA          | 89.0      | 86.0    | 85.0    | 75.7   | 85.3   | 86.6   | 92.9  |
| SociQA        | 79.5      | 75.9    | 78.2    | 73.9   | 81.1   | 68.3   | 81.4  |
| BigBench-Hard | 81.4      | 69.7    | 81.8    | 51.5   | 80.2   | 68.3   | 81.2  |
| WinoGrande    | 81.4      | 62.0    | 75.3    | 65.0   | 83.3   | 68.8   | 89.3  |
| OpenBookQA    | 89.8      | 85.8    | 88.6    | 82.6   | 91.8   | 86.0   | 95.2  |
| BoolQ         | 83.4      | 77.6    | 82.7    | 80.9   | 89.1   | 79.1   | 90.6  |
| CommonSenseQA | 81.8      | 78.1    | 82.0    | 79.0   | 84.4   | 79.6   | 88.5  |
| TruthfulQA    | 74.5      | 60.1    | 67.4    | 63.2   | 81.9   | 85.8   | 85.6  |
| HumanEval     | 74.4      | 37.8    | 39.6    | 60.4   | 78.7   | 62.2   | 92.1  |
| MBPP          | 80.3      | 60.2    | 70.7    | 67.7   | 81.3   | 77.8   | 90.4  |
| Average       | 78.6      | 66.7    | 74.5    | 67.3   | 81.2   | 73.8   | 84.8  |

### Livebench
Performance on LiveBench-2024-07-25. Models are ranked by their average score (AVG). *Baseline results are referenced from the official benchmark.

|                              | Reasoning | Coding   | Mathematics  | Data Analysis | Language | IF       | AVG      |
|------------------------------|-----------|----------|--------------|---------------|----------|----------|----------|
| Claude-3-haiku*              | 29.3      | 24.5     | 25.7         | 41.5          | 30.1     | 64.0     | 35.9     |
| Mixtral-8x22B-instruct-v0.1* | 29.3      | 32.0     | 28.3         | 31.7          | 26.5     | 63.1     | 35.2     |
| GPT-3.5-turbo-0125*          | 26.7      | 27.7     | 26.9         | 41.2          | 24.2     | 60.5     | 34.5     |
| **GRIN MoE**                 | **35.3**  | **23.7** | **29.8**     | **32.0**      | **16.9** | **57.6** | **32.5** |
| Mistral-small-2402*          | 26.0      | 21.2     | 28.2         |  31.9         | 22.1     | 63.9     | 32.2     |
| Command-r-plus*              | 28.7      | 19.5     | 24.9         |  24.6         | 23.9     | 71.5     | 32.2     |
| Gemma-2-9B-it*               | 17.3      | 22.5     | 24.0         |  35.1         | 27.6     | 61.6     | 31.3     |


## Training

### Model
|                     |     |
|---------------------|-----| 
| Developer           | Microsoft |
| Architecture        | GRIN MoE has 16x3.8B parameters with **6.6B active parameters** when using 2 experts. The model is a mixture-of-expert decoder-only Transformer model using the tokenizer with vocabulary size of 32,064. |
| Inputs              | Text. It is best suited for prompts using chat format. |
| Context length      | 4K tokens |
| GPUs                | 512 H100-80G |
| Training time       | 18 days |
| Training data       | 4.0T tokens |
| Outputs             | Generated text in response to the input |
| Dates               | Trained between April and June 2024 |
| Status              | This is a static model trained on an offline dataset with cutoff date October 2023 for publicly available data. Future versions of the tuned models may be released as we improve models. |
| Supported languages | English |
| Release date        | Sep 2024 |
| License             | MIT |

### Training Datasets
Our training data includes a wide variety of sources, totaling 4 trillion tokens, and is a combination of 1) publicly available documents filtered rigorously for quality, selected high-quality educational data, and code; 2) newly created synthetic, “textbook-like” data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.); 3) high quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness. More details about data can be found in the [Phi-3 Technical Report](https://arxiv.org/pdf/2404.14219).

## Responsible AI Considerations
Like other language models, Gradient Informed (GRIN) MoE model can potentially behave in ways that are unfair, unreliable, or offensive. Some of the limiting behaviors to be aware of include:  
* Quality of Service: GRIN MoE is trained primarily on English text. Languages other than English will experience worse performance. English language varieties with less representation in the training data might experience worse performance than standard American English.   
* Representation of Harms & Perpetuation of Stereotypes: This model can over- or under-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real-world patterns and societal biases. 
* Inappropriate or Offensive Content: This model may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case. 
* Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.  
* Limited Scope for Code: Majority of the training data is based in Python and use common packages such as "typing, math, random, collections, datetime, itertools". If the model generates Python scripts that utilize other packages or scripts in other languages, we strongly recommend users manually verify all API uses.   

Developers should apply responsible AI best practices and are responsible for ensuring that a specific use-case complies with relevant laws and regulations (e.g. privacy, trade, etc.). Important areas for consideration include: 
* Allocation: The model may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (ex: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.
* High-Risk Scenarios: Developers should assess suitability of using models in high-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (ex: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context. 
* Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).   
* Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case. 
* Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.
* Copyrighted content: The model might generate content that infringes on copyright protections. Developers should implement measures to detect and filter copyrighted material, and end-users should be informed about the potential for unintended copyright violations and the importance of verifying original sources to avoid legal complications.
* Election Misinformation: Developers should ensure robust verification mechanisms are in place to detect and correct false information regarding elections and should inform users of the need for critical evaluation of AI-generated election-related content to mitigate the spread of misinformation.
  
## License
The model is licensed under the [MIT license](./LICENSE).

## Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
