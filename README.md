# Open-Domain Question Answering Goes Conversational via Question Rewriting

[**Tasks**](#task-description) | [**Dataset**](#dataset) | [**Evaluation**](#evaluation) |
[**Paper**](https://arxiv.org/abs/2010.04898) |
[**Citation**](#citation) | [**License**](#license)

We introduce QReCC (**Q**uestion **Re**writing in **C**onversational **C**ontext), an end-to-end open-domain question answering dataset comprising of 14K conversations with 81K question-answer pairs.
The goal of this dataset is to provide a challenging benchmark for end-to-end conversational question answering that includes the individual subtasks of question rewriting, passage retrieval and reading comprehension.
Please refer to our paper [Open-Domain Question Answering Goes Conversational via Question Rewriting](https://arxiv.org/abs/2010.04898) for details.

## Task Description

The task in QReCC is to find answers to conversational questions within a collection of 10M web pages split into 54M passages.
Answers to questions in the same conversation may be distributed across several web pages.

## Dataset

QReCC contains 14K conversations with 81K question-answer pairs.
We build QReCC on questions from [TREC CAsT](https://github.com/daltonj/treccastweb/tree/master/2019/data), [QuAC](https://quac.ai) and [Google Natural Questions](https://github.com/google-research-datasets/natural-questions).
While TREC CAsT and QuAC datasets contain multi-turn conversations, Natural Questions is not a conversational dataset.
We used questions in NQ dataset as prompts to create conversations explicitly balancing types of context-dependent questions, such as anaphora (co-references) and ellipsis.

For each query we collect query rewrites by resolving references, the resulting query rewrite is a context-independent version of the original (context-dependent) question.
The rewritten query is then used to with a search engine to answer the question. Each query is also annotated with answer, link to the web page that used to produce the answer.

Each conversation in the dataset contains a unique `Conversation_no`, `Turn_no` unique within a conversation, the original `Question`, `Context`, `Rewrite`, `Answer` with `Answer_URL` and the `Conversation_source`.

```json
{
  "Context": [
    "What are the pros and cons of electric cars?",
    "Some pros are: They're easier on the environment. Electricity is cheaper than gasoline. Maintenance is less frequent and less expensive. They're very quiet. You'll get tax credits. They can shorten your commute time. Some cons are: Most EVs have pretty short ranges. Recharging can take a while."
  ],
  "Question": "Tell me more about Tesla",
  "Rewrite": "Tell me more about Tesla the car company.",
  "Answer": "Tesla Inc. is an American automotive and energy company based in Palo Alto, California. The company specializes in electric car manufacturing and, through its SolarCity subsidiary, solar panel manufacturing.",
  "Answer_URL": "https://en.wikipedia.org/wiki/Tesla,_Inc.",
  "Conversation_no": 74,
  "Turn_no": 2,
  "Conversation_source": "trec"
}
```

## Evaluation

### Evaluate performance on Retrieval Question Answering task

To evaluate retrieval QA, use [evaluate_retrieval.py](https://github.com/apple/ml-qrecc/blob/main/utils/evaluate_retrieval.py)

### Evaluate performance on Extractive Question Answering task

To evaluate extractive QA, use [evaluate_qa.py](https://github.com/apple/ml-qrecc/blob/main/utils/evaluate_qa.py)

## Citation

Please cite the following if you found QReCC dataset, our [paper](https://arxiv.org/abs/2010.04898), or these resources useful.

```bibtex
@article{qrecc,
  title={Open-Domain Question Answering Goes Conversational via Question Rewriting},
  author={Anantha, Raviteja and Vakulenko, Svitlana and Tu, Zhucheng and Longpre, Shayne and Pulman, Stephen and Chappidi, Srinivas},
  journal={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2021}
}
```

## License

The code in this repository is licensed according to the [LICENSE](LICENSE) file.

The QReCC dataset is licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/3.0/.

## Contact Us

To contact us feel free to email the authors in the paper or create an issue in this repository.
