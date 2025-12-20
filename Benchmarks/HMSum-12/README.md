HMSum-12 Dataset
=
Overview
-
HMSum-12 is an industrial-scale benchmark for function-level code summarization, constructed from HarmonyOS version 12.
The dataset is designed to support research on generating practically acceptable code summaries that align with developers’ documentation expectations, beyond semantic correctness alone.

Each function in HMSum-12 is paired with an officially released, human-written code summary, curated and reviewed by professional developers, and strictly following industrial documentation standards. We collect the data from [HarmonyOS DEVELOPER.]([http://blog.csdn.net/guodongxiaren](https://developer.huawei.com/consumer/cn/doc/harmonyos-references/development-intro-api))

Dataset Statistics
-
| Item           | Value                                     |
| -------------- | ----------------------------------------- |
| Project        | HarmonyOS                                 |
| Version        | 12                                        |
| Functions Number   | 22,138                                    |
|Packages Number  | 1,123|
|Avg. tokens per summary | 12.81 |
|Max. tokens per summary | 133.0 |
|Avg. parameters per method | 1.6 |
|Max. parameters per method | 330 |
|Avg. inheritance depth | 3.86 |
| Language       | C/C++, ArkTS |
| Summary Source | Officially released documentation         |
| Summary Level  | Function-level                            |

Usage Notes
-
* HMSum-12 is intended for research purposes only, the dataset can be used for:

* Code summarization

* Developer-centric documentation studies

* Evaluation of industrial applicability of LLM-based methods

* When reporting results, please clearly state that HMSum-12 is an industrial benchmark, not a community open-source dataset
