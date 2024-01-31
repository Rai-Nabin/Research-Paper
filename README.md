# AI Research Paper Reading List

Welcome to my curated list of research papers in the field of Artificial Intelligence. Explore cutting-edge papers on various AI topics.

## Table of Contents
| S.N | Topics |
| ---- | ---- |
| 1 | [Computer Vision](#computer-vision) |
| 2 | [Large Language Model](#large-language-model) |
## Computer Vision
| S.N | Paper | Resources | Abstract | Note Link |
| ---- | ---- | ---- | ---- | ---- |
| 1 | MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications |  |  |  |

## Large Language Model
| S.N | Paper | Resources | Year of Release | Abstract | Note Link |
| ---- | ---- | ---- | :--: | :--- | ---- |
| 1 | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | - [Paper](https://arxiv.org/pdf/2101.03961.pdf)<br>- [Video](https://www.youtube.com/watch?v=iAR8LkkMMIM&ab_channel=YannicKilcher) | 2022 | In deep learning, models typically reuse the same parameters for all inputs. Mixture<br>of Experts (MoE) models defy this and instead select different parameters for each in-<br>coming example. The result is a sparsely-activated model—with an outrageous number<br>of parameters—but a constant computational cost. However, despite several notable suc-<br>cesses of MoE, widespread adoption has been hindered by complexity, communication costs,<br>and training instability. We address these with the introduction of the Switch Transformer.<br>We simplify the MoE routing algorithm and design intuitive improved models with reduced<br>communication and computational costs. Our proposed training techniques mitigate the<br>instabilities, and we show large sparse models may be trained, for the first time, with lower<br>precision (bfloat16) formats. We design models based off T5-Base and T5-Large (Raffel<br>et al., 2019) to obtain up to 7x increases in pre-training speed with the same computational<br>resources. These improvements extend into multilingual settings where we measure gains<br>over the mT5-Base version across all 101 languages. Finally, we advance the current scale<br>of language models by pre-training up to trillion parameter models on the “Colossal Clean<br>Crawled Corpus”, and achieve a 4x speedup over the T5-XXL model. | [link](./llm/switch-transformers/README.md) |
|  |  |  |  |  |  |

## Contribution Guidelines
Feel free to contribute by adding new papers or suggesting improvements. 

