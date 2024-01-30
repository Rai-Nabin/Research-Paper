## Overview


![Title](./images/title.png)The paper discusses a **model with a trillion parameters**, which is claimed to be significantly larger than GPT-3's 175 billion parameters. However, the comparison between the two models is debatable because the trillion parameters are not utilized in the same manner as in traditional transformers. 

The new architecture, referred to as the **Switch Transformer**, builds upon the concept of **Mixture of Experts** (MoE), a pre-existing idea. The MoE approach **involves dividing the feed-forward layer into experts**, and in this case, each token is routed to only one expert in a sparse manner.

The **Switch Transformer** takes the idea of MoE to an extreme by implementing hard routing of information, allowing **each token to be directed to a single expert per layer**. This sparse approach enables scaling of the number of experts and parameters in the model without increasing computational requirements during a forward pass. This unique architecture **allows for an increase in the model's parameters without a proportional increase in computational complexity**, making it distinct from conventional transformer models.

To **ensure stable training**, the paper introduces new techniques such as **selective dropout**, **selective casting of parameters to different precisions**, and **improved initialization**. 

Despite the catchy title of a trillion parameters, **most of the experiments in the paper are conducted with models in the order of billions of parameters**. The trillion-parameter model, while intriguing, does not perform as well as their smaller models, suggesting that working with such large models may still be challenging and resource-intensive.

The paper concludes by cautioning that trillion-parameter models may not be practical or widely adopted soon due to their complexity, cost, and potentially fuzzy performance. The comparison is made to the original ResNet paper, which presented a 1,000-layer convolutional neural network, even though contemporary ResNets typically have fewer layers. The ability to build such models is **highlighted as a demonstration of capability** rather than an immediate practical implementation.

