# IPP-Course

## Summary

### Datsets: storytelling

### Process:

### Models:

### Methologoly:

### Evaluation:

### Interested point:

## Multi-VQG: Generating Engaging Questions for Multiple Images

要点： 

1. 


## Let's Talk! Striking Up Conversations via Conversational Visual Question Generation

要点：

1. 总共分成两部分，先生成故事，再从故事生成问题
2. 生成故事用到了知识图谱增强，问题生成用的是transformer fine-tuned

## Educational Question Generation of Children Storybooks via Question Type Distribution Learning and Event-Centric Summarization

datasets: fairytale QA 

codes:https://github.com/zhaozj89/Educational-Question-Generation

### 要点：

1. we propose a novel question generation method that first learns the question type distribution of an input story paragraph, and then summarizes salient events which can be used to generate high-cognitive-demand questions.

2. 是不是可以在问题的基础上加限制，比如Why did the queen want to kill Snow White+在什么事情之前，更好促进思考，解释前因后果/事件联系

### 做法：

1. To train the event-centric summarizer, we finetune a pre-trained transformer-based sequenceto-sequence model using silver samples composed by educational question-answer pairs.

2. In the first stage, we learn to predict the question type distribution for a given input and add pseudo-label so that after prediction, we can know both the types of questions and how many questions of each type. In the second stage, we extract salient events that are most likely for educators to design questions on and then generate an event-centric summarization of the original input. Finally, in the third stage, Each summarization is used to generate one question

3. we first predict the type distribution of output questions p = (p1, p2, . . . , pT ), where pi denotes the probability of question type i, T is the total number of question types. We then transform the distribution into the number of questions under each question type l = (l1, l2, . . . , lT ). Afterwards, we first generate li summaries of type i with the input paragraph d, and then generate li questions of type i with the corresponding summaries

4. We fine-tune another BART model to generate questions, with the type and order control signals added before the input summary to control the generated results.

### Evaluation（评估）：
1. Rouge-L score
2. BERTScore (Zhang et al., 2020a) to evaluate the semantic similarity of generated questions with the ground-truth questions
3. Krippendoff’s alpha scores for human evaluation

## It is AI’s Turn to Ask Humans a Question: Question-Answer Pair Generation for Children’s Story Books

datasets: FairytaleQA

codes: https://github.com/WorkInTheDark/FairytaleQA_QAG_System

## 要点：

1. 和上一篇很像，但是更多的是直接问答形式的问题

## 做法：

1. to extract candidate answers from the given storybook passages through carefully designed heuristics based on a pedagogical framework; (2) to generate appropriate questions corresponding to each of the extracted answers using a state-of-the-art (SOTA) language model; and (3) to rank top QA-pairs with a specific threshold for the maximum amount of QA-pairs for each section

2. 模型：a heuristics-based answer generation module (AG), followed by a BARTbased (Lewis et al., 2019) question generation module (QG) module fine-tuned on FairytaleQA dataset, and a DistilBERT-based(Sanh et al., 2019) ranking module fine-tuned on FairytaleQA dataset to rank and select top N QA-pairs for each input section.

### Evaluation

1. Mean Average Precision

2. human evaluation：• Readability: The generated QA pair is in readable English grammar and words. • Question Relevancy: The generated question is relevant to the storybook section. • Answer Relevancy: The generated answer is relevant to the question.

## CausalQA: A Benchmark for Causal Question Answering

代码：https://github.com/webis-de/coling-22

数据集：Webis-CausalQA-22

### 要点：

1. Since neither SQuAD nor other large QA benchmarks explicitly label causal questions, the difference in effectiveness between causal and other questions remains unclear, We distinguish different types of causal questions using a novel typology derived from a data-driven, manual analysis of questions from ten large question answering (QA) datasets.

2. we manually analyzed samples and developed a two-dimensional typology of causal questions based on their semantic properties and pragmatic interpretation

3. To category the causal questiom:At the semantic level, we group causal question types in terms of which component of a causal chain a question addresses.  In two dimension: (1)Semantic Dimension: questions about an antecedent & questions about a consequent & questions about the chain （2）The Pragmatic Dimension: solution seeking, knowledge seeking, and opinion seeking.

### Evaluation:

1. ROUGE-L, recall, and F1 and traditional exact match (EM) and F1 measures


## RoViST: Learning Robust Metrics for Visual Storytelling(之后还需要细看每个evaluation怎么评分的）

代码：https://github.com/usydnlp/rovist

数据集：StoryTelling

### 要点：

1. Nowadays, the past works on VST have used existing popular n-gram based metrics such as BLEU, METEOR, ROUGE, CIDEr, and SPICE to evaluate their models, such metrics based on n-gram matching tend to have poor correlation with human evaluation scores and do not explicitly consider other criteria necessary for storytelling such as sentence structure or topic coherence. In addition, these metrics mentioned so far still heavily rely on similarity with references, potentially leading to bias for VST tasks as the references may not fully cover the possible ways to write a story for an image sequence

2. Thus we propose several unreferenced metrics for the VST task based on the three aforementioned criteria: 1) visual grounding, 2) coherence, and 3) non-redundancy

3. i. visual ground: focus on norn, calculated by cos similiarity between text embedding and image embedding, multiplied by inverse document frequency 

   ii. Coherence Scorer: calculated by the softmax function of pooled 1024dimensional vector representation hn, but what is pn? 
   
   iii. Non-redundancy Scorer:we propose calculating the Jaccard Similarity (JS) between and within sentences. The JS is defined as the intersection size divided by the union size of two sets. The result is a score between 0 and 1 where a value closer to 1 means that the story tends to contain less redundancy
 
## Guiding Visual Question Generation

代码：https://github.com/nihirv/guiding-vqg

数据集：VQA datasets v2.0 answering category from Information Maximizing Visual Question Generation

### 要点：

1. 其中一个variant是根据答案种类，对象，和目标问题来生成问题？We use the answer category and objects/concepts based on an image and target question as inputs to our decoder. 或者这个explicit model感觉是根据场景来提问？但是为什么还会有算法随机？

2. implicit variant 没太看懂。。

3. contribution：The first work to explore guiding using object labels in Visual Question Generation; 2) A novel generative Transformer-based set-to-sequence approach for Visual Question Generation; 3) The first work to explore discrete variable models in Visual Question Generation; and 4) A substantial increase in quantitative metrics

4. 
方法：

(1) 在训练阶段，他们什么信息都会用，这样的话会不会不贴合实际情况？
(2) framework: Information Maximizing Visual Question Generation codes: https://github.com/ranjaykrishna/iq
(3) text encoder: Bert pre-trained model
    image encoder: transformer
    text decoder: hugging face pre-trained transformer

### evaluation:
We report BLEU (Papineni et al., 2002), ROUGE (Lin, 2004), CIDEr (Vedantam et al., 2015), METEOR (Lavie and Agarwal, 2007), and MSJ (Montahaei et al., 2019) as evaluation metrics.

baseline model:

We compare our models with four recently proposed VQG models Information Maximising VQG (IMVQG; supervised with image and answer category) (Krishna et al., 2019), What BERT Sees (WBS; supervised with image and image caption) (Scialom et al., 2020), Deep Bayesian Network (DBN; supervised with image, scenes, image captions and tags/concepts) (Patro et al., 2020), and Category Consistent Cyclic VQG (C3VQG; supervised with image and answer category) (Uppal et al., 2020).

### Conclusion: We presented a guided approach to visual question generation (VQG), which allows for the generation of questions that focus on specific chosen aspects of the input image. We introduced three variants for this task, the explicit, implicit, and variational implicit. The former generates questions based on an explicit answer category and a set of concepts from the image. In contrast, the latter two discretely predict these concepts internally, receiving only the image as input. The explicit model achieves SoTA results when evaluated against comparable models.


