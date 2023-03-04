# IPP-Course

## Summary

### Datsets: storytelling

### Process:

### Models:

### Methologoly:

### Evaluation:

### Interested point:

# Contents  
- [Multiple images Question Generation](#Multiple-images-Question-Generation)  
- [Question Generation based Question types](#Question-Generation-based-Question-types)  
- [Current Evaluation of Visual Storytelling](#Current-Evaluation-of-Visual-Storytelling)  
- [Current Visual Question Generation](#Current-Visual-Question-Generation) 
- [Current Methogology on Visual-Language Tasks or Visual Question Generation/Storytelling](#Current-Methogology-on-Visual-Language-Tasks-or-Visual-Question-Generation/Storytelling) 
  - [Pre-trained Language Model](#Pre-trained-Language-Model)  
  - [Knowledge Enhance Methods](#Knowledge-Enhance-Methods)
- [Interesting Paper Related Causal Question](#Interesting-Paper-Related-Causal-Question) 

# Multiple images Question Generation:

## Multi-VQG: Generating Engaging Questions for Multiple Images

### Datasets：https://github.com/academiasinicanlplab/mvqg-dataset-of-generating-engaging-questions-for-multiple-images

### Keypoint： 

1. Motivation：However, most answers to questions in traditional questionanswering (QA) datasets are factoids, which reduce individuals’ willingness to answer. Furthermore, traditional visual question generation (VQG) confines the source data for question generation to single images, resulting in a limited ability to comprehend time-series information of the underlying event. Most questions are do not seek people to reply.

2. The instruction to build the MVQG datasets:Our instruction, on the other hand, asked workers to imagine that they want to have a conversation with people on Twitter and hence to write a question to start that conversation.

3. Model Structure:We choose VL-T5 (Cho et al., 2021) as the backbone in particular because it treats all VL tasks as text-generating tasks, which is appropriate for our question generation scenario. Inspired by Shen et al. (2022), we propose an additional baseline model by employing the visual encoder of CLIP (Radford et al., 2021) instead of the self-trained image feature extractor in our fusion encoder.

4. Three principles for self-made datasets

5. Analyse the starting structure of the private datasets and the word frequency to show that the private dataset is reasonable to the engage question

6. baseline model:
(1) For the end-to-end baselines, we chose the VLT5 model (Cho et al., 2021) as the backbone

7. Here we seek to determine whether the most relevant image can represent the entire image sequence, as questions can focus on only one certain event or object by calculating the CLIP scores. This suggests that other images in the sequence assist in the reconstruction of missing information and even leave room for more imagination

### evaluation：

1. Evaluate the quality of problems in the dataset from the average length of the problem, syntactic complexity, etc
2. To the generated question, set private engaging benchmark by the author ，engage degree from high to low，（but I think it is very unreasonable），Baseline is divided into four groups to evaluate separately, but in fact, the gap between each group is very small, and I think it is easy to produce biases，
3. We evaluate the baselines with BLEU (Papineni et al., 2002), METEOR (Banerjee and Lavie, 2005), and BLEURT (Sellam et al., 2020).


## Let's Talk! Striking Up Conversations via Conversational Visual Question Generation

### Keypoints：

1. 总共分成两部分，先生成故事，再从故事生成问题
2. 生成故事用到了知识图谱增强，问题生成用的是transformer fine-tuned
(1) 第一阶段，知识图谱增强生成故事，input首先是image term sets 知识图谱告诉所有的term之间的relations, KG-Story uses a RNN-based language model to obtain a relation with lowest perplexity. The chosen relation is inserted to original term sequence expanding the number of term sets from 5 to 6. Then, KG-Story leverages Transformer (Vaswani et al. 2017) with expanded term sets from Stage 2 as input to generate story.
(2) 第二阶段，T5 as pre-trained model，fine-tuned on SQuAD as output

### evaluation：

1. baselines: Generating Natural Questions About an Image

# Question Generation based Question types

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

# Current Evaluation of Visual Storytelling  

## RoViST: Learning Robust Metrics for Visual Storytelling(之后还需要细看每个evaluation怎么评分的）

代码：https://github.com/usydnlp/rovist

数据集：StoryTelling

### 要点：

1. Nowadays, the past works on VST have used existing popular n-gram based metrics such as BLEU, METEOR, ROUGE, CIDEr, and SPICE to evaluate their models, such metrics based on n-gram matching tend to have poor correlation with human evaluation scores and do not explicitly consider other criteria necessary for storytelling such as sentence structure or topic coherence. In addition, these metrics mentioned so far still heavily rely on similarity with references, potentially leading to bias for VST tasks as the references may not fully cover the possible ways to write a story for an image sequence

2. Thus we propose several unreferenced metrics for the VST task based on the three aforementioned criteria: 1) visual grounding, 2) coherence, and 3) non-redundancy

3. i. visual ground: focus on norn, calculated by cos similiarity between text embedding and image embedding, multiplied by inverse document frequency 

   ii. Coherence Scorer: calculated by the softmax function of pooled 1024dimensional vector representation hn, but what is pn? 
   
   iii. Non-redundancy Scorer:we propose calculating the Jaccard Similarity (JS) between and within sentences. The JS is defined as the intersection size divided by the union size of two sets. The result is a score between 0 and 1 where a value closer to 1 means that the story tends to contain less redundancy

## What Makes A Good Story? Designing Composite Rewards for Visual Storytelling

### datasets: VIST

### baselines: (1) AREL (Wang et al. 2018b) (2) HSRL (Huang et al. 2019)

### codes:

### keypoints:

1. Motivation: Notably, current studies did not directly (or explicitly) examine what accounts for a good story to the human eye, which is the main focus of our work.

2. we propose three assessment criteria: relevance, coherence and expressiveness. We further propose a reinforcement learning framework, ReCo-RL, with reward functions designed to capture the essence of these quality criteria. In addition, we find that simply optimizing on standard automatic evaluation metrics may even hurt the performance of story generation according to other assessments that are more important to the human eye.

3. Dimension explaination: 1) Relevance: telling a story that accurately describes the objects and the concepts that appear in the photos. 2) Coherence: the consecutive sentences should be semantically and logically coherent with each other, instead of being mutually-independent sentences describing each photo separately. 3) Expressiveness: describe the visual scenes and actions in the photos, the language used for creating the story should contain a rich vocabulary and diverse style

4. The first relevance function gives a high reward to a generated description that mentions fine-grained concepts in an image. The second coherence function measures the fluency of a generated sentence given its preceding sentence, using a pre-trained language model. The third expressiveness function penalizes phrasal overlap between a generated sentence and its preceding sentences.

5. Model structure: Encoder(CNN), manager(RNN), worker(LSTM).

**idea: I notice that most storytelling are ignore the conjunction, could not connect various events relations such as causal inference etc. maybe it is a gap. For our propose, could we propose a question like why.... after?** 

## Learning to Rank Visual Stories from Human Ranking Data

### codes: https://github.com/academiasinicanlplab/vhed

### datasets:

### keypoints:

1. motivation: It remains unclear whether conventional automatic evaluation metrics for text generation are applicable on VIST. In addition, they show that the results from commonly adopted automatic metrics for text generation have little correlation with those obtained from human evaluation, which motivates us to directly utilize human evaluation results to learn the automatic evaluation model. In other words, existing evaluation methods are unable to capture the true quality of the generated stories.

2. Rethinking this postulation in evaluation, we believe the dependence on references should be minimized and human evaluation results should be fully utilized instead, because human judgements contain more meaningful signals.
 
   
# Current Visual Question Generation
 
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

### Conclusion: 

We presented a guided approach to visual question generation (VQG), which allows for the generation of questions that focus on specific chosen aspects of the input image. We introduced three variants for this task, the explicit, implicit, and variational implicit. The former generates questions based on an explicit answer category and a set of concepts from the image. In contrast, the latter two discretely predict these concepts internally, receiving only the image as input. The explicit model achieves SoTA results when evaluated against comparable models.

# Current Methogology on Visual-Language Tasks or Visual Question Generation/Storytelling

# Pre-trained Language Model

## An Empirical Study of Training End-to-End Vision-and-Language Transformers

### Keypoint:

Overview of Vision-Language Model: Given a text sentence l and an image v, a VLP model first extracts both text features l = ⟨l1, · · · , lN ⟩ and visual features v = ⟨v1, · · · , vM ⟩ via a text encoder and a vision encoder. The text and visual features are then fed into a multimodal fusion module to produce cross-modal representations, which are then optionally fed into a decoder before generating the final outputs.

1. We present METER, a Multimodal End-to-end TransformER framework, through which we thoroughly investigate how to design and pre-train a fully transformer based VLP model in an end-to-end manner.

2. They seperate vision-language tasks as several parts: vision ecoder, language encoder and fusion model, with only encoder or encoder-decoder structure. They analysis that: 1) Vision transformer (ViT) plays a more vital role than language transformer in VLP, and the performance of transformers on pure vision or language tasks is not a good indicator for its performance on VL tasks. 2) The inclusion of cross-attention benefits multimodal fusion, which results in better downstream performance than using self-attention alone. 3) Under a fair comparison setup, the encoder-only VLP model performs better than the encoder-decoder model for VQA and zero-shot image-text retrieval tasks. 4) Adding the masked image modeling loss in VLP will not improve downstream task performance in our settings

3. Drawback of current vision model: 1)Object detector feature extractor:such as ViLBERT, LXMERT, VisualBERT, VL-BERT, UNITER, OSCAR and VinVL, however, extracting region features can be timeconsuming, and the pre-trained ODs are usually frozen during pre-training, which limits the capacity of VLP models 2) CNN-based Grid Features: such as PixelBERT, CLIP-ViL, SOHO, although it is efficient,  inconsistent optimizers are typically used for CNN and transformer, and CNN can achieve slightly worse accuracy/FLOPs trade-offs than their ViT counterparter.

4. for the text encoder: the fusion module is typically initialized with BERT, and the role of text encoding and multimodal fusion is therefore entangled and absorbed in a single BERT model. Here, we aim to disentangle the two modules, and use a text encoder first before sending the features into the fusion module.

5. fusion part: Merged attention and co-attention. Figure 2 of the paper. For region-based VLP models, as shown, the merged attention and co-attention models can achieve comparable performance. Yet, the merged attention module is more parameter-efficient, as the same set of parameters are used for both modalities.

6. 1)Masked Language Modeling: Specifically, given an image-caption pair, we randomly mask some of the input tokens, and the model is trained to reconstruct the original tokens given the masked tokens lmask and its corresponding visual input v. 2) Masked image Modeling: Formally, given a sequence of visual features v = ⟨v1, · · · , vM ⟩, where vi is typically a region feature, we randomly mask some of the visual features, and the model outputs the reconstructed visual features ov given the rest of the visual features and the unmasked tokens t, and regression aims to minimize the mean squared error loss. the author use patch to do the task of MIM, but what the patch actually mean?

## Unifying Vision-and-Language Tasks via Text Generation

## Learning Transferable Visual Models From Natural Language Supervision(CLIP)

### keypoints:

1. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet.

2. To test this we constructed a new dataset of 400 million (image, text) pairs collected form a variety of publicly available sources on the Internet(WebImageText) 

3. While standard image models jointly train an image feature extractor and a linear classifier to predict some label, CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training examples. 混合text/image encoder

4. image encoder: (1) ResNet50 (2) ViT 
   text encoder: transformer
   
5. CLIP is pre-trained to predict if an image and a text snippet are paired together in WIT.

6. To do this, CLIP learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of the N 2 − N incorrect pairings

# Knowledge Enhance Methods

## Knowledge-Enriched Visual Storytelling

### datasets: 

VIST Dataset provides image-to-term materials for learning in Stage 1. For Stage 2, the object relations from Visual Genome or the term relations from OpenIE are the materials for Stage 2. ROCStories Corpora supplies a large quantity of pure textual stories for generation in Stage 3, and the VIST Dataset, the sole end-to-end visual storytelling dataset, is used to fine-tune the model.

### Key points:

1. KG-Story distills a set of representative words from the input prompts, enriches the word set by using external knowledge graphs, and finally generates stories based on the enriched word set. This distill-enrich-generate framework allows the use of external resources not only for the enrichment phase, but also for the distillation and generation phases.

2. motivation: However, existing visual storytelling approaches produce monotonous stories with repetitive text and low lexical diversity.

3. Notice: they said However, each such caption is a single story-like sentence and is independent of other captions; combined, the captions do not constitute a context-coherent story.

4. Stages: 1. Word distillation from input prompts/images. 2.Word enrichment using knowledge graphs 3.Story generation. We use a Transformer architecture to transform term paths into stories.

5. Stage 1: pre-trained faster R-CNN, To reduce computational complexity, only the object features within the top 25 confidence scores are used. Main difference:In contrast to the positional encoding in the original setting, object features are summed with trainable image-order embeddings as input, as objects in the same image are sequentially independent.

  Stage 2: we introduce semantic terms as the intermediate and link terms in two adjacent images using the relations provided by the knowledge graph. we pair the terms from two consecutive images and query the knowledge graph for all possible tuples and also consider the two-hop. Here the knowledge graph serves as the source of ideas connecting two images and the language model then ensures the coherence of the generated story when using the selected idea
  
  Stage 3: Generate story by transformer. We add to the original Transformer model three different modifications: (i) lengthdifference positional encoding for variable-length story generation, (ii) anaphoric expression generation for the unification of anaphor representation, and (iii) a repetition penalty for removing redundancy under beam search.
  
  ### evaluation:
  
  1. human rank and Observing Automatic Metrics: BLEU1 BLEU4 METEOR ROUGE CIDEr

## Commonsense Knowledge Aware Concept Selection for Diverse and Informative Visual Storytelling

### code： https://github.com/sairin1202/Commonsense-Knowledge-Aware-Concept-Selection-For-Diverse-and-Informative-Visual-Storytelling

### datasets

### baseline: INet, KS, KG-Story, Image+Bart

### key point:

1. We propose to foster the diversity and informativeness of a generated story by using a concept selection module that suggests a set of concept candidates. In other words, we aim to improve the concept selection for increasing the diversity and informativeness of VST.

2. motivation: it is shown that the stories tend to be monotonous which contains limited lexical diversity and knowledge. However, their method directly generates concepts from the images using sequenceto-sequence models. Since the concept is selected from the full vocabulary, this kind of direct generation often produces concepts of low quality which affects the informativeness of the story.

3. Totally the first stage:two novel modules SSM and MCSM to select concepts from the given candidates concepts under a plan-write two-stage visual storytelling system. The second steps: modified BART as our story generation module to mitigate the problem caused by limited vocabulary and knowledge in the dataset.

4. How to select: 1. We send the images into ResNet152 (He et al. 2016) to obtain image features 2.we use clarifai1 to obtain the top 10 seed concepts from each image. Each concept is used as a query to select relative commonsense concepts in the ConceptNet and we make several rules to filter some concepts which are less useful 3.To incorporate the visual information into the concepts, we also connect the image feature to its corresponding concept features in the graph.  4. to select the concept, from SSM model, updated concept features into the encoder, and the decoder will output the selected concepts, finally the concept with the highest probability is selected as the output concept, while its feature is directly copied for the generation of the next step. From MCSM model, this method aims to calculate the co-occurrence probability of all candidate concepts cs in the graph.

5. how to generate story: 1) a simple encoder-decoder module that uses multi-head pooling to encode the concept embeddings, and decode the sentences with a RNN decoder. 2) a large scale encoder-decoder which both can encode the input and output the sentences(bart).

### evaluation:

1. automatic: We report BLEU (B), METEOR (M), ROUGH-L (R), and CIDEr (C) scores.
2. human evaluation: ranking

# Interesting Paper Related Causal Question

## CausalQA: A Benchmark for Causal Question Answering

Code：https://github.com/webis-de/coling-22

Datasets：Webis-CausalQA-22

### Keypoints：

1. Since neither SQuAD nor other large QA benchmarks explicitly label causal questions, the difference in effectiveness between causal and other questions remains unclear, We distinguish different types of causal questions using a novel typology derived from a data-driven, manual analysis of questions from ten large question answering (QA) datasets.

2. we manually analyzed samples and developed a two-dimensional typology of causal questions based on their semantic properties and pragmatic interpretation

3. To category the causal questiom:At the semantic level, we group causal question types in terms of which component of a causal chain a question addresses.  In two dimension: (1)Semantic Dimension: questions about an antecedent & questions about a consequent & questions about the chain （2）The Pragmatic Dimension: solution seeking, knowledge seeking, and opinion seeking.

### Evaluation:

1. ROUGE-L, recall, and F1 and traditional exact match (EM) and F1 measures




