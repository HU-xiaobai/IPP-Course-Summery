# IPP-Course-Summery

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
