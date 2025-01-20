# Multilingual-spacy-resume-analyser
Spacy Resume Analyser: An NLP-based tool using spaCy to categorize resumes, analyze skills, generate word clouds, and compare job-specific requirements.

#### Introduction

In this project, we are going to use spacy for entity recognition on 1000 Resume and experiment around various NLP tools for text analysis. The main purpose of this project is to help recruiters go throwing hundreds of applications within a few minutes. We have also added skills match feature so that hiring managers can follow a metric that will help them to decide whether they should move to the interview stage or not. We will be using two datasets; the first contains resume texts and the second contains skills that we will use to create an entity ruler
#### Datset
A collection of 2400+ Resume Examples taken from livecareer.com for categorizing a given resume into any of the labels defined in the dataset: Resume Dataset.
Inside the CSV

ID: Unique identifier and file name for the respective pdf.
    Resume_str : Contains the resume text only in string format.
    Resume_html : Contains the resume data in html format as present while web scrapping.
    Category : Category of the job the resume was used to apply.

#### Present categories
HR, Designer, Information-Technology, Teacher, Advocate, Business-Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital-Media, Automobile, Chef, Finance, Apparel, Engineering, Accountant, Construction, Public-Relations, Banking, Arts, Aviation

**Resume DATSET Loading**
Using Pandas read_csv to read dataset containing text data about Resume.

 
we are going to randomized Job categories so that 1000 samples contain various job categories instead of one.
 we are going to limit our number of samples to 1000 as processing 2400+ takes time.
 
 SPacy model load
in the terminal type `python -m download spacy en_core_web_lg` 


**1. What is EntityRuler?**
The EntityRuler in spaCy is a pipeline component that allows you to define rules to identify specific entities in text. It supplements the named entity recognition (NER) system by allowing you to highlight custom entities (e.g., skills, job roles) that the pre-trained spaCy models might not recognize out of the box

**Entity Ruler**
To create an entity ruler we need to add a pipeline and then load the .jsonl file containing skills into ruler. As you can see we have successfully added a new pipeline entity_ruler. Entity ruler helps us add additional rules to highlight various categories within the text, such as skills and job description in our case.

```
ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_pattern_path)
nlp.pipe_names
```

1. tok2vec
Meaning: Token-to-vector.
Purpose: This component converts individual tokens (words, punctuation, etc.) into dense vector representations (numerical embeddings).
Why it’s important: These vectors capture the semantic meaning of tokens and are used by downstream components like the tagger, parser, and ner.
How it works: It uses machine learning models to map tokens into a high-dimensional space where similar words have similar vectors.
2. tagger
Meaning: Part-of-speech (POS) tagger.
Purpose: Assigns a part-of-speech tag (e.g., noun, verb, adjective) to each token in the text.
Why it’s important: POS tags are essential for understanding the grammatical structure of a sentence and for downstream tasks like parsing or named entity recognition.
Example:
python
Copy
Edit
"Strawberries are sweet."
"Strawberries" → Noun
"are" → Verb
"sweet" → Adjective
3. parser
Meaning: Dependency parser.
Purpose: Analyzes the grammatical relationships between words in a sentence.
Why it’s important: It identifies how words are related to one another, such as which word is the subject, object, or modifier.
Example:
python
Copy
Edit
"I love strawberries."
"I" → Subject
"love" → Verb (root of the sentence)
"strawberries" → Object
4. attribute_ruler
Meaning: Attribute modification and standardization.
Purpose: Modifies token attributes based on predefined rules, such as normalizing casing or correcting token text. It is often used for lemmatization (described below) or customizing spaCy’s behavior.
Why it’s important: Helps in fine-tuning token-level attributes for consistency across processing tasks.
Example:
Converts all uppercase tokens like "PYTHON" to "Python".
Normalizes "ain't" to "isn't".
5. lemmatizer
Meaning: Lemmatization tool.
Purpose: Reduces words to their base or dictionary form (lemma).
Why it’s important: Lemmatization helps in text normalization and simplifies further processing by treating words with the same root as equivalent.
Example:
"running" → "run"
"better" → "good"
6. ner
Meaning: Named Entity Recognizer.
Purpose: Identifies and categorizes named entities in the text, such as people, organizations, locations, dates, or other specific entities.
Why it’s important: Extracts valuable structured information from unstructured text.
Example:
python
Copy
Edit
"Elon Musk is the CEO of SpaceX."
"Elon Musk" → PERSON
"SpaceX" → ORG
7. entity_ruler
Meaning: Rule-based named entity recognition.
Purpose: Allows you to define custom patterns for identifying entities that might not be covered by the ner component.
Why it’s important: Extends the ner functionality with specific, user-defined rules for entity recognition.
Example:
Custom rule: Recognize "Python" as a SKILL.
Example of a Pipeline in Action:
Consider the pipeline you shared:

['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner', 'entity_ruler']
Here’s how text might flow through the pipeline:

tok2vec: Tokens are converted into dense vectors.
tagger: Each token is assigned a part-of-speech tag.
parser: Grammatical relationships between tokens are analyzed.
attribute_ruler: Token attributes are standardized or modified.
lemmatizer: Words are reduced to their base forms.
ner: Named entities are identified and categorized.
entity_ruler: Custom rules are applied to highlight additional entities

**Skills**

We will create two python functions to extract all the skills within a resume and create an array containing all the skills. Later we are going to apply this function to our dataset and create a new feature called skill. This will help us visualize trends and patterns within the dataset.

   
get_skills is going to extract skills from a single text.
    unique_skills will remove duplicates.


```
def get_skills(text):
    doc = nlp(text)
    myset = []
    subset = []
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            subset.append(ent.text)
    myset.append(subset)
    return subset


def unique_skills(x):
    return list(set(x))
```

Cleaning Resume Text

We are going to use nltk library to clean our dataset in a few steps:


We are going to use regex to remove hyperlinks, special characters, or punctuations.
- 		Lowering text
-     Splitting text into array based on space
-     Lemmatizing text to its base form for 		normalizations
-     Removing English stopwords
-     Appending the results into an array.

As we can observe INFORMATION-TECHNOLOGY job category's skills distributions.

**Top Skills**

- Software
-  Support
-  Business

If you are looking to improve your chance of getting hired by 

here we can select options like AVIATION AND INFORMATION-TECHNOLOGY to get top skills of the job category


### Find Most Used Words
Most used words
In this part, we are going to display the most used words in the Resume filter by job category. In Information technology, the most words used are system, network, and database. We can also discover more patterns by exploring the word cloud below.




