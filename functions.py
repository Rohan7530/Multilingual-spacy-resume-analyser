#spacy
import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc

#gensim
import gensim
from gensim import corpora

#Visualization
from spacy import displacy
import pyLDAvis.gensim_models
from wordcloud import WordCloud
import plotly.express as px
import matplotlib.pyplot as plt

#Data loading/ Data manipulation
import pandas as pd
import numpy as np
import jsonlines

#nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['stopwords','wordnet'])

#warning
import warnings 
warnings.filterwarnings('ignore')

#here we will feed the entity ruler and also load the spacy model
df = pd.read_csv("resumeScraper/Resume.csv")
df = df.reindex(np.random.permutation(df.index))
data = df.copy().iloc[
    0:100,
]
print(data.head())


#loading the spacy model
nlp = spacy.load("en_core_web_lg")
skill_pattern_path = "jz_skill_patterns.jsonl"

print("model loaded successfully")

#adding entity ruler to the pipeline
ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_pattern_path) #here it is getting the patterns from the jsonl file to load the entity ruler
print(nlp.pipe_names)

#adding two functions to get the skills
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

#using nltk to clean the dataset
clean = []
for i in range(data.shape[0]):
    review = re.sub(
        '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
        " ",
        data["Resume_str"].iloc[i],
    )
    review = review.lower()
    review = review.split()
    lm = WordNetLemmatizer()
    review = [
        lm.lemmatize(word)
        for word in review
        if not word in set(stopwords.words("english"))
    ]
    review = " ".join(review)
    clean.append(review)

#this is the code for applying functions to the clean array which contains word related to the entity ruler
# 1. create a column called "clean resume" and add the clean keywords in it as par the ner
# 2. apply get skills function to the clean array to filterout words that contain the skills only
# 3. only keep the unique skills in the "clean resume" column

data["Clean_Resume"] = clean
data["skills"] = data["Clean_Resume"].str.lower().apply(get_skills)
data["skills"] = data["skills"].apply(unique_skills)
print(data.head())

fig = px.histogram(
    data, x="Category", title="Distribution of Jobs Categories"
).update_xaxes(categoryorder="total descending")
fig.show()

#creating a variable called job_cat to search regarding the job category
Job_cat = data["Category"].unique()
Job_cat = np.append(Job_cat, "ALL")

#this si the graph that shows skills in a single job category
Job_Category = input("Enter job category (or 'ALL' for all categories): ")

# Total_skills = []
# if Job_Category != "ALL":
#     fltr = data[data["Category"] == Job_Category]["skills"]
#     for x in fltr:
#         for i in x:
#             Total_skills.append(i)
# else:
#     fltr = data["skills"]
#     for x in fltr:
#         for i in x:
#             Total_skills.append(i)

# fig = px.histogram(
#     x=Total_skills,
#     labels={"x": "Skills"},
#     title=f"{Job_Category} Distribution of Skills",
# ).update_xaxes(categoryorder="total descending")
# fig.show() 
# #uncomment the above code for the skills of job category defined

# here we will build a wordcloud, which means the most words used in the job category
text = ""
for i in data[data["Category"] == Job_Category]["Clean_Resume"].values:
    text += i + " "
# print(data["Category"].unique()) #UNIT TEST PASSED
# print(len(data[data["Category"] == Job_Category])) #UNIT TEST PASSED
# print(text) #UNIT TEST PASSED

plt.figure(figsize=(8, 8))

x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

wc = WordCloud(
    width=900,
    height=800,
    background_color="white",
    min_font_size=6,
    repeat=True,
    mask=mask,
)
wc.generate(text)

plt.axis("off")
# plt.imshow(wc, interpolation="bilinear")
plt.imshow(wc)

plt.title(f"Most Used Words in {Job_Category} Resume", fontsize=15)
plt.savefig("wordcloud.png")
plt.show()


# Process the text from the dataset, named entity recognition
resume_text = data["Resume_str"].iloc[0]
doc = nlp(resume_text)

# Save highlighted entities to a .txt file
with open("highlighted_resume.txt", "w", encoding="utf-8") as file:
    # Generate HTML with entity highlighting
    html = displacy.render(doc, style="ent")
    # Write the HTML to a file
    file.write(html)

#Now we will do the dependency parsing of the text, where we can predit the relationship between the words
displacy.render(sent[0:10], style="dep", options={"distance": 90})

