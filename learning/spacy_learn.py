import spacy
from spacy.tokens import Doc, Token, Span

# objects : Span, Doc

# load
nlp = spacy.load('en_core_web_sm')

introduction_text = 'This tutorial is about Natural Language Processing in Spacy.'

# process text
introduction_doc = nlp(introduction_text)

# tokenizing
for token in introduction_doc:
    print(token.text)

doc = nlp("This is a text")
span = doc[2:4]
print(span.text)
# 'a text'

# Create a Doc object
doc = nlp("I live in New York")
# Span for "New York" with label GPE (geopolitical)

# Linguistic features
#   Part-of-speech tags (predicted by statistical model)
doc = nlp("This is a text.")
# Coarse-grained part-of-speech tags
print([token.pos_ for token in doc])
# ['DET', 'VERB', 'DET', 'NOUN', 'PUNCT']
# Fine-grained part-of-speech tags
print([token.tag_ for token in doc])
# ['DT', 'VBZ', 'DT', 'NN', '.']


# Syntactic dependencies (predicted by statistical model)
doc = nlp("This is a text.")
# Dependency labels
print([token.dep_ for token in doc])
# ['nsubj', 'ROOT', 'det', 'attr', 'punct']
# Syntactic head token (governor)
print([token.head.text for token in doc])
# ['is', 'is', 'text', 'is', 'is']

# Named Entities (predicted by statistical model)
doc = nlp("Larry Page founded Google")
# Text and label of named entity span
print([(ent.text, ent.label_) for ent in doc.ents])
# [('Larry Page', 'PERSON'), ('Google', 'ORG')]


# Sentences (usually needs the dependency parser)
doc = nlp("This a sentence. This is another one.")
# doc.sents is a generator that yields sentence spans
print([sent.text for sent in doc.sents])
# ['This is a sentence.', 'This is another one.']
# Base noun phrases (needs the tagger and parser)
doc = nlp("I have a red car")
# doc.noun_chunks is a generator that yields spans
print([chunk.text for chunk in doc.noun_chunks])
# ['I', 'a red car']
# Label explanations
spacy.explain("RB")
# 'adverb'
spacy.explain("GPE")
# 'Countries, cities, states'

# Word vectors and similarity
doc1 = nlp("I like cats")
doc2 = nlp("I like dogs")
# Compare 2 documents
print(doc1.similarity(doc2))
# Compare 2 tokens
print(doc1[2].similarity(doc2[2]))
# Compare tokens and spans
print(doc1[0].similarity(doc2[1:3]))

# Vector as a numpy array
doc = nlp("I like cats")
# The L2 norm of the token's vector
print(doc[2].vector)
print(doc[2].vector_norm)

# Pipeline components

# Functions that take a Doc object, modify it and return it.
# text   =>   (tokenizer -> parser -> ner -> ...)   =>   doc
nlp = spacy.load("en_core_web_sm")
print(nlp.pipe_names)
# ['tagger', 'parser', 'ner']
print(nlp.pipeline)


# [('tagger', <spacy.pipeline.Tagger>),
# ('parser', <spacy.pipeline.DependencyParser>),
# ('ner', <spacy.pipeline.EntityRecognizer>)]

# Function that modifies the doc and returns it
def custom_component(doc):
    print("Do something to the doc here!")
    return doc


# Add the component first in the pipeline
nlp.add_pipe(custom_component, first=True)

from spacy.tokens import Doc, Token, Span

doc = nlp("The sky over New York is blue")

# Extension attributes
# Custom attributes that are registered on the global Doc, Token and Span classes and become available as ._.

doc = nlp("The sky over New York is blue")
# Attribute extensions (with default value)
# Register custom attribute on Token class
Token.set_extension("is_color", default=False)
# Overwrite extension attribute with default value
# doc[6]._.is_color = True
# Property extensions (with getter & setter)
# Register custom attribute on Doc class
get_reversed = lambda doc: doc.text[::-1]
Doc.set_extension("reversed", getter=get_reversed)
# Method extensions (callable method)
# Register custom attribute on Span class
has_label = lambda span, label: span.label_ == label
Span.set_extension("has_label", method=has_label)
# Compute value of extension attribute with method
doc[3:5].has_label("GPE")
# True


print('Done')
