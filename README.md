# Document Understanding

Companies often face the problem of getting to information that’s locked away inside of textual documents. Within certain sectors, this problem is magnified by the fact that there are often millions of documents and the information inside them is often business critical.

## Solution Contents

In this solution, we look at a number of different techniques that can be used to understand large amounts of text.

![flow-chart](https://sagemaker-solutions-prod-us-east-2.s3.us-east-2.amazonaws.com/Document-understanding/docs/flow_chart.png)

Using the following document as an example, we'll show the different outputs at each stage.

```
Documents are a primary tool for communication,
collaboration, record keeping, and transactions across industries,
including financial, medical, legal, and real estate. The format of data
can pose an extra challenge in data extraction, especially if the content
is typed, handwritten, or embedded in a form or table. Furthermore,
extracting data from your documents is manual, error-prone, time-consuming,
expensive, and does not scale. Amazon Textract is a machine learning (ML)
service that extracts printed text and other data from documents as well as
tables and forms. We’re pleased to announce two new features for Amazon
Textract: support for handwriting in English documents, and expanding
language support for extracting printed text from documents typed in
Spanish, Portuguese, French, German, and Italian. Many documents, such as
medical intake forms or employment applications, contain both handwritten
and printed text. The ability to extract text and handwriting has been a
need our customers have asked us for. Amazon Textract can now extract
printed text and handwriting from documents written in English with high
confidence scores, whether it’s free-form text or text embedded in tables
and forms. Documents can also contain a mix of typed text or handwritten
text. The following image shows an example input document containing a mix
of typed and handwritten text, and its converted output document. You can
log in to the Amazon Textract console to test out the handwriting feature,
or check out the new demo by Amazon Machine Learning Hero Mike Chambers.
Not only can you upload documents with both printed text and handwriting,
you can also use Amazon Augmented AI (Amazon A2I), which makes it easy to
build workflows for a human review of the ML predictions. Adding in Amazon
A2I can help you get to market faster by having your employees or AWS
Marketplace contractors review the Amazon Textract output for sensitive
workloads. For more information about implementing a human review, see
Using Amazon Textract with Amazon Augmented AI for processing critical
documents. If you want to use one of our AWS Partners, take a look at how
Quantiphi is using handwriting recognition for their customers.
Additionally, we’re pleased to announce our language expansion. Customers
can now extract and process documents in more languages. Amazon Textract
now supports processing printed documents in Spanish, German, Italian,
French, and Portuguese. You can send documents in these languages,
including forms and tables, for data and text extraction, and Amazon
Textract automatically detects and extracts the information for you. You
can simply upload the documents on the Amazon Textract console or send them
using either the AWS Command Line Interface (AWS CLI) or AWS SDKs.
```

### Summarization

We start this solution by trying to understand documents from a high level. Summarization is useful when you want to distill the information found in a large amount of text down to a few sentences. We'll be using an 'extractive' summarization method in this solution, that extracts the most important sentences from the document verbatim. We don't cover 'abstractive' summarization here, because it's a lot more challenging and error prone to generate new sentences that summarize the document.

An example of extractive summarization on the document above is as follows:

```
Amazon Textract is a machine learning (ML) service that extracts printed text and other data from documents as well as tables and forms . many documents, such as medical intake forms or employment applications, contain both handwritten and printed text . new features include support for handwriting in English documents, and expanding language support for extracting printed text from documents in Spanish, Portuguese, French, German, and Italian.
```

### Question Answering

Up next we look at a technique that can be used to query the document for specifics, called Question Answering. Question Answering is useful when you want to query a large amount of text for specific information. Maybe you're interested in extracting the date a certain event happened. You can construct a question (or query) in natural language to retrive this information: e.g. 'When did Company X release Product Y?". Similar to extractive summarization we saw in the last notebook, Question Answering will return a verbatim slice of the text as the answer. It won't generate new words to answer the question.

An example of question answering on the document above with the question `What languages did Amazon Textract add support for?` is as follows:

```
{
  'score': 0.9566398820782922,
  'start': 768,
  'end': 817,
  'answer': 'Spanish, Portuguese, French, German, and Italian.'
}
```

### Entity Recognition

We then look at a technique that can be used to extract the key entities from a document, called Entity Recognition. As part of this stage we extract noun chunks and named entities which are also classified into a number of different types (e.g people, places, organizations, etc). Often many more noun chunks are found than named entities.

An example of entity recognition on the document above is as follows:

```
{
  'entities': [
    {
      'text': 'Amazon',
      'start_char': 622,
      'end_char': 628,
      'label': 'ORG'
    },
    {
      'text': 'English',
      'start_char': 666,
      'end_char': 673,
      'label': 'LANGUAGE'
    },
  ...
    {
      'text': 'Mike Chambers',
      'start_char': 1561,
      'end_char': 1574,
      'label': 'PERSON'
    }
  ],
  'noun_chunks': [
    {
      'text': ' Documents',
      'start_char': 0,
      'end_char': 10
    },
    {
      'text': 'a primary tool',
      'start_char': 15,
      'end_char': 29
    },
    ...
    {
      'text': 'AWS SDKs',
      'start_char': 2781,
      'end_char': 2789
    }
  ]
}
```

### Relationship Extraction

Once you have determined the entities of interest, you can use a technique called relationship extraction to determine the relationships between two given entities. Using the SemEval 2010 Task 8 dataset for training the model, we can extract 9 relationship types and infer the direction of the relationship. You can extract the following relationship types with the demo dataset and can provide your own labelled data to extract other relationship types.

* Cause-Effect
* Instrument-Agency
* Product-Producer
* Content-Container
* Entity-Origin
* Entity-Destination
* Component-Whole
* Member-Collection
* Message-Topic
* Other

An example of relationship extraction on a snippet is as follows:

```
# input
{
  'sequence': 'I have imported an audio book into the software',
  'entity_one_start': 25,
  'entity_one_end': 29,
  'entity_two_start': 39,
  'entity_two_end': 47
}
```

```
# output
{
  'id': 5,
  'str': "Entity-Destination(e1,e2)"
}
```

## Getting Started

You will need an AWS account to use this solution. Sign up for an account [here](https://aws.amazon.com/).

To run this JumpStart 1P Solution and have the infrastructure deploy to your AWS account you will need to create an active SageMaker Studio instance (see [Onboard to Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html)). When your Studio instance is *Ready*, use the instructions in [SageMaker JumpStart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html) to 1-Click Launch the solution.

The solution artifacts are included in this GitHub repository for reference.

*Note*: Solutions are available in most regions including us-west-2, and us-east-1.

**Caution**: Cloning this GitHub repository and running the code manually could lead to unexpected issues! Use the AWS CloudFormation template. You'll get an Amazon SageMaker Notebook instance that's been correctly setup and configured to access the other resources in the solution.

## Architecture

We focus on Amazon SageMaker components in this solution. Amazon SageMaker Training Jobs are used to train the relationship extraction model and Amazon SageMaker Endpoints are used to deploy the models in each stage. We use Amazon S3 alongside Amazon SageMaker to store the training data and model artifacts, and Amazon CloudWatch to log training and endpoint outputs.

![architecture-diagram](https://sagemaker-solutions-prod-us-west-2.s3.us-west-2.amazonaws.com/Document-understanding/docs/architecture_diagram_light.png)

## Credits

* Packages
  * [Hugging Face Transformers](https://huggingface.co/)
  * [Spacy](https://spacy.io/)
* Datasets
  * SemEval 2010 Task 8 (use to train relationship extraction model)
    * Iris Hendrickx, Su Nam Kim, Zornitsa Kozareva, Preslav Nakov, Diarmuid Ó Séaghdha, Sebastian Padó, Marco Pennacchiotti, Lorenza Romano and Stan Szpakowicz
* Models
  * T5
    * Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu
  * BERT
    * Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
  * Matching the Blanks
    * Livio Baldini Soares, Nicholas FitzGerald, Jeffrey Ling, Tom Kwiatkowski

## License

This project is licensed under the Apache-2.0 License.
