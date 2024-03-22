__author__ = 'leifos'
import math
from simiir.text_classifiers.base_classifier import BaseTextClassifier
from simiir.utils.tidy import clean_html
from simiir.utils.decorators import retry
import logging
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.output_parsers import StructuredOutputParser

log = logging.getLogger('llm_classifer.LLMTextClassifier')


class LLMTextClassifier(BaseTextClassifier):
    """

    """
    def __init__(self, topic, search_context, llmodel='llama2'):
        """

        """
        super(LLMTextClassifier, self).__init__(topic, search_context)
        self.updating = False

        self._template = """ 
        You are a journalist assessing the relevance of news articles for the following topic.
        Topic Description: "{topic_description}"
       
        Document Title: "{doc_title}"
        Document Contents: "{doc_content}"
        
        Judge whether the document is relevant given the topic desciption.
        {format_instructions}
        """
        self._topic_schema = ResponseSchema(
            name="topic",
            type='bool',
            description="Is the document about the subject matter in the topic description? \
                Answer True if about the topic in the description, else False"
            )
       

        self._explanation_schema = ResponseSchema(
            name="explain",
            description="Summarize the relevant information contained in \
              the document that meets the criteria in the topic description."
            )
       
        self._recommendation_schema = ResponseSchema(
            name="relevant",
            type='bool',
            description="Is the document relevant to the topic description? \
                Answer True if relevant, False if not relevant or unknown."
            )
        
        if llmodel.lower() == 'openai':
            self._llm = ChatOpenAI(temperature=0.0)
        else:
            self._llm = ChatOllama(model=llmodel)
        
        self._output_parser = StructuredOutputParser.from_response_schemas([ self._topic_schema, self._explanation_schema , self._recommendation_schema ])

        format_instructions = self._output_parser.get_format_instructions()
        print(format_instructions)
     
        self._prompt = PromptTemplate(
            template=self._template,
            input_variables=["topic_title", "topic_description", "doc_title", "doc_content"],
            partial_variables={"format_instructions": format_instructions})


    def update_model(self, search_context):
        """
        If updating is enabled, updates the underlying language model with the new snippet/document text.
        Returns True iif the language model is updated; False otherwise.

        When self.update_method==1, documents are considered; else snippets.
        """
        if self.updating:
            ## Once we develop more update methods, it is probably worth making this a strategy
            ## so that setting the update_method changes the list of documents to use.
            if self.update_method == 1:
                document_list = search_context.get_all_examined_documents()
            else:
                document_list = search_context.get_all_examined_snippets()

            # iterate through document_list, pull out relevant snippets / text
            rel_text_list = []
            for doc in document_list:
                if doc.judgment > 0:
                    rel_text_list.append('{0} {1}'.format(doc.title, doc.content))
            if rel_text_list:
                ##
                ## use relevant docs as examples for few shot ??
                return True
            else:
                return False

        return False


    @retry(max_retries=5, wait_time=1)
    def is_relevant(self, document):
        """

        """        
        doc_title = " ".join(clean_html(document.title))
        doc_content = " ".join(clean_html(document.content))
        topic_title = self._topic.title
        topic_description  = self._topic.content

        log.debug(self._prompt.format(topic_title=topic_title, topic_description=topic_description, doc_title=doc_title, doc_content=doc_content))
        ###
        chain = self._prompt | self._llm | self._output_parser

        out = chain.invoke({ 'topic_title':topic_title, 'topic_description': topic_description, 'doc_title':doc_title, 'doc_content': doc_content })        
        
        #print(doc_title)
        #print(doc_content)
        log.debug(out)
        print(out)
        #print(type(out))

        rel = out.get('relevant', False)
        #print(rel)

        if rel:
            self._search_context.relevant_information_found.append(out.get('explain',''))

        return rel
    









