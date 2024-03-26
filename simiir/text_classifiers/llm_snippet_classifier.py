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
from langchain_core.output_parsers import JsonOutputParser



log = logging.getLogger('llm_classifer.LLMSnippetTextClassifier')

from tenacity import retry,wait_exponential,stop_after_attempt
class LLMSnippetTextClassifier(BaseTextClassifier):
    """

    """
    def __init__(self, topic, search_context, llmodel='llama2'):
        """

        """
        super(LLMSnippetTextClassifier, self).__init__(topic, search_context)
        self.updating = False

        self._template = """ 
        You are a journalist assessing a news search result page, where you are looking for 
        results that might be relevant to the following topic.
        Topic Description: "{topic_description}"
       
        —BEGIN RESULT SUMMARY—
        Result Title: 
        {doc_title}
        Result Snippet: 
         {doc_content}
        —END RESULT SUMMARY—

        Judge whether this result is likely to contain relevant information.
        {format_instructions}
        """
        self._topic_schema = ResponseSchema(
            name="topic",
            type='bool',
            description="Is the result about the subject matter in the topic description? \
                Answer True if about the topic in the description, else False"
            )
       
        self._recommendation_schema = ResponseSchema(
            name="click",
            type='bool',
            description="Is it worth clicking on this result to inspect the document? \
                Answer True if it is worth clicking, else False."
            )
        
        print(f'Using {llmodel.lower()}')
        if llmodel.lower() == 'openai':
            self._llm = ChatOpenAI(temperature=0.0)
        else:
            self._llm = ChatOllama(model=llmodel)
        
        self._output_parser = JsonOutputParser(pydantic=[ self._topic_schema, self._recommendation_schema ])

        format_instructions = self._output_parser.get_format_instructions()
        
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


    @retry(wait=wait_exponential(multiplier=1,min=1,max=5),stop=stop_after_attempt(10))
    def is_relevant(self, document):
        """

        """
        print("in snippet llm classifier")
        doc_title = " ".join(clean_html(document.title))
        doc_content = " ".join(clean_html(document.content))
        topic_title = self._topic.title
        topic_description  = self._topic.content

        log.debug(self._prompt.format(topic_title=topic_title, topic_description=topic_description, doc_title=doc_title, doc_content=doc_content))
        ###
        chain = self._prompt | self._llm | self._output_parser

        #print('About to invoke the chain')
        out = chain.invoke({ 'topic_title': topic_title, 'topic_description': topic_description, 'doc_title': doc_title, 'doc_content': doc_content })        
        
        log.debug(out)
        print(f'Snippet: {doc_title}\n{doc_content}'.strip())
        print(f'Snippet Decision: {out}')
        rel = out.get('click', False)

        return rel
    









