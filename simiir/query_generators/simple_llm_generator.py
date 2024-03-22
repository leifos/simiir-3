from ifind.common.query_ranker import QueryRanker
from ifind.common.query_generation import SingleQueryGeneration
from simiir.query_generators.base_generator import BaseQueryGenerator

from simiir.query_generators.single_term_generator import SingleTermQueryGenerator
from simiir.query_generators.smarter_generator import SmarterQueryGenerator

import random
import string

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.output_parsers import StructuredOutputParser

from tenacity import retry,wait_exponential,stop_after_attempt



class SimpleLLMGenerator(BaseQueryGenerator):
    """
    Takes the SmarterQueryGenerator, and interleaves it with guaranteed dud queries (e.g. [dud, smarter, dud, smarter...])
    Dud queries are generated as random strings, consisting of letters and numbers.
    """
    def __init__(self, stopword_file, background_file=[]):
        super(SimpleLLMGenerator, self).__init__(stopword_file, background_file=background_file, allow_similar=True)
        self.__smarter = SmarterQueryGenerator(stopword_file, background_file)
        self._template = """ 
        You are a journalist assessing the relevance of news articles for the following topic and need to generate 
        search queries to find as much relevant material as quickly as possible.

        Queries should be as diverse as possible and avoid repetition and no more than 3 query terms.
        Topic Title: "{topic_title}"
        Topic Description: "{topic_description}"
       
        {format_instructions}
        """
        self._query_schema = ResponseSchema(
            name="queries",
            description="Generate an array of 30 possible queries for the topic of less than or equal to 3 search terms ordered by likelihood of success.",
            type="array"
            )
        
        #self._llm = ChatOllama(model="llama2:70b")
        #self._llm = ChatOllama(model="mistral")
        
        self._llm = ChatOpenAI(temperature=0.0)

        self._output_parser = StructuredOutputParser.from_response_schemas([ self._query_schema])

        format_instructions = self._output_parser.get_format_instructions()
     
        self._prompt = PromptTemplate(
            template=self._template,
            input_variables=["topic_title", "topic_description"],
            partial_variables={"format_instructions": format_instructions})

    @retry(wait=wait_exponential(multiplier=1,min=1,max=5),stop=stop_after_attempt(10))
    def generate_query_list(self, search_context):
        """
        Given a Topic object, produces a list of query terms that could be issued by the simulated agent.
        """
        topic = search_context.topic.title
        description = search_context.topic.content

        chain = self._prompt | self._llm | self._output_parser
        out = chain.invoke({ 'topic_title':topic, 'topic_description': description})        
        queries = out.get('queries',[])
        final = [[str(q),1] for q in queries]
        return final