from simiir.serp_impressions.base_serp_impression import BaseSERPImpression

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.output_parsers import StructuredOutputParser

from tenacity import retry,wait_exponential,stop_after_attempt

class LLMSerpImpression(BaseSERPImpression):
    """
    A simple approach to SERP impression judging.
    The de facto approach used in prior simulations; assume it's worth examining. Always return True.
    Also includes code to judge the patch type, a different thing from determining if the patch should be entered or not.
    """
    def __init__(self, search_context, qrel_file, host=None, port=None):
        super(LLMSerpImpression, self).__init__(search_context=search_context,
                                                 qrel_file=qrel_file,
                                                 host=host,
                                                 port=port)
        self._template = """ 
        You are a journalist assessing the relevance of news articles for the following topic:
        ```
        Topic Title: "{topic_title}"
        Topic Description: "{topic_description}"
        ```

        You are trying to determine if you should spend time viewing the current search engine result page with the following snippets:
        Current SERP:
        ```
        {new_snippets}
        ```

        You have previously found the following snippets, in ```, to be relevant:
        Previous snippets:
        ```
        {old_snippets}
        ```
        
        {format_instructions}
        """
        self._view_schema = ResponseSchema(
            name="view",
            description="Respond with True if you believe you should examine the current search engine page and False otherwise.",
            type="bool"
            )
        self._explain_schema = ResponseSchema(
            name="explain",
            description="Explain why you believe you should or should not examine the SERP.",
            type="string"
            )
        
        #self._llm = ChatOllama(model="llama2:70b")
        #self._llm = ChatOllama(model="mistral")
        
        self._llm = ChatOpenAI(temperature=0.0,verbose=True,model='gpt-4-turbo-preview')

        self._output_parser = StructuredOutputParser.from_response_schemas([ self._view_schema,self._explain_schema])

        format_instructions = self._output_parser.get_format_instructions(only_json=True)
        print(format_instructions)
     
        self._prompt = PromptTemplate(
            template=self._template,
            input_variables=["topic_title", "topic_description","old_snippets","new_snippets"],
            partial_variables={"format_instructions": format_instructions})
    
    
    def is_serp_attractive(self):
        """
        Determines whether the SERP is attractive.
        """
        topic = self._search_context.topic.title
        description = self._search_context.topic.content
        results_len = self._search_context.get_current_results_length()
        results_list = self._search_context.get_current_results()
        goto_depth = self.viewport_size
        
        if results_len < goto_depth:  # Sanity check -- what if the number of results is super small?
            goto_depth = results_len

        snippet_list = self._search_context.get_all_examined_snippets()
        old_snippets = []
        for snippet in snippet_list:
            if snippet.judgment > 0:
                old_snippets.append('Snippet: {0}\n'.format(snippet.title))
        
        new_snippets = []
        for i in range(0, goto_depth):
            snippet = "Snippet {0} {1}\n".format(i,results_list[i].title)
            new_snippets.append(snippet)
        
        chain = self._prompt | self._llm | self._output_parser
        out = chain.invoke({ 'topic_title':topic, 'topic_description': description,  'old_snippets':'\n'.join(old_snippets), 'new_snippets':'\n'.join(new_snippets)})        
        view = out.get('view',False)
        explain = out.get('explain',"")
        print(view,explain)
        return view