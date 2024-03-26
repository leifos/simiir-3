from simiir.search_contexts.search_context import SearchContext
from ifind.seeker.trec_qrel_handler import TrecQrelHandler
from simiir.utils.data_handlers import get_data_handler
import os


class TRECSearchContext(SearchContext):

    def __init__(self, search_interface, output_controller, topic, qrel_file):
        
        self._qrel_handler = TrecQrelHandler(qrel_file)
        super(TRECSearchContext, self).__init__(search_interface, output_controller, topic)

        #print(f'TRECSearchContext: {self.get_topic()} {self.get_topic().id}')

    def _assess_documents(self):
        rel_docs = self._relevant_documents

        rel_doc_list = [doc for doc in rel_docs]
        rel_doc_set = set(rel_doc_list)

        num_trec_rels = 0
        for doc in rel_doc_list:
            val = self._qrel_handler.get_value(self.get_topic().id, doc.doc_id)
            if val > 0:
                num_trec_rels += 1

        uniq_num_trec_rels = 0
        for doc in rel_doc_set:
            val = self._qrel_handler.get_value(self.get_topic().id, doc.doc_id)
            if val > 0:
                uniq_num_trec_rels += 1

        self._num_trec_rels = num_trec_rels 
        self._uniq_num_trec_rels = uniq_num_trec_rels

        #print(f'TRECSearchContext: {self.get_topic().id} num:{self._num_trec_rels}')


    def report(self):
        self._assess_documents()
        report_string = super(TRECSearchContext, self).report()
        user_precision = 0.0
        if len(self._relevant_documents) > 0:
            user_precision = round(self._num_trec_rels / len(self._relevant_documents),4)
        report_string = report_string + "    Number of TREC Relevant Docs: {0}{1}".format(self._num_trec_rels, os.linesep)
        report_string = report_string + "    Number of Unique TREC Relevant Docs: {0}{1}".format(self._uniq_num_trec_rels, os.linesep)
        report_string = report_string + "    Precision of Marked Docs: {0}{1}".format(user_precision, os.linesep)
        
        self._output_controller.log_info(info_type="TOTAL_TREC_RELEVANT_DOCS", text=self._num_trec_rels)
        self._output_controller.log_info(info_type="TOTAL_UNIQ_TREC_RELEVANT_DOCS", text=self._uniq_num_trec_rels)
       
        return report_string
