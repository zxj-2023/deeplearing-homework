import spacy
from collections import defaultdict
import pdb
import logging


class SpacyNER:
    def __init__(self,spacy_model):
        self.spacy_model = spacy.load(spacy_model)
        self.logger = logging.getLogger(__name__)

    def batch_ner(self, hash_id_to_passage, max_workers):
        passage_items = list(hash_id_to_passage.items())
        passage_hash_ids = [item[0] for item in passage_items]
        passage_list = [item[1] for item in passage_items]
        batch_size = max(1, len(passage_list) // max_workers)
        self.logger.info("spaCy NER batch_size=%d for %d passages", batch_size, len(passage_list))
        docs_list = self.spacy_model.pipe(passage_list, batch_size=batch_size)
        passage_hash_id_to_entities = {}
        sentence_to_entities = defaultdict(list)
        total = len(passage_list)
        for idx, doc in enumerate(docs_list, start=1):
            if idx % 50 == 0 or idx == total:
                self.logger.info("spaCy NER progress: %d/%d passages", idx, total)
            passage_hash_id = passage_hash_ids[idx - 1]
            single_passage_hash_id_to_entities,single_sentence_to_entities = self.extract_entities_sentences(doc,passage_hash_id)
            passage_hash_id_to_entities.update(single_passage_hash_id_to_entities)
            for sent, ents in single_sentence_to_entities.items():
                for e in ents:
                    if e not in sentence_to_entities[sent]:
                        sentence_to_entities[sent].append(e)
        return passage_hash_id_to_entities,sentence_to_entities
            
    def extract_entities_sentences(self, doc,passage_hash_id):
        sentence_to_entities = defaultdict(list)
        unique_entities = set()
        passage_hash_id_to_entities = {}
        #pdb.set_trace()
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            sent_text = ent.sent.text
            ent_text = ent.text
            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)
            unique_entities.add(ent_text)
        passage_hash_id_to_entities[passage_hash_id] = list(unique_entities)
        return passage_hash_id_to_entities,sentence_to_entities

    def question_ner(self, question: str):
        doc = self.spacy_model(question)
        question_entities = set()
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            question_entities.add(ent.text.lower())
        return question_entities
