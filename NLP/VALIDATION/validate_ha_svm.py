import numpy as np
import pandas as pd
import json
import re
import pymongo
import enum
from bson import json_util

from main.LOADERS.loader import Loader
from main.LOADERS.module_loader_ha import ModuleLoaderHA
from main.LOADERS.publication_loader_ha import PublicationLoaderHA
from main.MONGODB_PUSHERS.mongodb_pusher import MongoDbPusher

class Dataset(enum.Enum):
    MODULE = 1
    PUBLICATION = 2

class ValidateHASvm():
    """
        Performs SVM model validation for SDGs.
    """

    def __init__(self):
        """
            Initializes total number of SDGs, loader and MongoDB pusher.
        """
        self.num_has = 19
        self.loader = Loader()
        self.mongodb_pusher = MongoDbPusher()

    def module_string_matches_results(self) -> dict:
        """
            Loads string matching keyword counts for modules and stores the results as a dictionary.
        """
        data = ModuleLoaderHA().load_string_matches_results()
        data = json.loads(json_util.dumps(data)) # process mongodb response to a workable dictionary format.
        results = {}  # dictionary with Module_ID and SDG keyword counts.
        
        for module_id, module in data.items():
            ha_dict = module['Related_HA']
            counts = [0] * self.num_has

            for ha, word_found_dict in ha_dict.items():
                ha_match = re.search(r'\d(\d)?', ha)
                ha_num = int(ha_match.group()) if ha_match is not None else self.num_has
                count = len(word_found_dict['Word_Found'])
                counts[ha_num - 1] = count
            
            results[module_id] = counts # add Module_ID with array of SDG keyword counts to results.

        return results

    def publication_string_matches_results(self) -> dict:
        """
            Loads string matching keyword counts for scopus publications and stores the results as a dictionary.
        """
        data = PublicationLoaderHA().load_string_matches_results()
        data = json.loads(json_util.dumps(data)) # process mongodb response to a workable dictionary format.
        results = {} # dictionary with DOI and SDG keyword counts.

        for doi in data:
            ha_dict = data[doi]['Related_HA']
            counts = [0] * self.num_has

            for ha, word_found_dict in ha_dict.items():
                ha_match = re.search(r'\d(\d)?', ha)
                ha_num = int(ha_match.group()) if ha_match is not None else self.num_has
                count = len(word_found_dict['Word_Found'])
                counts[ha_num - 1] = count
            
            results[doi] = counts # add DOI with array of SDG keyword counts to results.

        return results

    def compute_similarity(self, vec_A: np.array, vec_B: np.array) -> float:
        """
            The cosine similarity metric is used to measure how similar a pair of vectors are.
            Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space.
        """
        dot = vec_A.dot(vec_B)
        vec_A_magnitude = np.sqrt(vec_A.dot(vec_A))
        vec_B_magnitude = np.sqrt(vec_B.dot(vec_B))
        return dot / (vec_A_magnitude * vec_B_magnitude)

    def validate(self, dataset: Dataset, svm_predictions: dict) -> dict:
        """
            Validate Svm model results with respect to string matching keyword occurances and store results in a dictionary.
        """
        if dataset == Dataset.MODULE:
            # Load module string matching results.
            model_data = svm_predictions['Module']
            count_data = self.module_string_matches_results()
        else:
            # Load publication string matching results.
            model_data = svm_predictions['Publication']
            count_data = self.publication_string_matches_results()

        e = 0.01 # small offset value added to counts which are zero.
        results = {}

        for key in model_data:
            vec_A = np.array(model_data[key]) # probability distribution of SVM model for SDGs.
            original_counts = count_data[key]
            counts = original_counts.copy()
            
            for i in range(self.num_has):
                if counts[i] == 0:
                    counts[i] = e
            counts_sum_inv = 1.0 / sum(counts)
            vec_B = np.array(counts) * counts_sum_inv # probability predicted by counting keyword occurances for each SDG.

            # Populate validation dictionary with Module_ID, Similarity and SDG keyword counts.
            validation_dict = {}
            validation_dict["Similarity"] = self.compute_similarity(vec_A, vec_B)
            validation_dict["HA_Keyword_Counts"] = original_counts
            results[key] = validation_dict

        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['Similarity'])) # sort dictionary by Similarity.
        return sorted_results

    def run(self) -> None:
        """
            Runs the Lda model validation against string matching keyword occurances for modules and scopus research publications.
        """
        svm_predictions = self.loader.load_svm_prediction_results_ha()

        module_results = self.validate(Dataset.MODULE, svm_predictions)
        scopus_results = self.validate(Dataset.PUBLICATION, svm_predictions)

        # Serialize validation results as JSON file.
        with open('main/NLP/VALIDATION/HA_RESULTS/module_validation.json', 'w') as outfile:
            json.dump(module_results, outfile)
        with open('main/NLP/VALIDATION/HA_RESULTS/scopus_validation.json', 'w') as outfile:
            json.dump(scopus_results, outfile)

        # Push validation results to MongoDB.
        self.mongodb_pusher.module_validation(module_results)
        self.mongodb_pusher.scopus_validation(scopus_results)

        print("Finished.")