import pyodbc
import datetime
import pandas as pd
import numpy as np
import json
import sys
import pymongo
from bson import json_util

from main.LOADERS.module_loader_ha import ModuleLoaderHA
from main.LOADERS.publication_loader_ha import PublicationLoaderHA
from main.NLP.PREPROCESSING.preprocessor import Preprocessor
from main.NLP.PREPROCESSING.module_preprocessor import ModuleCataloguePreprocessor

class HaSvmDataset():
    """
        Creates UCL modules and Scopus research publications dataset with HA tags for training the SVM.
        The dataset is a dataframe with columns {ID, Description, HA} where ID is either Module_ID or DOI.
    """

    def __init__(self):
        """
            Initializes the threshold for tagging a document with an HA, module loader, publication loader and output pickle file.
        """
        self.threshold = 20 # threshold value for tagging a document with an HA, for a probability greater than this value.
        self.module_loader = ModuleLoaderHA()
        self.publication_loader = PublicationLoaderHA()
        self.module_preprocessor = ModuleCataloguePreprocessor()
        self.publication_preprocessor = Preprocessor()
        self.svm_dataset = "main/NLP/SVM/SVM_dataset_ha.pkl"
        self.num_has = 18
        self.df_modules = self.module_loader.load("MAX") # dataframe with columns {Module_ID, Description}.
        self.df_publications = self.publication_loader.load("MAX") # dataframe with columns {DOI, Title, Description}.

    def __progress(self, count: int, total: int, custom_text: str, suffix: str ='') -> None:
        """
            Visualises progress for a process given a current count and a total count.
        """
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))
        percents = round(100.0 * count / float(total), 1)
        bar = '*' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write('[%s] %s%s %s %s\r' %(bar, percents, '%', custom_text, suffix))
        sys.stdout.flush()

    def get_module_description(self, module_id: str):
        """
            Returns the module description for a particular module_id.
        """
        
        # search for row in dataframe by Module_ID.
        df = self.df_modules.loc[self.df_modules["Module_ID"] == module_id]
        return None if len(df) == 0 else df["Module_Description"].values[0]

    def get_publication_description(self, doi: str):
        """
            Returns the publication description for a particular DOI.
        """

        # search for row in dataframe by DOI.
        df = self.df_publications.loc[self.df_publications["DOI"] == doi]
        return None if len(df) == 0 else df["Description"].values[0]
    
    def tag_modules(self) -> pd.DataFrame:
        """
            Returns a dataframe with columns {ID, Description, HA} for each module, where HA is a class tag for training the SVM.
        """
        results = pd.DataFrame(columns=['ID', 'Description', 'HA']) # ID = Module_ID
        data = self.module_loader.load_lda_prediction_results() # loads data from the ModulePrediction table in mongodb.

        doc_topics = data['Document Topics']
        num_modules = len(doc_topics)
        final_data = {}
        counter = 0
        
        for module_id in doc_topics:
            self.__progress(counter, num_modules, "Forming Modules Dataset for SVM...")
            raw_weights = doc_topics[module_id]
            weights = []
            for i in range(len(raw_weights)):
                raw_weights[i] = raw_weights[i].replace('(', '').replace(')', '').replace('%', '').replace(' ', '').split(',')
                ha_num = int(raw_weights[i][0])
                try:
                    w = float(raw_weights[i][1])
                except:
                    w = 0.0
                weights.append((ha_num, w))

            ha_weight_max = max(weights, key=lambda x: x[1]) # get tuple (sdg, weight) with the maximum weight.
            
            description = self.get_module_description(module_id)
            description = "" if description is None else self.module_preprocessor.preprocess(description)

            if description != "":
                if ha_weight_max[1] >= self.threshold:
                    # Set SDG tag of module to the SDG which has the maximum weight if its greater than the threshold value.
                    row_df = pd.DataFrame([[module_id, description, ha_weight_max[0]]], columns=results.columns)
                else:
                    # Set SDG tag of module to None if the maximum weight is less than the threshold value.
                    row_df = pd.DataFrame([[module_id, description, None]], columns=results.columns)
                
                results = results.append(row_df, verify_integrity=True, ignore_index=True)
            
            counter += 1
                
        return results

    def tag_publications(self) -> pd.DataFrame:
        """
            Returns a dataframe with columns {ID, Description, SDG} for each publication, where SDG is a class tag for training the SVM.
        """
        results = pd.DataFrame(columns=['ID', 'Description', 'HA']) # ID = DOI
        data = self.publication_loader.load_lda_prediction_results() # loads data from the PublicationPrediction table in mongodb.

        num_publications = len(data)
        final_data = {}
        counter = 0

        for doi in data:
            self.__progress(counter, num_publications,"Forming Publications Dataset for SVM...")
            raw_weights = data[doi]
            weights = [0] * self.num_has
            for i in range(self.num_has):
                ha_num = str(i + 1)
                try:
                    w = float(raw_weights[ha_num]) * 100.0 # convert probabilities in the range [0,1] to percentages.
                except:
                    w = 0.0
                weights[i] = w

            weights = np.asarray(weights)
            ha_max = weights.argmax() + 1 # gets SDG corresponding to the maximum weight.
            ha_weight_max = weights[ha_max - 1] # gets the maximum weight.

            description = self.get_publication_description(doi)
            description = "" if description is None else self.publication_preprocessor.preprocess(description)

            if description != "":
                if ha_weight_max >= self.threshold:
                    # Set SDG tag of publication to the SDG which has the maximum weight if its greater than the threshold value.
                    row_df = pd.DataFrame([[doi, description, ha_max]], columns=results.columns)
                else:
                    # Set SDG tag of module to None if the maximum weight is less than the threshold value.
                    row_df = pd.DataFrame([[doi, description, None]], columns=results.columns)

                results = results.append(row_df, verify_integrity=True, ignore_index=True)

            counter += 1

        return results

    def run(self, modules: bool, publications: bool) -> None:
        """
            Tags the modules and/or publications with their most related SDG, if related to one at all, and combines them into a single dataframe.
            Serializes the resulting dataframe as a pickle file.
        """
        df = pd.DataFrame() # column format of dataframe is {ID, Description, SDG} where ID is either Module_ID or DOI.
        if modules:
            df = df.append(self.tag_modules(), ignore_index=True)
        if publications:
            df = df.append(self.tag_publications(), ignore_index=True)

        df = df.reset_index()
        df.to_pickle(self.svm_dataset)